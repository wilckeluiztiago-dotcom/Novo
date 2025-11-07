# =====================================================================
# Previsão de Desemprego com Equações Estocásticas Avançadas (R base)
# - Modelo Estado-Espaço com Volatilidade Estocástica (SV)
# - Estimação por Máxima Verossimilhança via Filtro de Partículas (PF)
# - Previsão h-passos à frente usando partículas finais
# Autor: Luiz Tiago Wilcke (LT)
# =====================================================================

set.seed(123)

# ============================
# 0) Configurações de dados
# ============================
usar_csv        <- FALSE
caminho_csv     <- "desemprego.csv"     # CSV: colunas 'data,desemprego'
n_sintetico     <- 220                  # tamanho da série se não houver CSV
horizonte_prev  <- 12                   # passos à frente
num_particulas  <- 600                  # partículas do PF (aumente p/ mais precisão)

# ===========================================================
# 1) Leitura/Simulação da série de desemprego (R-base)
# ===========================================================
if (usar_csv && file.exists(caminho_csv)) {
  dados_brutos <- read.csv(caminho_csv, stringsAsFactors = FALSE)
  if (!all(c("data","desemprego") %in% names(dados_brutos))) {
    stop("CSV deve conter colunas: data, desemprego")
  }
  dados_brutos$data <- as.Date(dados_brutos$data)
  y     <- as.numeric(dados_brutos$desemprego)
  datas <- dados_brutos$data
  nome_serie <- "Desemprego (dados reais)"
} else {
  n       <- n_sintetico
  alpha   <- 0.02     # drift do nível
  rho     <- 0.98     # persistência do nível
  mu      <- -2.0     # média da log-variância
  phi     <- 0.95     # persistência da log-variância
  sigma_h <- 0.25     # vol. do processo da log-variância
  r       <- 0.04     # variância do ruído de medição
  
  x <- numeric(n)
  h <- numeric(n)
  y <- numeric(n)
  
  x[1] <- 8.5
  h[1] <- mu
  for (t in 2:n) {
    h[t] <- mu + phi * (h[t-1] - mu) + sigma_h * rnorm(1)
    x[t] <- alpha + rho * x[t-1] + exp(0.5 * h[t]) * rnorm(1)
  }
  y <- x + rnorm(n, sd = sqrt(r))
  datas <- seq(as.Date("2010-01-01"), by = "month", length.out = n)
  nome_serie <- "Desemprego (sintético)"
}
n <- length(y)

# ===========================================================
# 2) Utilidades (R-base)
# ===========================================================
tanh_inv <- function(x) 0.5*log((1+x)/(1-x))  # arctanh p/ inicializar parâmetros

reamostrar_sistematico <- function(pesos_norm) {
  N  <- length(pesos_norm)
  u0 <- runif(1, 0, 1/N)
  u  <- u0 + (0:(N-1))/N
  cpesos <- cumsum(pesos_norm)
  idx <- integer(N)
  j <- 1L
  for (i in 1:N) {
    while (u[i] > cpesos[j]) j <- j + 1L
    idx[i] <- j
  }
  idx
}

# ===========================================================
# 3) Filtro de Partículas para modelo SV (estado: x_t, h_t)
#     Obs: y_t = x_t + v_t, v_t ~ N(0, r)
#     Transições:
#        h_t = mu + phi*(h_{t-1}-mu) + sigma_h*e_t
#        x_t = alpha + rho*x_{t-1} + exp(0.5*h_t)*z_t
# ===========================================================
filtro_particulas <- function(y, params, N = num_particulas,
                              manter_quantis = TRUE,
                              quantis = c(0.05, 0.25, 0.50, 0.75, 0.95)) {
  alpha   <- params$alpha
  rho     <- params$rho
  mu      <- params$mu
  phi     <- params$phi
  sigma_h <- params$sigma_h
  r       <- params$r
  
  Tn <- length(y)
  
  # Inicialização heurística
  x_part <- rnorm(N, mean = y[1], sd = 0.5)
  h_part <- rnorm(N, mean = mu,   sd = 1.0)
  
  pesos_log <- rep(-log(N), N)   # pesos em log
  ll_total  <- 0
  
  media_filtrada <- numeric(Tn)
  var_filtrada   <- numeric(Tn)
  h_media        <- numeric(Tn)
  ess            <- numeric(Tn)
  
  # Quantis
  rot_quantis <- paste0("q", sprintf("%03d", as.integer(quantis * 100)))
  quantis_x <- matrix(NA_real_, nrow = Tn, ncol = length(quantis))
  colnames(quantis_x) <- rot_quantis
  quantis_h <- matrix(NA_real_, nrow = Tn, ncol = length(quantis))
  colnames(quantis_h) <- rot_quantis
  
  # Guardar partículas/weights finais
  x_part_final <- NULL
  h_part_final <- NULL
  w_final      <- NULL
  
  for (t in 1:Tn) {
    # Previsão
    if (t > 1) {
      h_part <- mu + phi * (h_part - mu) + sigma_h * rnorm(N)
      x_part <- alpha + rho * x_part + exp(0.5 * h_part) * rnorm(N)
    }
    
    # Atualização (likelihood)
    dens <- dnorm(y[t], mean = x_part, sd = sqrt(r), log = TRUE)
    pesos_log <- pesos_log + dens
    
    # Normalização estável (log-sum-exp)
    mmax <- max(pesos_log)
    w    <- exp(pesos_log - mmax)
    w    <- w / sum(w)
    
    # Contribuição da log-verossimilhança
    ll_total <- ll_total + (mmax + log(sum(exp(pesos_log - mmax))) - log(N))
    
    # Momentos filtrados
    media_filtrada[t] <- sum(w * x_part)
    var_filtrada[t]   <- sum(w * (x_part - media_filtrada[t])^2)
    h_media[t]        <- sum(w * h_part)
    ess[t]            <- 1 / sum(w^2)
    
    if (manter_quantis) {
      quantis_x[t, ] <- quantile(x_part, probs = quantis, names = FALSE, type = 7)
      quantis_h[t, ] <- quantile(h_part, probs = quantis, names = FALSE, type = 7)
    }
    
    # Reamostragem
    if (ess[t] < 0.5 * N) {
      idx <- reamostrar_sistematico(w)
      x_part   <- x_part[idx]
      h_part   <- h_part[idx]
      pesos_log <- rep(-log(N), N)
    } else {
      pesos_log <- log(w)  # mantém coerência numérica
    }
    
    # Se for o último tempo, guarda partículas e pesos normalizados
    if (t == Tn) {
      # pesos normalizados do passo t (w)
      w_final      <- w
      x_part_final <- x_part
      h_part_final <- h_part
    }
  }
  
  list(
    ll = ll_total,
    media_filtrada = media_filtrada,
    var_filtrada   = var_filtrada,
    h_media        = h_media,
    ess            = ess,
    quantis_x      = quantis_x,
    quantis_h      = quantis_h,
    x_part_final   = x_part_final,
    h_part_final   = h_part_final,
    w_final        = w_final
  )
}

# ===========================================================
# 4) Envoltório de verossimilhança para 'optim' (MLE)
# ===========================================================
construir_params <- function(theta) {
  list(
    alpha   = theta[1],             # R
    rho     = tanh(theta[2]),       # (-1,1)
    mu      = theta[3],             # R
    phi     = 0.999 * tanh(theta[4]), # (-0.999,0.999)
    sigma_h = exp(theta[5]),        # (0,inf)
    r       = exp(theta[6]) + 1e-6  # (1e-6,inf)
  )
}

logverossimilhanca_pf <- function(theta, y, N = num_particulas) {
  p <- construir_params(theta)
  if (abs(p$rho) >= 0.9999 || abs(p$phi) >= 0.9999 || p$sigma_h <= 0 || p$r <= 0) {
    return(1e6)
  }
  res <- filtro_particulas(y, p, N = N, manter_quantis = FALSE)
  -(res$ll)  # 'optim' minimiza
}

# Chutes iniciais
theta_ini <- c(
  0.01,           # alpha
  tanh_inv(0.95), # rho ~ 0.95
  -2.0,           # mu
  tanh_inv(0.9),  # phi ~ 0.9
  log(0.2),       # sigma_h ~ 0.2
  log(0.05)       # r ~ 0.05
)

cat("=== Otimizando parâmetros por MLE (PF) ===\n")
t0 <- proc.time()[3]
ajuste <- optim(theta_ini, logverossimilhanca_pf, y = y, N = num_particulas,
                method = "Nelder-Mead", control = list(maxit = 350, trace = 1, REPORT = 10))
t1 <- proc.time()[3]
cat(sprintf("Concluído em %.2f s\n", t1 - t0))

parametros_est <- construir_params(ajuste$par)
print(parametros_est)

# Rodada final do PF com quantis + partículas finais
pf_final <- filtro_particulas(y, parametros_est, N = num_particulas,
                              manter_quantis = TRUE)

# ===========================================================
# 5) Previsão h-passos usando partículas finais
#    (amostra ponderada das partículas finais -> forward simulation)
# ===========================================================
simular_previsao_com_particulas <- function(x_final, h_final, w_final, params,
                                            H = horizonte_prev, S = 3000) {
  # amostra S índices das partículas finais proporcionalmente a w_final
  # para inicializar múltiplos cenários de previsão
  idx_amostra <- sample.int(length(x_final), size = S, replace = TRUE, prob = w_final)
  x_cur <- x_final[idx_amostra]
  h_cur <- h_final[idx_amostra]
  
  alpha   <- params$alpha
  rho     <- params$rho
  mu      <- params$mu
  phi     <- params$phi
  sigma_h <- params$sigma_h
  r       <- params$r
  
  trajetorias <- matrix(NA_real_, nrow = H, ncol = S)
  for (t in 1:H) {
    h_cur <- mu + phi * (h_cur - mu) + sigma_h * rnorm(S)
    x_cur <- alpha + rho * x_cur + exp(0.5 * h_cur) * rnorm(S)
    y_sim <- x_cur + rnorm(S, sd = sqrt(r))
    trajetorias[t, ] <- y_sim
  }
  trajetorias
}

traj <- simular_previsao_com_particulas(
  pf_final$x_part_final, pf_final$h_part_final, pf_final$w_final,
  parametros_est, H = horizonte_prev, S = 3000
)

datas_fut <- seq(tail(datas, 1) + 30, by = "month", length.out = horizonte_prev)

probs_fan <- c(0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
fan <- t(apply(traj, 1, quantile, probs = probs_fan, names = FALSE))

# ===========================================================
# 6) Diagnósticos e Gráficos (muitos)
# ===========================================================
par_backup <- par(no.readonly = TRUE)

# Série observada
dev.new(); par(mfrow = c(1,1), mar=c(4,4,2,1))
plot(datas, y, type = "l", lwd = 2, main = paste0(nome_serie, " — Observada"),
     xlab = "Tempo", ylab = "Taxa de desemprego")

# Nível filtrado + bandas (quantis)
dev.new(); par(mfrow = c(1,1), mar=c(4,4,2,1))
plot(datas, y, type = "l", col = "gray60", lwd = 1,
     main = "Nível oculto (x_t) — média filtrada e bandas (quantis)",
     xlab = "Tempo", ylab = "Taxa (%)")
lines(datas, pf_final$media_filtrada, lwd = 2)
qs <- pf_final$quantis_x
if (!any(is.na(qs))) {
  q05 <- qs[, "q005"]; q25 <- qs[, "q025"]; q50 <- qs[, "q050"]
  q75 <- qs[, "q075"]; q95 <- qs[, "q095"]
  polygon(c(datas, rev(datas)), c(q95, rev(q05)),
          border = NA, col = rgb(0.7,0.7,0.7,0.20))
  polygon(c(datas, rev(datas)), c(q75, rev(q25)),
          border = NA, col = rgb(0.4,0.4,0.4,0.20))
  lines(datas, q50, lwd = 1, lty = 2)
}

# Log-variância e desvio-padrão condicional médio
dev.new(); par(mfrow = c(2,1), mar=c(4,4,2,1))
plot(datas, pf_final$h_media, type="l", lwd=2,
     main="Log-variância média filtrada (h_t)", xlab="Tempo", ylab="h_t")
sigma_t <- exp(0.5 * pf_final$h_media)
plot(datas, sigma_t, type="l", lwd=2,
     main="Desvio-padrão condicional médio (exp(h_t/2))",
     xlab="Tempo", ylab="sigma_t")

# ESS ao longo do tempo
dev.new(); par(mfrow = c(1,1), mar=c(4,4,2,1))
plot(datas, pf_final$ess, type="l", lwd=2,
     main="Tamanho Efetivo da Amostra (ESS)", xlab="Tempo", ylab="ESS")

# Resíduos e diagnósticos
residuos <- y - pf_final$media_filtrada
dev.new(); par(mfrow = c(2,2), mar=c(4,4,2,1))
plot(datas, residuos, type="l", main="Resíduos no tempo", xlab="Tempo", ylab="resíduo")
hist(residuos, breaks = "FD", main="Histograma dos resíduos", xlab="resíduo")
qqnorm(residuos, main="QQ-plot dos resíduos"); qqline(residuos)
acf(residuos, main="ACF dos resíduos")
dev.new(); par(mfrow = c(1,1), mar=c(4,4,2,1)); pacf(residuos, main="PACF dos resíduos")

# Fase: nível filtrado x sigma_t
dev.new(); par(mfrow = c(1,1), mar=c(4,4,2,1))
plot(pf_final$media_filtrada, sigma_t, pch=19, cex=0.6,
     main="Fase: nível filtrado × desvio-padrão condicional",
     xlab="Nível (média filtrada)", ylab="sigma_t")

# Observado vs média filtrada
dev.new(); par(mfrow = c(1,1), mar=c(4,4,2,1))
plot(datas, y, type="l", col="gray60", lwd=1,
     main="Observado vs Nível (média filtrada)", xlab="Tempo", ylab="Taxa (%)")
lines(datas, pf_final$media_filtrada, lwd=2)

# Fan chart de previsão
dev.new(); par(mar=c(4,4,2,1))
plot(c(datas, datas_fut), c(y, rep(NA, horizonte_prev)), type="n",
     xlab="Tempo", ylab="Taxa (%)", main="Previsão (fan chart)")
lines(datas, y, col="gray50")
faixa <- function(pinf, psup, col) {
  polygon(c(datas_fut, rev(datas_fut)),
          c(fan[, which(probs_fan==psup)],
            rev(fan[, which(probs_fan==pinf)])),
          border = NA, col = col)
}
faixa(0.05, 0.95, rgb(0.2,0.2,0.8,0.12))
faixa(0.10, 0.90, rgb(0.2,0.2,0.8,0.18))
faixa(0.25, 0.75, rgb(0.2,0.2,0.8,0.30))
lines(datas_fut, fan[, which(probs_fan==0.50)], lwd=2, lty=2)

# Série completa com mediana prevista
dev.new(); par(mar=c(4,4,2,1))
plot(c(datas, datas_fut),
     c(y, fan[, which(probs_fan==0.50)]), type="l", lwd=2,
     xlab="Tempo", ylab="Taxa (%)",
     main="Observado + Mediana da Previsão")
lines(datas, y, col="gray60")

# ===========================================================
# 7) Métricas/resumo no console
# ===========================================================
cat("\n================ RESULTADOS ================\n")
cat(sprintf("Log-verossimilhança (PF): %.3f\n", -ajuste$value))
cat("Parâmetros estimados:\n"); print(parametros_est)
rmse <- sqrt(mean((y - pf_final$media_filtrada)^2))
cat(sprintf("RMSE (observado vs média filtrada): %.4f\n", rmse))

# Restaurar layout
par(par_backup)

