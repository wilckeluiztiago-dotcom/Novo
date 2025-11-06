# ============================================================
# Brent (BZ=F) — Modelo Schwartz–Smith (2 fatores)
# MLE + Kalman estável + Previsões (faixa 90%)
#  Autor: Luiz Tiago Wilcke (LT)
# ============================================================

suppressPackageStartupMessages({
  pkgs <- c("dplyr","ggplot2","lubridate","readr","scales","tibble","quantmod","MASS")
  novo <- setdiff(pkgs, rownames(installed.packages()))
  if(length(novo)) install.packages(novo, quiet = TRUE)
  lapply(pkgs, library, character.only = TRUE)
})

set.seed(42)

# ---------------- Constantes de robustez ----------------
EPS_VAR    <- 1e-8   # piso p/ variâncias (ridge)
EPS_SIGMA  <- 1e-3   # piso p/ desvios-padrão
EPS_F      <- 1e-8   # piso p/ variância da inovação
BIG_PENAL  <- 1e6    # penalização se não-finito
DELTA_DIA  <- 1/252  # ~ dia útil em anos

# ---------------- 1) Dados ----------------
usar_yahoo  <- TRUE
caminho_csv <- NULL          # ex.: "brent.csv" com colunas data, preco
data_min    <- "2010-01-01"

carregar_dados <- function(usar_yahoo = TRUE, caminho_csv = NULL, desde = "2010-01-01"){
  if(!usar_yahoo && !is.null(caminho_csv) && file.exists(caminho_csv)){
    df <- readr::read_csv(caminho_csv, show_col_types = FALSE) |>
      transmute(data = as.Date(data), preco = as.numeric(preco))
  } else {
    getSymbols("BZ=F", src = "yahoo", from = desde, auto.assign = TRUE, warnings = FALSE)
    precos <- Cl(get("BZ=F"))
    df <- tibble(data = as.Date(index(precos)),
                 preco = as.numeric(precos))
  }
  df |>
    distinct(data, .keep_all = TRUE) |>
    arrange(data) |>
    filter(is.finite(preco), preco > 0) |>
    mutate(log_preco = log(preco)) |>
    filter(is.finite(log_preco))
}

dados <- carregar_dados(usar_yahoo, caminho_csv, data_min)
stopifnot(nrow(dados) > 50)
y     <- dados$log_preco
delta <- DELTA_DIA

# ---------------- 2) Matrizes do modelo ----------------
# Estado: s_t = [xi_t, chi_t]^T
# Observação: y_t = [1 1] s_t + eps_t
construir_matrizes <- function(theta, delta){
  mu        <- theta["mu"]
  kappa     <- max(theta["kappa"], 1e-6)
  sigma_xi  <- max(theta["sigma_xi"], EPS_SIGMA)
  sigma_chi <- max(theta["sigma_chi"], EPS_SIGMA)
  sigma_eps <- max(theta["sigma_eps"], EPS_SIGMA)
  
  phi     <- exp(-kappa*delta)
  var_xi  <- (sigma_xi^2) * delta
  var_chi <- (sigma_chi^2) * (1 - exp(-2*kappa*delta)) / (2*kappa)
  
  var_xi  <- max(var_xi,  EPS_VAR)
  var_chi <- max(var_chi, EPS_VAR)
  var_eps <- max(sigma_eps^2, EPS_VAR)
  
  Tm   <- matrix(c(1, 0,
                   0, phi), 2, 2, byrow = TRUE)
  cvec <- c(mu*delta, 0)
  Qm   <- matrix(c(var_xi, 0,
                   0, var_chi), 2, 2, byrow = TRUE)
  Zm   <- matrix(c(1, 1), 1, 2)
  Hm   <- matrix(var_eps, 1, 1)
  
  list(Tm=Tm, cvec=cvec, Qm=Qm, Zm=Zm, Hm=Hm,
       mu=mu, kappa=kappa, sigma_xi=sigma_xi, sigma_chi=sigma_chi,
       sigma_eps=sigma_eps, phi=phi)
}

# ---------------- 3) Kalman — loglik estabilizada ----------------
kalman_loglik_estavel <- function(y, theta, delta){
  mats <- construir_matrizes(theta, delta)
  Tm <- mats$Tm; Qm <- mats$Qm; Zm <- mats$Zm; Hm <- mats$Hm; cvec <- mats$cvec
  n <- length(y)
  
  a <- matrix(c(y[1], 0), nrow = 2)   # priori finita
  P <- diag(2)*5
  
  ll <- 0.0
  for(t in 1:n){
    y_pred <- drop(Zm %*% a)                        # escalar
    Fm     <- drop(Zm %*% P %*% t(Zm)) + drop(Hm)   # 1x1
    Fm     <- max(as.numeric(Fm), EPS_F)
    
    v      <- y[t] - y_pred
    invF   <- 1.0 / Fm
    
    incr <- -0.5 * (log(2*pi) + log(Fm) + (v*v)*invF)
    if(!is.finite(incr)) return(-BIG_PENAL)
    
    K  <- (P %*% t(Zm)) * invF
    a  <- a + K * v
    P  <- P - K %*% Zm %*% P
    
    a  <- Tm %*% a + matrix(cvec, ncol = 1)
    P  <- Tm %*% P %*% t(Tm) + Qm
    
    ll <- ll + incr
    if(!is.finite(ll)) return(-BIG_PENAL)
  }
  drop(ll)
}

# ---------------- 4) Objetivo (reparam log) ----------------
# p_raw = (mu, log_kappa, log_sigma_xi, log_sigma_chi, log_sigma_eps)
negloglik_wrapped <- function(p_raw, y, delta){
  theta <- c(
    mu        = p_raw[1],
    kappa     = exp(p_raw[2]),
    sigma_xi  = exp(p_raw[3]),
    sigma_chi = exp(p_raw[4]),
    sigma_eps = exp(p_raw[5])
  )
  val <- try(kalman_loglik_estavel(y, theta, delta), silent = TRUE)
  if(inherits(val, "try-error") || !is.finite(val)) return(BIG_PENAL)
  -val
}

# ---------------- 5) Chute inicial + jitter ----------------
parametros_padrao <- c(
  mu = 0.00,
  kappa = 1.0,
  sigma_xi = 0.25,
  sigma_chi = 0.6,
  sigma_eps = 0.03
)

p0 <- c(
  mu              = parametros_padrao["mu"],
  log_kappa       = log(parametros_padrao["kappa"]),
  log_sigma_xi    = log(parametros_padrao["sigma_xi"]),
  log_sigma_chi   = log(parametros_padrao["sigma_chi"]),
  log_sigma_eps   = log(parametros_padrao["sigma_eps"])
)

seguro_val <- negloglik_wrapped(p0, y, delta)
if(!is.finite(seguro_val) || seguro_val >= BIG_PENAL){
  p0 <- p0 + rnorm(length(p0), 0, 0.25)
}

# ---------------- 6) Otimização (MLE) ----------------
opt <- optim(
  par = p0,
  fn  = negloglik_wrapped,
  y = y, delta = delta,
  method = "BFGS",
  control = list(maxit = 2000, reltol = 1e-9)
)

theta_hat <- c(
  mu        = opt$par[1],
  kappa     = exp(opt$par[2]),
  sigma_xi  = exp(opt$par[3]),
  sigma_chi = exp(opt$par[4]),
  sigma_eps = exp(opt$par[5])
)

# ---------------- 7) Filtro + Suavização (RTS) ----------------
kalman_filtrar_suavizar <- function(y, theta, delta){
  mats <- construir_matrizes(theta, delta)
  Tm <- mats$Tm; Qm <- mats$Qm; Zm <- mats$Zm; Hm <- mats$Hm; cvec <- mats$cvec
  n <- length(y)
  
  a_f <- array(NA_real_, dim = c(2, n))
  P_f <- array(NA_real_, dim = c(2, 2, n))
  a_p <- array(NA_real_, dim = c(2, n))
  P_p <- array(NA_real_, dim = c(2, 2, n))
  
  a <- matrix(c(y[1], 0), nrow = 2)
  P <- diag(2)*5
  
  for(t in 1:n){
    Fm <- drop(Zm %*% P %*% t(Zm)) + drop(Hm)
    Fm <- max(as.numeric(Fm), EPS_F)
    v  <- y[t] - drop(Zm %*% a)
    invF <- 1.0/Fm
    K  <- (P %*% t(Zm)) * invF
    a  <- a + K * v
    P  <- P - K %*% Zm %*% P
    
    a_f[,t]  <- drop(a)
    P_f[,,t] <- P
    
    a <- Tm %*% a + matrix(cvec, ncol = 1)
    P <- Tm %*% P %*% t(Tm) + Qm
    
    a_p[,t]  <- drop(a)
    P_p[,,t] <- P
  }
  
  # Rauch–Tung–Striebel
  a_s <- array(NA_real_, dim = c(2, n))
  P_s <- array(NA_real_, dim = c(2, 2, n))
  a_s[,n]  <- a_f[,n]
  P_s[,,n] <- P_f[,,n]
  
  for(t in (n-1):1){
    invPp <- solve(P_p[,,t+1])
    C <- P_f[,,t] %*% t(Tm) %*% invPp
    a_s[,t]  <- a_f[,t] + C %*% (a_s[,t+1] - a_p[,t+1])
    P_s[,,t] <- P_f[,,t] + C %*% (P_s[,,t+1] - P_p[,,t+1]) %*% t(C)
  }
  list(a_f=a_f, P_f=P_f, a_s=a_s, P_s=P_s)
}

suave <- kalman_filtrar_suavizar(y, theta_hat, delta)

# ---------------- 8) Previsões por simulação (robustas) ----------------
prever_brent <- function(dados, suave, theta, delta, passos = 90, n_paths = 2000){
  mats <- construir_matrizes(theta, delta)
  Tm <- mats$Tm; Qm <- mats$Qm; Zm <- mats$Zm; cvec <- mats$cvec
  
  est_final <- suave$a_s[, nrow(dados)]
  xi0  <- est_final[1]; chi0 <- est_final[2]
  
  Qm[1,1] <- max(Qm[1,1], EPS_VAR)
  Qm[2,2] <- max(Qm[2,2], EPS_VAR)
  
  L <- try(chol(Qm), silent = TRUE)
  if(inherits(L, "try-error")) L <- diag(sqrt(pmax(diag(Qm), EPS_VAR)), 2)
  
  sim_logP <- matrix(NA_real_, nrow = passos, ncol = n_paths)
  
  for(j in 1:n_paths){
    xi  <- xi0; chi <- chi0
    for(h in 1:passos){
      inov <- as.numeric(L %*% rnorm(2))
      estado <- Tm %*% c(xi,chi) + cvec + inov
      xi <- estado[1]; chi <- estado[2]
      ysim <- drop(Zm %*% c(xi,chi))
      sim_logP[h, j] <- ifelse(is.finite(ysim), ysim, NA_real_)
    }
  }
  
  mediana_log <- apply(sim_logP, 1, median,   na.rm = TRUE)
  p05_log     <- apply(sim_logP, 1, quantile, probs = 0.05, na.rm = TRUE, names = FALSE, type = 7)
  p95_log     <- apply(sim_logP, 1, quantile, probs = 0.95, na.rm = TRUE, names = FALSE, type = 7)
  
  tibble(
    horizonte   = 1:passos,
    log_mediana = ifelse(is.finite(mediana_log), mediana_log, NA_real_),
    log_p05     = ifelse(is.finite(p05_log),     p05_log,     NA_real_),
    log_p95     = ifelse(is.finite(p95_log),     p95_log,     NA_real_)
  ) |>
    mutate(
      preco_mediana = exp(log_mediana),
      preco_p05     = exp(log_p05),
      preco_p95     = exp(log_p95)
    )
}

passos <- 90
prev <- prever_brent(dados, suave, theta_hat, delta, passos = passos, n_paths = 2000)

# ---------------- 9) Gráficos ----------------
ultimo_preco <- utils::tail(dados$preco, 1)

g1 <- ggplot(dados, aes(data, preco)) +
  geom_line(linewidth = 0.7) +
  scale_y_continuous(labels = scales::dollar_format(prefix = "US$ ")) +
  labs(title = "Brent — Série Histórica",
       x = "Data", y = "Preço (USD/barril)") +
  theme_minimal(base_size = 13)

comp <- tibble(
  data = dados$data,
  nivel_longo_prazo = suave$a_s[1,],
  componente_transitorio = suave$a_s[2,],
  log_preco_suav = nivel_longo_prazo + componente_transitorio,
  preco_suav = exp(log_preco_suav)
)

g2 <- ggplot(comp, aes(data)) +
  geom_line(aes(y = exp(nivel_longo_prazo)), linewidth = 0.85) +
  geom_line(aes(y = preco_suav), linewidth = 0.65, alpha = 0.8) +
  scale_y_continuous(labels = scales::dollar_format(prefix = "US$ ")) +
  labs(title = "Decomposição — nível (exp(ξ_t)) e preço suavizado",
       x = "Data", y = "Preço (USD/barril)") +
  theme_minimal(base_size = 13)

# calendário de dias úteis (seg–sex), sem 'last'
datas_prev <- {
  d0 <- max(dados$data)
  cand <- seq(d0 + 1, by = "day", length.out = passos*3)
  cand[lubridate::wday(cand, week_start = 1) <= 5][1:passos]
}

faixa <- tibble(
  data = datas_prev,
  mediana = prev$preco_mediana,
  p05 = prev$preco_p05,
  p95 = prev$preco_p95
) |>
  filter(is.finite(mediana) | is.finite(p05) | is.finite(p95))

g3 <- ggplot() +
  geom_line(data = utils::tail(dados, 250), aes(data, preco), linewidth = 0.7) +   # <-- tail() base R
  geom_ribbon(data = faixa, aes(x = data, ymin = p05, ymax = p95), alpha = 0.18) +
  geom_line(data = faixa, aes(data, mediana), linewidth = 0.95) +
  scale_y_continuous(labels = scales::dollar_format(prefix = "US$ ")) +
  labs(title = "Previsões do Brent — mediana e intervalo 90%",
       x = "Data", y = "Preço (USD/barril)",
       caption = sprintf("Último preço: ~US$ %.2f | Horizonte: %d dias úteis", ultimo_preco, passos)) +
  theme_minimal(base_size = 13)

print(g1); print(g2); print(g3)

# ---------------- 10) Sumário ----------------
cat("\n=== Parâmetros MLE (escala anual) ===\n")
print(round(theta_hat, 6))
meia_vida_dias_uteis <- log(2)/theta_hat["kappa"]*252
cat(sprintf("Meia-vida do choque transitório ≈ %.2f dias úteis\n", meia_vida_dias_uteis))

# ---------------- 11) Função utilitária ----------------
prever_percentual <- function(prev, preco_atual){
  tibble(
    horizonte = prev$horizonte,
    mediana = (prev$preco_mediana/preco_atual - 1)*100,
    p05     = (prev$preco_p05/preco_atual - 1)*100,
    p95     = (prev$preco_p95/preco_atual - 1)*100
  )
}
print(utils::head(prever_percentual(prev, ultimo_preco), 10))





