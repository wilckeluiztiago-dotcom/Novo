# ============================================================
# Preço do Pão Francês — SDE + Volatilidade Estocástica + Filtro de Partículas
# Autor: Luiz Tiago Wilcke (LT)

suppressPackageStartupMessages({
  pkgs <- c("dplyr","ggplot2","readr","lubridate","tidyr","scales","rlang","stats")
  novo <- setdiff(pkgs, rownames(installed.packages()))
  if(length(novo)) install.packages(novo)
  lapply(pkgs, library, character.only = TRUE)
})

set.seed(123)


# Se tiver CSV REAL, aponte aqui:
# Esperado: colunas -> data, preco_pao, (opcional) farinha, cambio, diesel, energia, ipca, salario_minimo
caminho_csv <- NULL  # ex: "precos_pao_joinville.csv"

if(!is.null(caminho_csv) && file.exists(caminho_csv)){
  dados_brutos <- readr::read_csv(caminho_csv, show_col_types = FALSE)
  mensagem_dados <- "Dados reais carregados do CSV."
} else {
  # ---- Simulação didática (semanal, 5 anos) ----
  n <- 52*5
  data_ini <- as.Date("2020-01-05")
  datas <- data_ini + 7*(0:(n-1))
  
  farinha <- cumsum(rnorm(n, 0.02, 0.5)) + 100
  cambio  <- 5.0 + 0.1*sin(2*pi*(1:n)/52) + rnorm(n,0,0.2)
  diesel  <- cumsum(rnorm(n, 0.01, 0.3)) + 4
  energia <- cumsum(rnorm(n, 0.005, 0.1)) + 100
  ipca    <- cumsum(abs(rnorm(n, 0.07, 0.05)))
  sal_min <- 1200 + cumsum(pmax(0, rnorm(n, 1, 5)))
  
  alpha <- 0.001
  beta  <- c(farinha= 0.0008, cambio=0.03, diesel=0.015, energia=0.0005,
             ipca=0.0002, salario_minimo=0.00005)
  
  mu_v <- -6.0; phi_v <- 0.95; sigma_v <- 0.15
  sigma_med <- 0.03
  
  x <- numeric(n)
  v <- numeric(n)
  z <- scale(cbind(farinha, cambio, diesel, energia, ipca, sal_min)) # padroniza pra simulação
  
  v[1] <- mu_v
  x[1] <- log(5.00)
  
  for(t in 2:n){
    v[t] <- mu_v + phi_v*(v[t-1]-mu_v) + sigma_v*rnorm(1)
    drift_t <- alpha + as.numeric(z[t,] %*% beta[c("farinha","cambio","diesel","energia","ipca","salario_minimo")])
    x[t] <- x[t-1] + drift_t + exp(0.5*v[t-1])*rnorm(1)
  }
  y <- x + rnorm(n, 0, sigma_med)
  preco_pao <- exp(y)
  
  dados_brutos <- tibble(
    data = datas,
    preco_pao = as.numeric(preco_pao),
    farinha = as.numeric(farinha),
    cambio  = as.numeric(cambio),
    diesel  = as.numeric(diesel),
    energia = as.numeric(energia),
    ipca    = as.numeric(ipca),
    salario_minimo = as.numeric(sal_min)
  )
  mensagem_dados <- "Dados simulados (demonstrativo)."
}

# ---------------------------
# 2) Preparar features (z_t) — VERSÃO ROBUSTA
# ---------------------------

# Escala segura: zera colunas com sd=0 após centrar
escala_segura <- function(M){
  if(is.null(dim(M))) { M <- matrix(M, ncol = 1) }
  S <- scale(M)
  sds <- apply(M, 2, sd, na.rm = TRUE)
  if(any(!is.finite(sds) | sds == 0)) {
    zcols <- which(!is.finite(sds) | sds == 0)
    for(j in zcols) S[,j] <- 0
  }
  colnames(S) <- colnames(M)
  S
}

prep_dados <- function(df){
  # Normaliza nome da coluna data (tenta acertar variantes comuns)
  if(!"data" %in% names(df)){
    candidatos_data <- c("data","date","dt")
    hitd <- candidatos_data[candidatos_data %in% names(df)]
    if(length(hitd)>=1) df <- dplyr::rename(df, data = !!sym(hitd[1]))
    else stop("Coluna de data ausente (esperado: data/date/dt).")
  }
  if(!"preco_pao" %in% names(df)){
    candidatos_preco <- c("preco_pao","preco","preco_frances","preco_pao_frances","preco_kg")
    hitp <- candidatos_preco[candidatos_preco %in% names(df)]
    if(length(hitp)>=1) df <- dplyr::rename(df, preco_pao = !!sym(hitp[1]))
    else stop("Coluna do preço ausente (ex.: preco_pao).")
  }
  
  df <- df |>
    mutate(data = as.Date(data)) |>
    arrange(data) |>
    mutate(y_log = log(preco_pao)) |>
    tidyr::drop_na(preco_pao, data)
  
  # Sinônimos -> canônicos
  sinonimos <- list(
    farinha = c("farinha","preco_farinha","indice_farinha","farinha_trigo"),
    cambio  = c("cambio","usdbrl","dolar","taxa_cambio","brlusd"),
    diesel  = c("diesel","preco_diesel","combustivel_diesel"),
    energia = c("energia","tarifa_energia","eletricidade","energia_eletrica","kwh"),
    ipca    = c("ipca","inflacao","indice_ipca"),
    salario_minimo = c("salario_minimo","salario","sal_min","salarioMinimo","salmin")
  )
  
  pares <- c() # new_name = old_name (formato rename)
  for (nome_can in names(sinonimos)){
    candidatos <- sinonimos[[nome_can]]
    hit <- candidatos[candidatos %in% names(df)]
    if(length(hit) >= 1){
      pares[nome_can] <- hit[1]
    }
  }
  if(length(pares)){
    df <- dplyr::rename(df, !!!pares)
  }
  
  alvo_cov <- c("farinha","cambio","diesel","energia","ipca","salario_minimo")
  presentes <- intersect(alvo_cov, names(df))
  
  if(length(presentes) == 0){
    df$constante0 <- 0
    nomes_cov <- "constante0"
  } else {
    nomes_cov <- presentes
  }
  
  covs <- df |> dplyr::select(all_of(nomes_cov)) |> as.matrix()
  covs_esc <- escala_segura(covs)
  
  lista <- list(
    df = df,
    z = covs_esc,
    nomes_cov = colnames(covs_esc),
    covs_presentes = presentes
  )
  return(lista)
}

lista <- prep_dados(dados_brutos)
df <- lista$df
z  <- lista$z
nomes_cov <- lista$nomes_cov
n <- nrow(df)

message(mensagem_dados, " Observações: ", n)
message("Covariáveis usadas: ", paste(nomes_cov, collapse=", "))

# Detecta passo temporal para datas futuras (dias)
passo_dias <- {
  if(n >= 2){
    diffs <- as.integer(diff(df$data))
    as.integer(stats::median(diffs, na.rm = TRUE))
  } else 7L
}
if(!is.finite(passo_dias) || passo_dias <= 0) passo_dias <- 7L

# ===========================================
# 3) Filtro de Partículas (Bootstrap Filter)
# ===========================================

sistematico_resample <- function(weights){
  N <- length(weights)
  positions <- (runif(1) + (0:(N-1)))/N
  cumsumw <- cumsum(weights/sum(weights))
  indexes <- integer(N); i <- 1; j <- 1
  while(i <= N){
    while(positions[i] > cumsumw[j]) j <- j + 1
    indexes[i] <- j
    i <- i + 1
  }
  indexes
}

# Log-verossimilhança aproximada via SMC para dado vetor de parâmetros
loglik_smc <- function(par, y, z, Npart = 300, ess_frac = 0.5){
  n_betas <- ncol(z)
  alpha <- par[1]
  betas <- par[2:(1+n_betas)]
  mu_v  <- par[2+n_betas]
  phi_v <- tanh(par[3+n_betas])
  sigma_v <- exp(par[4+n_betas])
  sigma_med <- exp(par[5+n_betas])
  v0 <- par[6+n_betas]
  
  n <- length(y)
  x_part <- rep(y[1], Npart)
  v_part <- rep(v0,   Npart)
  
  loglik <- 0
  w <- rep(1/Npart, Npart)
  
  for(t in 2:n){
    drift_t <- alpha + as.numeric(z[t,] %*% betas)
    v_part <- mu_v + phi_v*(v_part - mu_v) + sigma_v*rnorm(Npart)
    x_part <- x_part + drift_t + exp(0.5*v_part)*rnorm(Npart)
    
    dens <- dnorm(y[t], mean = x_part, sd = sigma_med, log = FALSE)
    w <- w * pmax(dens, .Machine$double.xmin)
    sw <- sum(w)
    if(!is.finite(sw) || sw<=0) return(-1e12)
    w <- w / sw
    loglik <- loglik + log(sw/Npart)
    
    ess <- 1/sum(w^2)
    if(ess < ess_frac * Npart){
      idx <- sistematico_resample(w)
      x_part <- x_part[idx]
      v_part <- v_part[idx]
      w <- rep(1/Npart, Npart)
    }
  }
  return(loglik)
}

# ==================================
# 4) Ajuste de parâmetros por 'optim'
# ==================================

y <- df$y_log
n_betas <- ncol(z)

par_ini <- c(
  alpha = 0.0,
  betas = rep(0.0, n_betas),
  mu_v  = -5.0,
  atanh_phi = atanh(0.9),
  log_sigma_v = log(0.2),
  log_sigma_med = log(0.05),
  v0 = -5.0
)

obj_fn <- function(par){
  -loglik_smc(par, y=y, z=z, Npart = 500, ess_frac = 0.5)
}

ajuste <- optim(
  par = par_ini,
  fn  = obj_fn,
  method = "L-BFGS-B",
  control = list(maxit = 80)
)

cat("\nStatus otimização:", ajuste$convergence, "-", ajuste$message, "\n")

# Extrair parâmetros ajustados
par_hat <- ajuste$par
alpha_hat     <- par_hat[1]
betas_hat     <- par_hat[2:(1+n_betas)]
mu_v_hat      <- par_hat[2+n_betas]
phi_v_hat     <- tanh(par_hat[3+n_betas])
sigma_v_hat   <- exp(par_hat[4+n_betas])
sigma_med_hat <- exp(par_hat[5+n_betas])
v0_hat        <- par_hat[6+n_betas]

cat("\n================ PARÂMETROS AJUSTADOS ================\n")
cat("alpha         :", round(alpha_hat,6), "\n")
for(j in seq_len(n_betas)){
  cat(sprintf("beta[%s]     : % .6f\n", nomes_cov[j], betas_hat[j]))
}
cat("mu_v          :", round(mu_v_hat,4), "\n")
cat("phi_v         :", round(phi_v_hat,4), "\n")
cat("sigma_v       :", round(sigma_v_hat,4), "\n")
cat("sigma_med     :", round(sigma_med_hat,4), "\n")
cat("v0            :", round(v0_hat,4), "\n")
cat("LogLik (aprox):", round(-ajuste$value,2), "\n")
cat("======================================================\n\n")

# =========================================
# 5) Filtragem dos estados e previsão (H)
# =========================================

filtrar_estados <- function(par, y, z, Npart = 1500){
  n_betas <- ncol(z); n <- length(y)
  alpha <- par[1]
  betas <- par[2:(1+n_betas)]
  mu_v  <- par[2+n_betas]
  phi_v <- tanh(par[3+n_betas])
  sigma_v <- exp(par[4+n_betas])
  sigma_med <- exp(par[5+n_betas])
  v0 <- par[6+n_betas]
  
  x_part <- rep(y[1], Npart)
  v_part <- rep(v0,   Npart)
  w <- rep(1/Npart, Npart)
  
  est <- tibble(
    t = 1:n,
    x_filtrado = NA_real_,
    vol_log_filtrada = NA_real_
  )
  est$x_filtrado[1] <- mean(x_part)
  est$vol_log_filtrada[1] <- mean(v_part)
  
  for(t in 2:n){
    drift_t <- alpha + as.numeric(z[t,] %*% betas)
    v_part <- mu_v + phi_v*(v_part - mu_v) + sigma_v*rnorm(Npart)
    x_part <- x_part + drift_t + exp(0.5*v_part)*rnorm(Npart)
    
    dens <- dnorm(y[t], mean = x_part, sd = sigma_med, log = FALSE)
    w <- w * pmax(dens, .Machine$double.xmin)
    w <- w/sum(w)
    
    est$x_filtrado[t] <- sum(w * x_part)
    est$vol_log_filtrada[t] <- sum(w * v_part)
    
    idx <- sistematico_resample(w)
    x_part <- x_part[idx]
    v_part <- v_part[idx]
    w <- rep(1/Npart, Npart)
  }
  est
}

est <- filtrar_estados(par_hat, y, z, Npart = 2000)

prever <- function(par, y_ultimo, v_ultimo, z_futuro, H = 12, Npart = 5000){
  n_betas <- ncol(z_futuro)
  alpha <- par[1]
  betas <- par[2:(1+n_betas)]
  mu_v  <- par[2+n_betas]
  phi_v <- tanh(par[3+n_betas])
  sigma_v <- exp(par[4+n_betas])
  sigma_med <- exp(par[5+n_betas])
  
  x_part <- rep(y_ultimo, Npart)
  v_part <- rep(v_ultimo, Npart)
  
  prev <- tibble(h=1:H, preco_med=NA_real_, li=NA_real_, ls=NA_real_)
  for(h in 1:H){
    drift_h <- alpha + as.numeric(z_futuro[h,] %*% betas)
    v_part <- mu_v + phi_v*(v_part - mu_v) + sigma_v*rnorm(Npart)
    x_part <- x_part + drift_h + exp(0.5*v_part)*rnorm(Npart)
    
    y_pred_part <- x_part + rnorm(Npart, 0, sigma_med)
    p_pred_part <- exp(y_pred_part)
    
    prev$preco_med[h] <- mean(p_pred_part)
    prev$li[h] <- quantile(p_pred_part, 0.10)
    prev$ls[h] <- quantile(p_pred_part, 0.90)
  }
  prev
}

# H passos à frente com covariáveis "congeladas" no último valor
H <- 12
z_futuro <- matrix(rep(z[n,], each=H), nrow=H)
prev <- prever(par_hat, tail(est$x_filtrado,1), tail(est$vol_log_filtrada,1), z_futuro, H=H)

# ==========================
# 6) Visualizações e saída
# ==========================

# Série observada vs. estado filtrado (nível de preço)
graf1 <- df %>%
  mutate(preco_filtrado = exp(est$x_filtrado)) %>%
  select(data, preco_pao, preco_filtrado) %>%
  pivot_longer(-data, names_to="serie", values_to="valor") %>%
  ggplot(aes(data, valor, linetype=serie)) +
  geom_line() +
  scale_y_continuous("Preço do pão (R$)", labels = label_number(big.mark=".", decimal.mark=",")) +
  scale_x_date(NULL) +
  labs(title="Preço observado vs. nível filtrado (estado latente)") +
  theme_minimal(base_size = 12)

# Volatilidade (desvio-padrão) filtrada
graf2 <- tibble(
  data = df$data,
  desvpad_ret = exp(0.5*est$vol_log_filtrada)
) %>%
  ggplot(aes(data, desvpad_ret)) +
  geom_line() +
  labs(title="Volatilidade estocástica filtrada (desv. padrão do retorno)",
       y="σ_t", x=NULL) +
  theme_minimal(base_size = 12)

# Previsões
datas_fut <- seq(from = max(df$data) + passo_dias, by = passo_dias, length.out = H)
graf3 <- prev %>%
  mutate(data = datas_fut) %>%
  ggplot(aes(data, preco_med)) +
  geom_ribbon(aes(ymin=li, ymax=ls), alpha=0.2) +
  geom_line() +
  labs(title=sprintf("Previsão do preço do pão — %d passos à frente", H),
       y="Preço previsto (R$)", x=NULL) +
  theme_minimal(base_size = 12)

print(graf1)
print(graf2)
print(graf3)

# Contribuições de covariáveis (efeito por 1 desvio-padrão padronizado)
efeitos <- tibble(
  covariavel = nomes_cov,
  beta = as.numeric(betas_hat)
) %>% arrange(desc(abs(beta)))
print(efeitos)
