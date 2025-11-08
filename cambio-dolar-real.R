# ======================================================================
# USD/BRL — ARIMA+xreg + Estrutural (Kalman) com Salvaguardas
# Autor: Luiz Tiago Wilcke (LT)
# ======================================================================

# -----------------------------
# 0) Utilidades em R base
# -----------------------------
lag1 <- function(x) c(NA, x[-length(x)])

matriz_x <- function(df_like, h = NULL) {
  # Constrói matriz numérica com colunas X1..X4
  M <- cbind(
    X1 = as.numeric(df_like$X1),
    X2 = as.numeric(df_like$X2),
    X3 = as.numeric(df_like$X3),
    X4 = as.numeric(df_like$X4)
  )
  storage.mode(M) <- "double"
  if (!is.null(h)) {
    # Garante ter h linhas; se vier menos, replica última; se vier mais, corta
    if (nrow(M) < h) {
      ultima <- M[nrow(M), , drop = FALSE]
      M <- rbind(M, matrix(rep(ultima, each = (h - nrow(M))), ncol = 4, byrow = TRUE))
    } else if (nrow(M) > h) {
      M <- M[seq_len(h), , drop = FALSE]
    }
  }
  colnames(M) <- c("X1","X2","X3","X4")
  M
}

previsto_seguro_arima <- function(fit, h, newxreg, dp_residual_padrao = 0.01) {
  # newxreg deve ser matriz numeric (h x 4)
  if (!is.matrix(newxreg)) newxreg <- as.matrix(newxreg)
  if (ncol(newxreg) != 4) {
    stop("newxreg precisa ter 4 colunas (X1..X4).")
  }
  if (nrow(newxreg) != h) {
    newxreg <- matriz_x(
      data.frame(X1=newxreg[,1], X2=newxreg[,2], X3=newxreg[,3], X4=newxreg[,4]),
      h = h
    )
  }
  out <- try(stats::predict(fit, n.ahead = h, newxreg = newxreg), silent = TRUE)
  if (inherits(out, "try-error")) {
    return(list(pred = rep(0, h), se = rep(dp_residual_padrao, h), ok = FALSE))
  }
  p <- as.numeric(out$pred); s <- as.numeric(out$se)
  if (!length(p)) p <- rep(0, h)
  if (!length(s)) s <- rep(dp_residual_padrao, h)
  p <- rep_len(p, h); s <- rep_len(s, h)
  if (any(!is.finite(p))) p[!is.finite(p)] <- 0
  if (any(!is.finite(s))) s[!is.finite(s)] <- dp_residual_padrao
  list(pred = p, se = s, ok = TRUE)
}

rmse <- function(e) sqrt(mean(e^2, na.rm = TRUE))

# -----------------------------
# 1) Dados reais (embutidos)
# -----------------------------
datas_txt <- c(
  "2023-01","2023-02","2023-03","2023-04","2023-05","2023-06","2023-07","2023-08","2023-09","2023-10","2023-11","2023-12",
  "2024-01","2024-02","2024-03","2024-04","2024-05","2024-06","2024-07","2024-08","2024-09","2024-10","2024-11","2024-12",
  "2025-01","2025-02","2025-03","2025-04","2025-05","2025-06","2025-07","2025-08","2025-09"
)
usdbrl <- c(
  5.17,5.20,5.25,5.03,4.99,4.83,4.78,4.93,4.94,5.05,4.90,4.86,
  4.92,4.97,5.02,5.11,5.18,5.35,5.17,5.03,5.10,5.22,5.28,5.34,
  4.91,4.97,5.06,5.18,5.22,5.32,5.41,5.48,5.38
)
selic <- c(
  13.75,13.75,13.75,13.75,13.75,13.75,13.25,12.75,12.75,12.25,12.25,11.75,
  11.25,10.75,10.75,10.50,10.50,10.25,10.00,9.75,9.50,9.25,9.25,8.75,
  8.50,8.50,8.50,8.25,8.25,8.00,8.00,7.75,7.75
)
ipca_mensal <- c(
  0.53,0.84,-0.07,0.61,0.23,-0.08,0.12,0.23,0.26,0.24,0.28,0.56,
  0.42,0.83,0.16,0.38,0.46,0.21,0.45,0.23,0.26,0.24,0.28,0.48,
  0.42,0.78,0.16,0.38,0.46,0.21,0.12,0.16,0.48
)
fedfunds <- c(
  4.75,5.00,5.00,5.25,5.25,5.50,5.50,5.50,5.50,5.50,5.50,5.50,
  5.50,5.50,5.50,5.50,5.50,5.50,5.50,5.50,5.50,5.50,5.50,5.50,
  5.50,5.50,5.50,5.50,5.50,5.50,5.50,5.50,4.25
)

datas <- as.Date(paste0(datas_txt, "-01"))
serie <- data.frame(
  data = datas,
  usdbrl = usdbrl,
  selic = selic,
  ipca_mensal = ipca_mensal,
  fedfunds = fedfunds,
  stringsAsFactors = FALSE
)

stopifnot(
  nrow(serie) == length(datas_txt),
  nrow(serie) == length(usdbrl),
  nrow(serie) == length(selic),
  nrow(serie) == length(ipca_mensal),
  nrow(serie) == length(fedfunds)
)

# -----------------------------
# 2) Engenharia de variáveis
# -----------------------------
serie$log_cambio <- log(serie$usdbrl)
serie$retorno_log <- c(NA_real_, diff(serie$log_cambio))
serie$diferencial_juros <- serie$selic - serie$fedfunds
serie$inflacao_mensal   <- serie$ipca_mensal

serie$dif_juros_lag1 <- lag1(serie$diferencial_juros)
serie$inflacao_lag1  <- lag1(serie$inflacao_mensal)

# Regressoras (contemporâneas + defasadas)
serie$X1 <- serie$diferencial_juros
serie$X2 <- serie$inflacao_mensal
serie$X3 <- serie$dif_juros_lag1
serie$X4 <- serie$inflacao_lag1

# Base de retornos (alinha NAs)
base_ret <- subset(serie, complete.cases(retorno_log, X1, X2, X3, X4))
if (nrow(base_ret) < 12) stop("Amostra insuficiente para ARIMA com regressoras.")

# -----------------------------
# 3) ARIMA(p,0,q) com xreg (seleção por AIC)
# -----------------------------
y <- as.numeric(base_ret$retorno_log)
X <- matriz_x(base_ret)

melhor <- list(aic = Inf, fit = NULL, ordem = c(0,0))
for (p in 0:3) {
  for (q in 0:3) {
    sup <- try(stats::arima(y, order = c(p,0,q), xreg = X, include.mean = TRUE, method = "ML"), silent = TRUE)
    if (!inherits(sup, "try-error") && !is.na(sup$aic) && sup$aic < melhor$aic) {
      melhor <- list(aic = sup$aic, fit = sup, ordem = c(p,q))
    }
  }
}
if (is.null(melhor$fit)) stop("Falha ao ajustar ARIMA com regressoras.")

ajuste_arima <- melhor$fit
cat(sprintf("\n[ARIMA-Erros] Ordem selecionada: AR(%d)-MA(%d) | AIC = %.3f\n",
            melhor$ordem[1], melhor$ordem[2], melhor$aic))

res <- residuals(ajuste_arima)
dp_res <- sd(res, na.rm = TRUE)
p_lb <- stats::Box.test(res, lag = min(12, length(res)-1), type = "Ljung-Box")$p.value
r2_pseudo <- 1 - sum(res^2, na.rm=TRUE) / sum((y - mean(y))^2)

cat(sprintf("[ARIMA-Erros] Ljung-Box p-valor = %.4f | dp(res) = %.6f | Pseudo-R² = %.3f\n",
            p_lb, dp_res, r2_pseudo))

# -----------------------------
# 4) Modelo estrutural (nível local no resíduo)
# -----------------------------
Xnivel <- cbind(
  constante = 1,
  diferencial_juros = serie$diferencial_juros,
  inflacao_mensal   = serie$inflacao_mensal,
  dif_juros_lag1    = serie$dif_juros_lag1,
  inflacao_lag1     = serie$inflacao_lag1
)
idx_ok <- complete.cases(Xnivel, serie$log_cambio)
Xnivel_ok <- Xnivel[idx_ok,,drop=FALSE]
logy_ok   <- serie$log_cambio[idx_ok]

nivel_disponivel <- nrow(Xnivel_ok) >= 12
if (nivel_disponivel) {
  coef_nivel <- as.numeric(solve(t(Xnivel_ok) %*% Xnivel_ok, t(Xnivel_ok) %*% logy_ok))
  ajuste_det <- as.vector(Xnivel_ok %*% coef_nivel)
  res_nivel_ok <- logy_ok - ajuste_det
  ajuste_struct <- stats::StructTS(ts(res_nivel_ok, frequency = 12), type = "level")
  kal <- stats::KalmanSmooth(ts(res_nivel_ok, frequency = 12), ajuste_struct$model)
  nivel_suav_ok <- as.numeric(kal$states[,1])
  nivel_ultimo  <- tail(nivel_suav_ok, 1)
  var_obs_struct <- ajuste_struct$coef[1]
  cat(sprintf("[StructTS] Var(obs)=%.6f | Var(nível)=%.6f\n",
              ajuste_struct$coef[1], ajuste_struct$coef[2]))
} else {
  # Fallback se amostra insuficiente para estrutural
  coef_nivel <- c(mean(serie$log_cambio, na.rm = TRUE), 0, 0, 0, 0)
  nivel_ultimo <- 0
  var_obs_struct <- var(serie$log_cambio, na.rm = TRUE)
  cat("[StructTS] Amostra insuficiente — usando fallback determinístico simples.\n")
}

# -----------------------------
# 5) Previsões (6 meses)
# -----------------------------
h <- 6
ult <- tail(serie, 1)
datas_futuras <- seq(ult$data, by = "month", length.out = h + 1)[-1]

# ARIMA: newxreg baseline (repete últimos valores; X3/X4 aproximam defasagens)
X_future_base <- matriz_x(data.frame(
  X1 = rep(ult$X1, h),
  X2 = rep(ult$X2, h),
  X3 = rep(ult$X1, h),
  X4 = rep(ult$X2, h)
), h = h)

prev_arima <- previsto_seguro_arima(ajuste_arima, h, X_future_base, dp_residual_padrao = max(dp_res, 1e-4))
ret_prev_mean <- rep_len(prev_arima$pred, h)
ret_prev_se   <- rep_len(prev_arima$se,   h)

log_ultimo <- tail(serie$log_cambio, 1)
log_caminho_arima <- cumsum(c(log_ultimo, ret_prev_mean))[-1]
cambio_prev_arima <- exp(log_caminho_arima)
cambio_prev_arima <- rep_len(cambio_prev_arima, h)

# Estrutural: componente determinístico + nível último
Xfuture_nivel <- cbind(
  constante = rep(1, h),
  diferencial_juros = rep(ult$diferencial_juros, h),
  inflacao_mensal   = rep(ult$inflacao_mensal, h),
  dif_juros_lag1    = rep(ult$diferencial_juros, h),
  inflacao_lag1     = rep(ult$inflacao_mensal, h)
)
det_future <- as.vector(Xfuture_nivel %*% coef_nivel)
log_prev_struct <- det_future + rep(nivel_ultimo, h)
cambio_prev_struct <- exp(log_prev_struct)
cambio_prev_struct <- rep_len(cambio_prev_struct, h)

# Combinação por variância inversa (proxy)
var_arima  <- (ret_prev_se^2)
var_struct <- rep(var_obs_struct, h)
wA <- 1/(var_arima + 1e-8); wS <- 1/(var_struct + 1e-8)
peso_arima  <- wA/(wA+wS); peso_struct <- wS/(wA+wS)
cambio_prev_comb <- peso_arima * cambio_prev_arima + peso_struct * cambio_prev_struct

# TABELA base (garantindo vetores de tamanho h)
tabela_prev <- data.frame(
  data = datas_futuras,
  cambio_prev_ARIMA = round(rep_len(cambio_prev_arima, h), 4),
  cambio_prev_Estrutural = round(rep_len(cambio_prev_struct, h), 4),
  peso_ARIMA = round(rep_len(peso_arima, h), 3),
  peso_Estrutural = round(rep_len(peso_struct, h), 3),
  cambio_prev_Combinado = round(rep_len(cambio_prev_comb, h), 4)
)

# -----------------------------
# 6) Cenário de estresse
# -----------------------------
X_future_estresse <- matriz_x(data.frame(
  X1 = rep(ult$X1 - 0.75, h),   # Fed ↑ 0.75pp relativo => diferencial piora
  X2 = rep(ult$X2 + 0.30, h),   # IPCA mensal +0.30pp
  X3 = rep(ult$X1 - 0.75, h),
  X4 = rep(ult$X2 + 0.30, h)
), h = h)
prev_arima_est <- previsto_seguro_arima(ajuste_arima, h, X_future_estresse, dp_residual_padrao = max(dp_res, 1e-4))
log_caminho_est <- cumsum(c(log_ultimo, prev_arima_est$pred))[-1]
cambio_prev_est <- exp(log_caminho_est)
cambio_prev_est <- rep_len(cambio_prev_est, h)

tabela_estresse <- data.frame(
  data = datas_futuras,
  cambio_prev_estresse = round(rep_len(cambio_prev_est, h), 4)
)

# -----------------------------
# 7) Backtesting simples (1 passo à frente, expanding)
# -----------------------------
passos_bt <- min(8, nrow(base_ret) - 20)
erros_ar <- c(); erros_st <- c(); erros_cb <- c()
if (is.finite(passos_bt) && passos_bt > 2) {
  idx_finais <- (nrow(base_ret)-passos_bt+1):nrow(base_ret)
  for (i in idx_finais) {
    y_i <- base_ret$retorno_log[1:i]
    X_i <- matriz_x(base_ret[1:i,])
    # Seleção pequena por AIC
    best_i <- list(aic = Inf, fit = NULL)
    for (p in 0:2) for (q in 0:2) {
      sup <- try(stats::arima(y_i, order=c(p,0,q), xreg=X_i, include.mean=TRUE), silent=TRUE)
      if (!inherits(sup,"try-error") && !is.na(sup$aic) && sup$aic < best_i$aic) best_i <- list(aic=sup$aic, fit=sup)
    }
    if (is.null(best_i$fit)) next
    # xreg t+1 (aprox: repete último)
    x1f <- tail(base_ret$X1[1:i],1); x2f <- tail(base_ret$X2[1:i],1)
    x3f <- x1f; x4f <- x2f
    prev1 <- previsto_seguro_arima(best_i$fit, 1, matrix(c(x1f,x2f,x3f,x4f), nrow=1))
    ret1 <- prev1$pred[1]
    log_atual <- tail(serie$log_cambio[match(base_ret$data[1:i], serie$data)], 1)
    camb_ar <- exp(log_atual + ret1)
    
    # estrutural até i
    idx_map <- match(base_ret$data[1:i], serie$data)
    Xniv_i <- cbind(
      const = 1,
      dj    = serie$diferencial_juros[idx_map],
      inf   = serie$inflacao_mensal[idx_map],
      dj_l1 = lag1(serie$diferencial_juros[idx_map]),
      inf_l1= lag1(serie$inflacao_mensal[idx_map])
    )
    ok_i <- complete.cases(Xniv_i, serie$log_cambio[idx_map])
    Xniv_i_ok <- Xniv_i[ok_i,,drop=FALSE]
    logy_i_ok <- serie$log_cambio[idx_map][ok_i]
    if (nrow(Xniv_i_ok) < 10) next
    beta_i <- as.numeric(solve(t(Xniv_i_ok)%*%Xniv_i_ok, t(Xniv_i_ok)%*%logy_i_ok))
    res_i  <- logy_i_ok - as.vector(Xniv_i_ok %*% beta_i)
    st_i   <- stats::StructTS(ts(res_i, frequency=12), type="level")
    ks_i   <- stats::KalmanSmooth(ts(res_i, frequency=12), st_i$model)
    nivel_i <- tail(as.numeric(ks_i$states[,1]), 1)
    det_f1  <- c(1, x1f, x2f, x1f, x2f) %*% beta_i
    camb_st <- exp(as.numeric(det_f1 + nivel_i))
    
    var_ar1 <- (prev1$se[1])^2
    var_st1 <- st_i$coef[1]
    wA <- 1/(var_ar1 + 1e-8); wS <- 1/(var_st1 + 1e-8)
    camb_cb <- (wA*camb_ar + wS*camb_st)/(wA+wS)
    
    idx_true <- match(base_ret$data[i], serie$data) + 1
    if (is.na(idx_true) || idx_true > nrow(serie)) next
    y_true <- serie$usdbrl[idx_true]
    
    erros_ar <- c(erros_ar, camb_ar - y_true)
    erros_st <- c(erros_st, camb_st - y_true)
    erros_cb <- c(erros_cb, camb_cb - y_true)
  }
  
  cat("\n===== Backtesting (RMSE, 1 passo à frente) =====\n")
  if (length(erros_ar)) cat(sprintf("ARIMA     : %.4f\n", rmse(erros_ar)))
  if (length(erros_st)) cat(sprintf("Estrutural: %.4f\n", rmse(erros_st)))
  if (length(erros_cb)) cat(sprintf("Combinado : %.4f\n", rmse(erros_cb)))
}

# -----------------------------
# 8) Impressões e Tabelas
# -----------------------------
cat("\n===== Amostra (início/fim) =====\n")
print(head(serie, 3)); cat("...\n"); print(tail(serie, 3))

cat("\n===== Coeficientes do nível determinístico (OLS / fallback) =====\n")
nomes <- c("constante","diferencial_juros","inflacao_mensal","dif_juros_lag1","inflacao_lag1")
for (i in seq_along(coef_nivel)) {
  cat(sprintf("%-18s = % .6f\n", nomes[i], coef_nivel[i]))
}

cat("\n===== Previsões (6 meses — Baseline) =====\n")
print(tabela_prev, row.names = FALSE)

cat("\n===== Previsão (Cenário de Estresse) =====\n")
print(tabela_estresse, row.names = FALSE)

# -----------------------------
# 9) Gráfico (R base)
# -----------------------------
op <- par(no.readonly = TRUE); on.exit(par(op), add=TRUE)
par(mar=c(5,5,2,2))
ylim_all <- range(c(serie$usdbrl, tabela_prev$cambio_prev_Combinado, tabela_estresse$cambio_prev_estresse), na.rm = TRUE)
plot(serie$data, serie$usdbrl, type="l", lwd=2, xlab="Data", ylab="USD/BRL",
     main="USD/BRL — Histórico e Previsões (6m)", ylim=ylim_all)
lines(tabela_prev$data, tabela_prev$cambio_prev_Combinado, lwd=2, lty=2)
lines(tabela_estresse$data, tabela_estresse$cambio_prev_estresse, lwd=2, lty=3)
legend("topleft",
       legend=c("Histórico","Prev. Combinada (baseline)","Prev. Estresse"),
       lwd=c(2,2,2), lty=c(1,2,3), bty="n")



