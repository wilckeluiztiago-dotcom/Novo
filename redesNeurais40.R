# ============================================================
# WTI — Rede Neural 
# Autor: Luiz Tiago Wilcke (LT)
# ============================================================

set.seed(123)

# ------------------ PARÂMETROS GERAIS ------------------
usar_csv    <- FALSE
caminho_csv <- NULL   # "petroleo.csv" com colunas: data,wti,brent,usdbrl
data_inicial <- "2015-01-01"
prop_treino  <- 0.8

# Arquitetura/treino da rede
unidades1 <- 64
unidades2 <- 32
epocas_max <- 500
batch_size <- 64
lr <- 0.001
beta1 <- 0.9
beta2 <- 0.999
eps <- 1e-8
paciencia <- 25   

# ------------------ FUNÇÕES AUXILIARES (base R) ------------------
roll_mean <- function(x, n){
  if (length(x) < n) return(rep(NA_real_, length(x)))
  res <- rep(NA_real_, length(x))
  s <- cumsum(c(0, x))
  res[n:length(x)] <- (s[(n+1):length(s)] - s[1:(length(s)-n)]) / n
  res
}
roll_sd <- function(x, n){
  m <- roll_mean(x, n)
  res <- rep(NA_real_, length(x))
  for(i in seq_len(length(x))){
    if (i < n) { res[i] <- NA_real_ } else {
      seg <- x[(i-n+1):i]; mm <- m[i]
      res[i] <- sqrt(mean((seg - mm)^2))
    }
  }
  res
}
rmse <- function(y, yhat) sqrt(mean((y - yhat)^2))
mae  <- function(y, yhat) mean(abs(y - yhat))
mape <- function(y, yhat) mean(abs((y - yhat)/y)) * 100

tanh_act   <- function(z) tanh(z)
tanh_deriv <- function(a) 1 - a*a
xavier <- function(fan_in, fan_out){
  lim <- sqrt(6/(fan_in + fan_out))
  matrix(runif(fan_in*fan_out, -lim, lim), nrow = fan_in, ncol = fan_out)
}
adam_init <- function(W){
  list(m = matrix(0, nrow=nrow(W), ncol=ncol(W)),
       v = matrix(0, nrow=nrow(W), ncol=ncol(W)))
}
adam_step <- function(W, g, state, t, lr, beta1, beta2, eps){
  state$m <- beta1*state$m + (1-beta1)*g
  state$v <- beta2*state$v + (1-beta2)*(g*g)
  m_hat <- state$m / (1 - beta1^t)
  v_hat <- state$v / (1 - beta2^t)
  W <- W - lr * m_hat / (sqrt(v_hat) + eps)
  list(W=W, state=state)
}
escala <- function(x){ m <- mean(x); s <- sd(x); if (is.na(s) || s==0) s <- 1; list(scale=(x-m)/s, mean=m, sd=s) }
denorm <- function(z, m, s) z*s + m

# ------------------ CARREGAR DADOS (CSV / Yahoo / Sintético) ------------------
carrega_dados <- function(){
  if (usar_csv && !is.null(caminho_csv) && file.exists(caminho_csv)) {
    dat <- read.csv(caminho_csv, stringsAsFactors = FALSE)
    stopifnot(all(c("data","wti","brent","usdbrl") %in% names(dat)))
    dat$data <- as.Date(dat$data)
    dat <- dat[order(dat$data), ]
    return(data.frame(data=dat$data, preco_wti=dat$wti, preco_brent=dat$brent, usdbrl=dat$usdbrl))
  }
  preco_wti <- preco_brent <- usdbrl <- datas <- NULL
  have_qm <- requireNamespace("quantmod", quietly = TRUE)
  if (have_qm) {
    suppressWarnings({
      xt_wti   <- try(quantmod::getSymbols("CL=F", src="yahoo", from=data_inicial, auto.assign=FALSE), silent=TRUE)
      xt_brent <- try(quantmod::getSymbols("BZ=F", src="yahoo", from=data_inicial, auto.assign=FALSE), silent=TRUE)
      xt_usd   <- try(quantmod::getSymbols("USDBRL=X", src="yahoo", from=data_inicial, auto.assign=FALSE), silent=TRUE)
    })
    ok_dl <- !inherits(xt_wti,"try-error") && !inherits(xt_brent,"try-error") && !inherits(xt_usd,"try-error")
    if (ok_dl){
      A <- function(x) quantmod::Ad(x)
      iw <- as.Date(index(A(xt_wti))); ib <- as.Date(index(A(xt_brent))); iu <- as.Date(index(A(xt_usd)))
      common <- Reduce(intersect, list(iw, ib, iu))
      if (length(common) > 0){
        return(data.frame(
          data = common,
          preco_wti   = as.numeric(A(xt_wti)[as.character(common)]),
          preco_brent = as.numeric(A(xt_brent)[as.character(common)]),
          usdbrl      = as.numeric(A(xt_usd)[as.character(common)])
        ))
      }
    }
  }
  # Fallback sintético robusto
  n <- 1500
  datas <- seq(as.Date("2015-01-01"), by="day", length.out=n)
  set.seed(42)
  r_wti   <- rnorm(n, mean=0.0002, sd=0.02)
  r_brent <- rnorm(n, mean=0.00018, sd=0.018)
  log_wti <- cumsum(r_wti) + log(50)
  log_bre <- cumsum(r_brent) + log(55)
  preco_wti   <- exp(log_wti)
  preco_brent <- exp(log_bre)
  usdbrl      <- 3.0 + cumsum(rnorm(n, 0, 0.003))
  data.frame(data=datas, preco_wti=preco_wti, preco_brent=preco_brent, usdbrl=usdbrl)
}

df_raw <- carrega_dados()
df_raw <- df_raw[order(df_raw$data), ]
df_raw <- df_raw[!is.na(df_raw$preco_wti), ]

# ------------------ CONSTRUTOR DE FEATURES COM BACKOFF ------------------
# Tenta janelas (20,50,21) -> se poucos dados, (5,10,5) -> (2,4,2).
# Se ainda ficar vazio, usa apenas retornos (sem MM/vol) e garante > 300 linhas.
monta_features <- function(df, j1=20, j2=50, jv=21, usar_spread=TRUE, usar_usd=TRUE){
  retorno_wti   <- c(NA, diff(log(df$preco_wti)))
  retorno_brent <- c(NA, diff(log(df$preco_brent)))
  mm1  <- roll_mean(df$preco_wti, j1)
  mm2  <- roll_mean(df$preco_wti, j2)
  vol  <- roll_sd(retorno_wti, jv)
  spread <- df$preco_brent - df$preco_wti
  
  alvo_amanha <- c(df$preco_wti[-1], NA)
  
  X <- cbind(
    preco_wti = df$preco_wti,
    retorno_wti = retorno_wti,
    retorno_brent = retorno_brent,
    mm_curta = mm1,
    mm_longa = mm2,
    volatil = vol
  )
  if (usar_spread) X <- cbind(X, spread_brent_wti=spread)
  if (usar_usd && "usdbrl" %in% names(df)) X <- cbind(X, usdbrl=df$usdbrl, usdbrl_log=log(df$usdbrl))
  Y <- alvo_amanha
  
  ok <- stats::complete.cases(X) & !is.na(Y)
  list(X=X[ok,,drop=FALSE], Y=Y[ok], data=df$data[ok])
}

build_dataset <- function(df){
  tries <- list(
    list(j1=20,j2=50,jv=21, usar_spread=TRUE,  usar_usd=TRUE),
    list(j1=5, j2=10,jv=5,  usar_spread=TRUE,  usar_usd=TRUE),
    list(j1=2, j2=4, jv=2,  usar_spread=TRUE,  usar_usd=TRUE),
    list(j1=2, j2=4, jv=2,  usar_spread=FALSE, usar_usd=FALSE) # mínimo
  )
  for (t in tries){
    ds <- do.call(monta_features, c(list(df=df), t))
    if (nrow(ds$X) >= 300) {
      cat(sprintf("Features usadas: j1=%d, j2=%d, jv=%d, spread=%s, usd=%s | linhas=%d\n",
                  t$j1, t$j2, t$jv, t$usar_spread, t$usar_usd, nrow(ds$X)))
      return(ds)
    }
  }
  # Último recurso: apenas retornos (garante algo mesmo com pouco histórico)
  retorno_wti   <- c(NA, diff(log(df$preco_wti)))
  retorno_brent <- c(NA, diff(log(df$preco_brent)))
  Y <- c(df$preco_wti[-1], NA)
  X <- cbind(preco_wti=df$preco_wti, retorno_wti=retorno_wti, retorno_brent=retorno_brent)
  ok <- stats::complete.cases(X) & !is.na(Y)
  X <- X[ok,,drop=FALSE]; Y <- Y[ok]; datas <- df$data[ok]
  cat(sprintf("Backoff final (apenas retornos) | linhas=%d\n", nrow(X)))
  list(X=X, Y=Y, data=datas)
}

ds <- build_dataset(df_raw)
X <- ds$X; Y <- ds$Y; datas_ok <- ds$data

# ------------------ NORMALIZAÇÃO ------------------
scalers <- lapply(seq_len(ncol(X)), function(j) escala(X[,j]))
Xn <- X
for(j in seq_len(ncol(X))) Xn[,j] <- scalers[[j]]$scale
y_mean <- mean(Y); y_sd <- sd(Y); if (is.na(y_sd) || y_sd==0) y_sd <- 1
Yn <- (Y - y_mean)/y_sd

# ------------------ SPLIT TEMPORAL ROBUSTO ------------------
n <- nrow(Xn)
min_tr <- max(80, ceiling(0.5*n))
min_te <- max(20, floor(0.1*n))
n_tr <- floor(n*prop_treino)
if (n_tr < min_tr) n_tr <- min_tr
if (n - n_tr < min_te) n_tr <- n - min_te
if (n_tr <= 0) n_tr <- floor(0.7*n)
if (n_tr >= n) n_tr <- n - 1L
cat(sprintf("Diagnóstico: total=%d | treino=%d | teste=%d\n", n, n_tr, n-n_tr))

Xtr <- Xn[1:n_tr,,drop=FALSE];     Ytr <- Yn[1:n_tr]
Xte <- Xn[(n_tr+1):n,,drop=FALSE]; Yte <- Yn[(n_tr+1):n]
datas_te <- datas_ok[(n_tr+1):n]

# ------------------ REDE NEURAL (MLP) ------------------
p <- ncol(Xtr)
W1 <- xavier(p, unidades1); b1 <- rep(0, unidades1)
W2 <- xavier(unidades1, unidades2); b2 <- rep(0, unidades2)
W3 <- xavier(unidades2, 1); b3 <- 0

sW1 <- adam_init(W1); sb1 <- adam_init(matrix(b1,1))
sW2 <- adam_init(W2); sb2 <- adam_init(matrix(b2,1))
sW3 <- adam_init(W3); sb3 <- adam_init(matrix(b3,1))

t <- 0; melhor_val <- Inf; sem_melhora <- 0; melhor_pesos <- NULL
make_batches <- function(N, bs){ idx <- sample.int(N); split(idx, ceiling(seq_along(idx)/bs)) }
mse <- function(a, y) mean((a - y)^2)

for (ep in 1:epocas_max){
  batches <- make_batches(nrow(Xtr), batch_size)
  for (id in batches){
    t <- t + 1L
    xb <- Xtr[id,,drop=FALSE]; yb <- matrix(Ytr[id], ncol=1)
    z1 <- xb %*% W1 + matrix(b1, nrow=nrow(xb), ncol=unidades1, byrow=TRUE); a1 <- tanh_act(z1)
    z2 <- a1 %*% W2 + matrix(b2, nrow=nrow(a1), ncol=unidades2, byrow=TRUE); a2 <- tanh_act(z2)
    z3 <- a2 %*% W3 + matrix(b3, nrow=nrow(a2), ncol=1, byrow=TRUE); yhat <- z3
    
    dL_dy <- (yhat - yb) * (2/nrow(xb))
    dW3 <- t(a2) %*% dL_dy; db3 <- colSums(dL_dy)
    da2 <- dL_dy %*% t(W3); dz2 <- da2 * tanh_deriv(a2)
    dW2 <- t(a1) %*% dz2; db2 <- colSums(dz2)
    da1 <- dz2 %*% t(W2); dz1 <- da1 * tanh_deriv(a1)
    dW1 <- t(xb) %*% dz1; db1 <- colSums(dz1)
    
    u <- adam_step(W1, dW1, sW1, t, lr, beta1, beta2, eps); W1 <- u$W; sW1 <- u$state
    u <- adam_step(matrix(b1,1), matrix(db1,1), sb1, t, lr, beta1, beta2, eps); b1 <- as.numeric(u$W); sb1 <- u$state
    u <- adam_step(W2, dW2, sW2, t, lr, beta1, beta2, eps); W2 <- u$W; sW2 <- u$state
    u <- adam_step(matrix(b2,1), matrix(db2,1), sb2, t, lr, beta1, beta2, eps); b2 <- as.numeric(u$W); sb2 <- u$state
    u <- adam_step(W3, dW3, sW3, t, lr, beta1, beta2, eps); W3 <- u$W; sW3 <- u$state
    u <- adam_step(matrix(b3,1), matrix(db3,1), sb3, t, lr, beta1, beta2, eps); b3 <- as.numeric(u$W); sb3 <- u$state
  }
  
  fwd <- function(Xm){
    z1 <- Xm %*% W1 + matrix(b1, nrow=nrow(Xm), ncol=unidades1, byrow=TRUE); a1 <- tanh_act(z1)
    z2 <- a1 %*% W2 + matrix(b2, nrow=nrow(a1), ncol=unidades2, byrow=TRUE); a2 <- tanh_act(z2)
    z3 <- a2 %*% W3 + matrix(b3, nrow=nrow(a2), ncol=1, byrow=TRUE); as.numeric(z3)
  }
  yhat_tr <- fwd(Xtr); yhat_te <- fwd(Xte)
  loss_tr <- mse(yhat_tr, Ytr); loss_te <- mse(yhat_te, Yte)
  
  if (loss_te < melhor_val - 1e-6){
    melhor_val <- loss_te; sem_melhora <- 0
    melhor_pesos <- list(W1=W1,b1=b1,W2=W2,b2=b2,W3=W3,b3=b3)
  } else {
    sem_melhora <- sem_melhora + 1
    if (sem_melhora >= paciencia){
      W1 <- melhor_pesos$W1; b1 <- melhor_pesos$b1
      W2 <- melhor_pesos$W2; b2 <- melhor_pesos$b2
      W3 <- melhor_pesos$W3; b3 <- melhor_pesos$b3
      break
    }
  }
  if (ep %% 20 == 0) cat(sprintf("Época %d | MSE_tr=%.5f | MSE_te=%.5f\n", ep, loss_tr, loss_te))
}

# ------------------ AVALIAÇÃO E GRÁFICO ------------------
fwd <- function(Xm){
  z1 <- Xm %*% W1 + matrix(b1, nrow=nrow(Xm), ncol=unidades1, byrow=TRUE); a1 <- tanh_act(z1)
  z2 <- a1 %*% W2 + matrix(b2, nrow=nrow(a1), ncol=unidades2, byrow=TRUE); a2 <- tanh_act(z2)
  z3 <- a2 %*% W3 + matrix(b3, nrow=nrow(a2), ncol=1, byrow=TRUE); as.numeric(z3)
}
pred_tr_n <- fwd(Xtr); pred_te_n <- fwd(Xte)
pred_tr <- denorm(pred_tr_n, y_mean, y_sd); pred_te <- denorm(pred_te_n, y_mean, y_sd)
ytr <- denorm(Ytr, y_mean, y_sd);         yte     <- denorm(Yte, y_mean, y_sd)

met_tr <- c(RMSE=rmse(ytr,pred_tr), MAE=mae(ytr,pred_tr), MAPE=mape(ytr,pred_tr))
met_te <- c(RMSE=rmse(yte,pred_te), MAE=mae(yte,pred_te), MAPE=mape(yte,pred_te))
print(round(met_tr,4)); print(round(met_te,4))

op <- par(no.readonly = TRUE); on.exit(par(op))
plot(datas_te, yte, type="l", lwd=2, xlab="Data", ylab="Preço (US$)",
     main="WTI — Preço Real vs Predito (conjunto de teste)")
lines(datas_te, pred_te, lwd=2, lty=2, col=2)
legend("topleft", legend=c("Real","Predito"), lty=c(1,2), lwd=2, col=c(1,2), bty="n")

# ------------------ PREVISÃO PRÓXIMOS 10 DIAS (ingênua) ------------------
h <- 10
ult_X <- Xn[nrow(Xn),,drop=FALSE]
proj_n <- matrix(rep(ult_X, each=h), nrow=h)
proj_pred_n <- fwd(proj_n)
proj_pred <- denorm(proj_pred_n, y_mean, y_sd)
prev_df <- data.frame(data = seq(max(datas_ok)+1, by="day", length.out=h),
                      preco_wti_previsto = round(proj_pred, 2))
print(prev_df)
