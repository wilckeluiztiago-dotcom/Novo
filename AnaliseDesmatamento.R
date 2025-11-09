# ============================================================
# Análise de Risco de Desmatamento — MLP em R Puro (do zero)
# Autor: Luiz Tiago Wilcke (LT)
# Descrição:
#   - Gera (ou lê) base com variáveis ambientais/socioeconômicas.
#   - Implementa MLP do zero (forward/backprop) p/ classificação binária.
#   - Regularização L2, dropout, early stopping, semente reprodutível.
#   - Split temporal: treino/validação/teste.
#   - Muitas figuras: séries, matriz de correlação, curvas ROC/PR,
#     matriz de confusão, calibração, PDP, importância, heatmaps sintéticos,
#     curvas de aprendizado e de perda.
# Saídas:
#   - CSVs de previsões, métricas, importância por permutação.# ============================================================
# Análise de Risco de Desmatamento — MLP em R Puro (do zero)
#   - Gráficos em tela (e PNG se quiser).
# ============================================================

rm(list=ls()); gc()
set.seed(42)

# -------------------------- UTILIDADES --------------------------
escala_padrao <- function(X){
  mu <- apply(X, 2, mean)
  sdv <- apply(X, 2, sd)
  sdv[sdv == 0] <- 1
  Xs <- sweep(sweep(X, 2, mu, "-"), 2, sdv, "/")
  list(X= Xs, mu=mu, sd=sdv)
}
aplicar_escala <- function(X, mu, sd){
  sd[sd == 0] <- 1
  sweep(sweep(X, 2, mu, "-"), 2, sd, "/")
}

# --- Ativações que preservam DIMENSÕES ---
relu <- function(z) {
  out <- pmax(0, z)
  dim(out) <- dim(z)  # garante matriz
  out
}
drelu <- function(z) {
  out <- (z > 0) * 1.0
  dim(out) <- dim(z)  # garante matriz
  out
}
sigmoid <- function(z) {
  out <- 1/(1 + exp(-z))
  dim(out) <- dim(z)  # garante matriz
  out
}
dsigmoid <- function(a) {
  out <- a * (1 - a)
  dim(out) <- dim(a)   # garante matriz
  out
}

# Funções métricas (sem pacotes)
logloss_bin <- function(y, p){
  eps <- 1e-12; p <- pmin(pmax(p, eps), 1-eps)
  -mean(y*log(p) + (1-y)*log(1-p))
}
acc <- function(y, p, thr=0.5){
  mean((p>=thr) == y)
}
prec_rec_f1 <- function(y, p, thr=0.5){
  yp <- as.integer(p>=thr)
  TP <- sum(yp==1 & y==1)
  FP <- sum(yp==1 & y==0)
  FN <- sum(yp==0 & y==1)
  prec <- ifelse(TP+FP==0, 0, TP/(TP+FP))
  rec  <- ifelse(TP+FN==0, 0, TP/(TP+FN))
  f1   <- ifelse(prec+rec==0, 0, 2*prec*rec/(prec+rec))
  c(prec=prec, rec=rec, f1=f1)
}
roc_curve <- function(y, p){
  thr <- sort(unique(c(0, p, 1)))
  TPR <- FPR <- numeric(length(thr))
  P <- sum(y==1); N <- sum(y==0)
  for(i in seq_along(thr)){
    t <- thr[i]; yp <- as.integer(p>=t)
    TP <- sum(yp==1 & y==1); FP <- sum(yp==1 & y==0)
    TPR[i] <- ifelse(P==0,0,TP/P)
    FPR[i] <- ifelse(N==0,0,FP/N)
  }
  ord <- order(FPR, TPR)
  x <- FPR[ord]; y2 <- TPR[ord]
  auc <- sum(diff(x) * (head(y2,-1)+tail(y2,-1))/2)
  list(FPR=x, TPR=y2, AUC=auc)
}
pr_curve <- function(y, p){
  thr <- sort(unique(c(0, p, 1)))
  Ptot <- sum(y==1)
  Prec <- Rec <- numeric(length(thr))
  for(i in seq_along(thr)){
    t <- thr[i]; yp <- as.integer(p>=t)
    TP <- sum(yp==1 & y==1); FP <- sum(yp==1 & y==0)
    Rec[i] <- ifelse(Ptot==0, 0, TP/Ptot)
    Prec[i] <- ifelse(TP+FP==0, 1, TP/(TP+FP))
  }
  ord <- order(Rec, Prec)
  r <- Rec[ord]; q <- Prec[ord]
  ap <- sum(diff(r) * (head(q,-1)+tail(q,-1))/2)
  list(Recall=r, Precision=q, AP=ap)
}

# ------------------ GERAÇÃO/LEITURA DE DADOS ------------------
usar_csv <- FALSE
caminho_csv <- NULL  # "desmatamento.csv" // precisa colunas como abaixo

if(usar_csv && !is.null(caminho_csv) && file.exists(caminho_csv)){
  dados <- read.csv(caminho_csv)
} else {
  # Dados sintéticos realistas (5 anos mensais x 200 células)
  anos <- 2018:2022
  meses <- 1:12
  celulas <- 1:200
  reg <- expand.grid(ano=anos, mes=meses, id_celula=celulas)
  n <- nrow(reg)
  
  reg$precipitacao        <- round(200 + 80*sin(2*pi*reg$mes/12) + rnorm(n,0,40))
  reg$temperatura         <- round(22 + 7*cos(2*pi*(reg$mes-2)/12) + rnorm(n,0,2),1)
  reg$ndvi                <- pmin(pmax(0.2 + 0.3*sin(2*pi*(reg$mes+1)/12) + rnorm(n,0,0.08), 0), 1)
  reg$evi                 <- pmin(pmax(0.15 + 0.25*sin(2*pi*reg$mes/12) + rnorm(n,0,0.07), 0), 1)
  reg$indice_seca         <- pmin(pmax(0.4 - (reg$precipitacao-200)/300 + rnorm(n,0,0.1),0),1)
  reg$proximidade_estradas<- rexp(n, rate=1/30)  # km
  reg$densidade_pop       <- pmax(0, rnorm(n, 15, 8))
  reg$preco_soja          <- round(90 + 10*sin(2*pi*(reg$ano-2017)/5) + rnorm(n,0,8),1)
  reg$preco_gado          <- round(260 + 20*cos(2*pi*(reg$ano-2017)/5) + rnorm(n,0,12),1)
  reg$pressao_agricola    <- pmin(1, pmax(0, 0.3 + 0.25*(reg$preco_soja-90)/20 + 0.15*(reg$preco_gado-260)/20 + rnorm(n,0,0.1)))
  reg$dist_uc             <- rexp(n, rate=1/40)  # km
  reg$declividade         <- pmin(30, pmax(0, rnorm(n, 8, 5)))
  reg$altitude            <- pmax(0, rnorm(n, 200, 120))
  reg$area_queimada       <- pmax(0, rnorm(n, 5 + 2*reg$indice_seca, 3))
  reg$historico_alertas   <- pmax(0, rpois(n, lambda=0.8 + 0.5*reg$pressao_agricola + 0.3*reg$indice_seca))
  reg$governanca          <- pmin(1, pmax(0, 0.6 - 0.2*(reg$densidade_pop>20) + rnorm(n,0,0.1)))
  reg$dist_frente_agro    <- rexp(n, rate=1/25)
  
  z <- -2.4 +
    0.9*reg$pressao_agricola +
    0.7*reg$indice_seca +
    0.5*(1 - reg$governanca) +
    0.4*pmax(0, (15 - reg$proximidade_estradas)/15) +
    0.35*pmax(0, (10 - reg$dist_uc)/10) +
    0.25*(reg$historico_alertas>1) +
    0.2*(reg$ndvi<0.35) +
    0.15*(reg$evi<0.25) +
    0.1*(reg$area_queimada>6) +
    0.1*(reg$dist_frente_agro<8) +
    0.05*(reg$declividade<5) +
    rnorm(n,0,0.3)
  
  pr <- 1/(1+exp(-z))
  reg$desmatou <- rbinom(n, 1, pr)
  dados <- reg
}

# Ordena temporalmente
dados$data_ord <- with(dados, ano*100 + mes)
dados <- dados[order(dados$data_ord, dados$id_celula), ]

# -------------------- DIVISÃO TREINO/VAL/TESTE --------------------
datas <- sort(unique(dados$data_ord))
nT <- length(datas)
idx_treino <- datas[1:floor(0.6*nT)]
idx_valid  <- datas[(floor(0.6*nT)+1):floor(0.8*nT)]
idx_teste  <- datas[(floor(0.8*nT)+1):nT]

col_alvo <- "desmatou"
col_id   <- c("ano","mes","id_celula","data_ord")
col_x <- setdiff(colnames(dados), c(col_alvo, col_id))

X <- as.matrix(dados[, col_x])
y <- as.integer(dados[[col_alvo]])

X_tr <- X[dados$data_ord %in% idx_treino, , drop=FALSE]
y_tr <- y[dados$data_ord %in% idx_treino]
X_va <- X[dados$data_ord %in% idx_valid , , drop=FALSE]
y_va <- y[dados$data_ord %in% idx_valid]
X_te <- X[dados$data_ord %in% idx_teste , , drop=FALSE]
y_te <- y[dados$data_ord %in% idx_teste]

# Escalonamento com estatísticas do treino
esc <- escala_padrao(X_tr)
X_trs <- esc$X
X_vas <- aplicar_escala(X_va, esc$mu, esc$sd)
X_tes <- aplicar_escala(X_te, esc$mu, esc$sd)

# ------------------ MLP DO ZERO (BINÁRIA) ------------------
mlp_criar <- function(n_in, n_hidden=c(32,16), ativ=c("relu","relu"),
                      seed=42){
  set.seed(seed)
  L <- length(n_hidden)
  params <- list()
  fan_in <- n_in
  for(l in 1:L){
    fan_out <- n_hidden[l]
    # He init p/ ReLU
    W <- rnorm(fan_in*fan_out, 0, sqrt(2/fan_in))
    W <- matrix(W, nrow=fan_in, ncol=fan_out)
    b <- rep(0, fan_out)
    params[[paste0("W",l)]] <- W
    params[[paste0("b",l)]] <- b
    fan_in <- fan_out
  }
  # Camada de saída (1 neurônio)
  Wout <- rnorm(fan_in*1, 0, sqrt(2/fan_in))
  Wout <- matrix(Wout, nrow=fan_in, ncol=1)
  bout <- 0
  params$Wout <- Wout; params$bout <- bout
  params$ativ <- ativ
  params
}

mlp_forward <- function(X, params, dropout_p=0){
  L <- length(params$ativ)
  A <- list(); Z <- list(); M <- list()
  a <- X
  for(l in 1:L){
    W <- params[[paste0("W",l)]]
    b <- params[[paste0("b",l)]]
    z <- a %*% W + matrix(b, nrow=nrow(a), ncol=ncol(W), byrow=TRUE)
    
    if(params$ativ[l]=="relu"){
      a <- relu(z)
      da <- drelu(z)
    } else if(params$ativ[l]=="sigmoid"){
      a <- sigmoid(z)
      da <- dsigmoid(a)
    } else {
      a <- z
      dim(a) <- dim(z)
      da <- matrix(1, nrow=nrow(z), ncol=ncol(z))
    }
    
    # Defesa extra
    if (is.null(dim(a))) dim(a) <- dim(z)
    
    # Dropout (apenas no treino): máscara compatível com a
    if(dropout_p>0){
      mask <- matrix(
        rbinom(n = length(a), size = 1, prob = 1 - dropout_p),
        nrow = nrow(a), ncol = ncol(a)
      )
      a <- a * mask/(1 - dropout_p)
      M[[l]] <- mask
    } else {
      M[[l]] <- matrix(1, nrow=nrow(a), ncol=ncol(a))
    }
    
    A[[l]] <- a; Z[[l]] <- z
  }
  # Saída
  z_out <- a %*% params$Wout + matrix(params$bout, nrow = nrow(a), ncol = 1)
  p <- sigmoid(z_out)
  list(p=p, A=A, Z=Z, M=M)
}

mlp_treinar <- function(X, y, Xv, yv,
                        n_hidden=c(64,32), ativ=c("relu","relu"),
                        lr=1e-3, l2=1e-4, dropout_p=0.1,
                        batch=256, epocas=200,
                        paciencia=20, seed=42, verbose=TRUE){
  set.seed(seed)
  n_in <- ncol(X)
  params <- mlp_criar(n_in, n_hidden, ativ, seed)
  melhor <- list(loss=Inf, params=NULL, epoca=0)
  sem_melhora <- 0
  
  n <- nrow(X)
  passos <- ceiling(n/batch)
  
  hist <- data.frame(epoca=integer(), loss_tr=double(), loss_va=double(),
                     acc_tr=double(), acc_va=double(), stringsAsFactors=FALSE)
  
  for(ep in 1:epocas){
    # embaralha
    ord <- sample(n)
    X <- X[ord,,drop=FALSE]; y <- y[ord]
    
    for(s in 1:passos){
      ini <- (s-1)*batch + 1
      fim <- min(s*batch, n)
      xb <- X[ini:fim,,drop=FALSE]
      yb <- matrix(y[ini:fim], ncol=1)
      
      # forward com dropout
      fw <- mlp_forward(xb, params, dropout_p=dropout_p)
      p  <- fw$p
      
      # perda + L2 (todas as camadas + saída)
      Lc <- length(params$ativ)
      l2sum <- 0
      for(l in 1:Lc){ l2sum <- l2sum + sum(params[[paste0("W",l)]]^2) }
      loss <- logloss_bin(yb, p) + l2*l2sum + l2*sum(params$Wout^2)
      
      # backprop
      eps <- 1e-12; p <- pmin(pmax(p, eps), 1-eps)
      dz_out <- (p - yb) # derivada da BCE para sigmoid
      dWout <- t(fw$A[[Lc]]) %*% dz_out + 2*l2*params$Wout
      dbout <- sum(dz_out)
      
      # propaga para trás pelas camadas ocultas
      da <- dz_out %*% t(params$Wout)
      for(l in Lc:1){
        if(params$ativ[l]=="relu"){
          dz <- da * drelu(fw$Z[[l]])
        } else if(params$ativ[l]=="sigmoid"){
          a_l <- fw$A[[l]]
          dz <- da * dsigmoid(a_l)
        } else {
          dz <- da
        }
        # aplica máscara de dropout (mesma do forward)
        dz <- dz * fw$M[[l]]
        
        a_ant <- if(l==1) xb else fw$A[[l-1]]
        dW <- t(a_ant) %*% dz + 2*l2*params[[paste0("W",l)]]
        db <- colSums(dz)
        
        # atualização (SGD)
        params[[paste0("W",l)]] <- params[[paste0("W",l)]] - lr*dW
        params[[paste0("b",l)]] <- params[[paste0("b",l)]] - lr*db
        
        if(l>1){
          da <- dz %*% t(params[[paste0("W",l)]])
        } else {
          da <- dz %*% t(params[[paste0("W",l)]])
        }
      }
      # atualiza saída
      params$Wout <- params$Wout - lr*dWout
      params$bout <- params$bout - lr*dbout
    }
    
    # Avaliação por época (sem dropout)
    p_tr <- mlp_forward(X,  params, dropout_p=0)$p[,1]
    p_va <- mlp_forward(Xv, params, dropout_p=0)$p[,1]
    ll_tr <- logloss_bin(y, p_tr); ll_va <- logloss_bin(yv, p_va)
    a_tr  <- acc(y, p_tr);        a_va  <- acc(yv, p_va)
    
    hist <- rbind(hist, data.frame(epoca=ep, loss_tr=ll_tr, loss_va=ll_va,
                                   acc_tr=a_tr, acc_va=a_va))
    if(verbose && ep%%5==0){
      cat(sprintf("[Época %03d] loss_tr=%.4f | loss_va=%.4f | acc_va=%.3f\n", ep, ll_tr, ll_va, a_va))
    }
    
    # Early stopping pela validação
    if(ll_va + 1e-6 < melhor$loss){
      melhor$loss <- ll_va
      melhor$params <- params
      melhor$epoca <- ep
      sem_melhora <- 0
    } else {
      sem_melhora <- sem_melhora + 1
      if(sem_melhora >= paciencia){
        if(verbose) cat(sprintf("Parando cedo na época %d (melhor=%d, loss_va=%.4f)\n",
                                ep, melhor$epoca, melhor$loss))
        break
      }
    }
  }
  
  list(params=melhor$params, historico=hist, epoca_melhor=melhor$epoca)
}

mlp_prever <- function(X, params){
  mlp_forward(X, params, dropout_p=0)$p[,1]
}

# ---------------------- TREINAR MODELO ----------------------
modelo <- mlp_treinar(
  X_trs, y_tr, X_vas, y_va,
  n_hidden=c(64,32), ativ=c("relu","relu"),
  lr=2e-3, l2=5e-4, dropout_p=0.2,
  batch=512, epocas=250, paciencia=25, seed=42, verbose=TRUE
)

params <- modelo$params

# ---------------------- AVALIAÇÃO FINAL ----------------------
p_tr <- mlp_prever(X_trs, params)
p_va <- mlp_prever(X_vas, params)
p_te <- mlp_prever(X_tes, params)

# Escolher threshold pelo melhor F1 na validação
ths <- seq(0.05, 0.95, by=0.01)
f1s <- sapply(ths, function(t) prec_rec_f1(y_va, p_va, t)["f1"])
thr_melhor <- ths[which.max(f1s)]

metricas <- function(y, p, thr){
  c(
    logloss = logloss_bin(y,p),
    acc     = acc(y,p,thr),
    prec_rec_f1(y,p,thr)
  )
}

met_tr <- metricas(y_tr, p_tr, thr_melhor)
met_va <- metricas(y_va, p_va, thr_melhor)
met_te <- metricas(y_te, p_te, thr_melhor)

cat("\n--- Métricas ---\n")
print(round(met_tr,3))
print(round(met_va,3))
print(round(met_te,3))
cat(sprintf("Limiar escolhido (validação): %.2f\n", thr_melhor))

# ----------------------- GRÁFICOS -----------------------
par(mfrow=c(2,2))
# 1) Curvas de perda e acurácia
plot(modelo$historico$epoca, modelo$historico$loss_tr, type="l", lwd=2,
     xlab="Época", ylab="Perda (logloss)",
     main="Curva de Perda (Treino vs Validação)")
lines(modelo$historico$epoca, modelo$historico$loss_va, lwd=2, lty=2)
legend("topright", c("Treino","Validação"), lty=c(1,2), lwd=2, bty="n")

plot(modelo$historico$epoca, modelo$historico$acc_tr, type="l", lwd=2,
     xlab="Época", ylab="Acurácia",
     main="Curva de Acurácia (Treino vs Validação)")
lines(modelo$historico$epoca, modelo$historico$acc_va, lwd=2, lty=2)
legend("bottomright", c("Treino","Validação"), lty=c(1,2), lwd=2, bty="n")

# 2) Matriz de correlação (amostra)
am <- cor(as.matrix(dados[sample(nrow(dados), min(3000,nrow(dados))), col_x]))
image(1:ncol(am), 1:ncol(am), t(am)[,ncol(am):1],
      xaxt="n", yaxt="n", main="Correlação entre Variáveis")
axis(1, at=1:ncol(am), labels=col_x, las=2, cex.axis=0.6)
axis(2, at=1:ncol(am), labels=rev(col_x), las=2, cex.axis=0.6)
box()

# 3) Distribuição de probabilidades no teste
hist(p_te, breaks=30, main="Distribuição p(desmatou) — Teste", xlab="Probabilidade")

par(mfrow=c(1,1))

# 4) ROC e PR no teste
roc_te <- roc_curve(y_te, p_te)
pr_te  <- pr_curve(y_te, p_te)

par(mfrow=c(1,2))
plot(roc_te$FPR, roc_te$TPR, type="l", lwd=2,
     xlab="FPR", ylab="TPR", main=sprintf("ROC (AUC=%.3f)", roc_te$AUC))
abline(0,1,lty=3)

plot(pr_te$Recall, pr_te$Precision, type="l", lwd=2,
     xlab="Recall", ylab="Precisão",
     main=sprintf("Precisão-Recall (AP=%.3f)", pr_te$AP))

par(mfrow=c(1,1))

# 5) Matriz de confusão no teste
yp <- as.integer(p_te >= thr_melhor)
TP <- sum(yp==1 & y_te==1)
FP <- sum(yp==1 & y_te==0)
FN <- sum(yp==0 & y_te==1)
TN <- sum(yp==0 & y_te==0)
mat <- matrix(c(TP,FP,FN,TN), nrow=2, byrow=TRUE)
colnames(mat) <- c("Positivo","Negativo")
rownames(mat) <- c("Previsto Pos","Previsto Neg")
par(mar=c(4,4,2,1))
image(1:2, 1:2, t(mat)[,2:1], xaxt="n", yaxt="n",
      main="Matriz de Confusão (Teste)")
axis(1, at=1:2, labels=colnames(mat), las=1)
axis(2, at=1:2, labels=rev(rownames(mat)), las=1)
text(rep(1:2, each=2), 3 - rep(1:2,2), labels=as.vector(mat), cex=1.4, font=2)

# 6) Calibração / Confiabilidade
bins <- cut(p_te, breaks=seq(0,1,by=0.05), include.lowest=TRUE)
cal <- aggregate(list(obs=y_te, prob=p_te), by=list(bin=bins),
                 FUN=function(z) c(mean=mean(z)))
obs_bin <- sapply(cal$obs, function(v) v[1])
prb_bin <- sapply(cal$prob, function(v) v[1])
mid <- sapply(strsplit(as.character(cal$bin), ","), function(s){
  lo <- as.numeric(sub("\\(|\\[","",s[1]))
  hi <- as.numeric(sub("\\]","",s[2])); (lo+hi)/2
})
plot(mid, prb_bin, type="b", pch=16, ylim=c(0,1), xlim=c(0,1),
     xlab="Previsão média no bin", ylab="Frequência observada",
     main="Confiabilidade / Calibração (Teste)")
abline(0,1,lty=2)
points(mid, obs_bin, type="p", pch=17)
legend("topleft", c("Prob. Média","Obs. Média"), pch=c(16,17), bty="n")

# 7) Importância por permutação (Teste)
perm_importancia <- function(X, y, p_ref, params, thr, nperm=1){
  base_acc <- acc(y, p_ref, thr)
  imp <- numeric(ncol(X))
  for(j in 1:ncol(X)){
    queda <- 0
    for(k in 1:nperm){
      Xb <- X; Xb[,j] <- sample(Xb[,j])
      pb <- mlp_prever(Xb, params)
      queda <- queda + (base_acc - acc(y, pb, thr))
    }
    imp[j] <- queda/nperm
  }
  names(imp) <- colnames(X)
  imp
}
imp_te <- perm_importancia(X_tes, y_te, p_te, params, thr_melhor, nperm=1)
imp_ord <- sort(imp_te, decreasing=TRUE)

par(mfrow=c(1,1))
barplot(imp_ord[1:min(12,length(imp_ord))],
        las=2, main="Importância por Permutação (Top 12)",
        ylab="Queda na acurácia")

# 8) Partial Dependence (PDP) para algumas variáveis
pdp_var <- head(names(imp_ord), 4)
par(mfrow=c(2,2))
for(v in pdp_var){
  xs <- X_tes
  gridv <- seq(quantile(xs[,v],0.05), quantile(xs[,v],0.95), length.out=25)
  mp <- numeric(length(gridv))
  for(i in seq_along(gridv)){
    xs[,v] <- gridv[i]
    mp[i] <- mean(mlp_prever(xs, params))
  }
  plot(gridv, mp, type="l", lwd=2, xlab=v, ylab="p(desmatou)",
       main=paste("PDP —", v))
}
par(mfrow=c(1,1))

# 9) Heatmap espacial sintético (prob média por célula no último ano do teste)
ult_ano <- max(dados$ano[dados$data_ord %in% idx_teste])
sub_ult <- subset(dados, ano==ult_ano)
lado_x <- 20; lado_y <- 10  # 20x10 = 200 células
if(length(unique(sub_ult$id_celula)) >= lado_x*lado_y){
  ids_ord <- sort(unique(sub_ult$id_celula))[1:(lado_x*lado_y)]
  sub_ult <- sub_ult[sub_ult$id_celula %in% ids_ord,]
  linhas_teste <- which(dados$data_ord %in% idx_teste)
  # mapeia a probabilidade média por célula (ordem simples)
  idx_map <- match(sub_ult$id_celula, dados$id_celula[linhas_teste])
  p_map <- p_te[idx_map]
  M <- matrix(p_map[1:(lado_x*lado_y)], nrow=lado_y, byrow=TRUE)
  image(1:lado_x, 1:lado_y, t(M)[,lado_y:1], main=paste("Mapa Sintético — p(desmatou),", ult_ano),
        xlab="lon (índice)", ylab="lat (índice)")
}

# 10) Séries temporais agregadas por mês (taxa observada vs prevista)
agg <- aggregate(list(obs=dados[[col_alvo]]),
                 by=list(data=dados$data_ord), mean)
pred_por_data <- function(dados, Xs, idx_datas, p_vec=NULL){
  dt <- unique(dados$data_ord[dados$data_ord %in% idx_datas])
  out <- data.frame(data=dt, prev=NA_real_)
  for(i in 1:nrow(out)){
    d <- out$data[i]
    if(d %in% idx_teste){
      linhas_teste <- which(dados$data_ord %in% idx_teste)
      mask <- dados$data_ord[linhas_teste]==d
      out$prev[i] <- mean(p_te[mask], na.rm=TRUE)
    } else if(d %in% idx_valid){
      linhas_valid <- which(dados$data_ord %in% idx_valid)
      mask <- dados$data_ord[linhas_valid]==d
      out$prev[i] <- mean(p_va[mask], na.rm=TRUE)
    } else {
      linhas_treino <- which(dados$data_ord %in% idx_treino)
      mask <- dados$data_ord[linhas_treino]==d
      out$prev[i] <- mean(p_tr[mask], na.rm=TRUE)
    }
  }
  out[order(out$data),]
}
agg_prev <- pred_por_data(dados, NULL, datas, NULL)
par(mfrow=c(1,1))
plot(agg$data, agg$obs, type="l", lwd=2, ylim=c(0,1),
     xlab="AAAAMM", ylab="Taxa média",
     main="Série Temporal — Observado vs Previsto (taxa média)")
lines(agg_prev$data, agg_prev$prev, lwd=2, lty=2)
legend("topleft", c("Observado","Previsto"), lty=c(1,2), lwd=2, bty="n")

# ---------------------- EXPORTAR SAÍDAS ----------------------
dir.create("saidas_desmatamento", showWarnings=FALSE)
# CSV previsões
write.csv(data.frame(split=c(rep("treino",length(p_tr)),rep("valid",length(p_va)),rep("teste",length(p_te))),
                     prob=c(p_tr,p_va,p_te)),
          "saidas_desmatamento/probabilidades.csv", row.names=FALSE)
# métricas
met_df <- rbind(
  cbind(split="treino", t(met_tr)),
  cbind(split="valid",  t(met_va)),
  cbind(split="teste",  t(met_te))
)
write.csv(met_df, "saidas_desmatamento/metricas.csv", row.names=FALSE)
# importância
imp_tab <- data.frame(variavel=names(imp_ord), importancia=as.numeric(imp_ord))
write.csv(imp_tab, "saidas_desmatamento/importancia_permutacao.csv", row.names=FALSE)

cat("\nArquivos salvos em: ./saidas_desmatamento/\n")

# ---------------------- COMO USAR DADOS REAIS ----------------------
# dados <- read.csv("desmatamento.csv")
# Esperado:
#   colunas: ano, mes, id_celula, desmatou (0/1),
#            precipitacao, temperatura, ndvi, evi, indice_seca,
#            proximidade_estradas, densidade_pop, preco_soja, preco_gado,
#            pressao_agricola, dist_uc, declividade, altitude,
#            area_queimada, historico_alertas, governanca, dist_frente_agro
# Repita o mesmo pipeline de split/escala/treino.

