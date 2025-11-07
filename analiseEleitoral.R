# ========================================================================
# Análise Eleitoral Avançada (Frequentista L1 + Bayesiano MH)
# Autor: Luiz Tiago Wilcke (LT)
# ========================================================================

set.seed(123)

# ============================
# 1) Dados sintéticos
# ============================
n <- 2500
municipios <- paste0("Mun", 1:40)
regioes    <- c("Centro","Zona Norte","Zona Sul","Zona Leste","Zona Oeste")

dados <- data.frame(
  municipio          = sample(municipios, n, TRUE),
  regiao             = sample(regioes,    n, TRUE),
  idade              = pmax(16, round(rnorm(n, 42, 13))),
  genero             = sample(c("Feminino","Masculino","Outro"), n, TRUE, prob = c(0.5,0.49,0.01)),
  renda_mensal       = exp(rnorm(n, log(2800), 0.7)),
  escolaridade       = sample(c("Fundamental","Médio","Superior"), n, TRUE, prob = c(0.33,0.44,0.23)),
  interesse_politica = pmin(pmax(round(rnorm(n, 5.5, 2),1),0),10),
  satisf_governo     = pmin(pmax(round(rnorm(n, 5.0, 2),1),0),10),
  religiosidade      = pmin(pmax(round(rnorm(n, 5.5, 2.5),1),0),10),
  uso_redes          = pmin(pmax(round(rnorm(n, 4.5, 2),1),0),10)
)

# Processo gerador "verdadeiro"
eta <- -1.9 +
  0.40*log(dados$renda_mensal/1000) +
  0.18*(dados$idade/10) +
  0.45*(dados$interesse_politica - 5) +
  0.50*(dados$satisf_governo - 5) -
  0.25*(dados$uso_redes - 4) +
  ifelse(dados$escolaridade=="Superior", 0.35, ifelse(dados$escolaridade=="Médio", 0.12, 0)) +
  ifelse(dados$genero=="Feminino", 0.10, 0) +
  ifelse(dados$regiao=="Centro", 0.25, 0)

efeito_mun <- rnorm(length(municipios), 0, 0.35); names(efeito_mun) <- municipios
eta <- eta + efeito_mun[dados$municipio]

prob_A <- 1/(1+exp(-eta))
dados$voto_A <- rbinom(n, 1, prob_A)
dados$peso   <- runif(n, 0.7, 1.3)

# Garante que variáveis categóricas são factors
dados$regiao       <- factor(dados$regiao, levels = regioes)
dados$genero       <- factor(dados$genero, levels = c("Feminino","Masculino","Outro"))
dados$escolaridade <- factor(dados$escolaridade, levels = c("Fundamental","Médio","Superior"))

# ============================
# 2) EDA rápida
# ============================
par(mfrow=c(1,2))
hist(dados$renda_mensal, breaks=40, main="Renda mensal", xlab="R$", col="gray")
plot(dados$idade, jitter(dados$voto_A,.05), pch=20, col=rgb(0,0,0,.3),
     main="Idade vs voto_A", ylab="voto_A", xlab="idade")
par(mfrow=c(1,1))

# ============================
# 3) Split e pré-processamento
# ============================
set.seed(42)
idx <- sample.int(n, floor(.8*n))
treino <- dados[idx,]
teste  <- dados[-idx,]

# Mantém os MESMOS níveis no teste
fix_levels <- function(test_df, ref_df, cols){
  for(cl in cols){
    if(is.factor(ref_df[[cl]])){
      test_df[[cl]] <- factor(test_df[[cl]], levels = levels(ref_df[[cl]]))
    }
  }
  test_df
}
teste <- fix_levels(teste, treino, c("regiao","genero","escolaridade"))

# Fórmulas separadas (EVITA o erro no model.matrix):
form_y <- voto_A ~ idade + log(renda_mensal) + escolaridade + genero +
  interesse_politica + satisf_governo + religiosidade +
  uso_redes + regiao
form_x <- ~ idade + log(renda_mensal) + escolaridade + genero +
  interesse_politica + satisf_governo + religiosidade +
  uso_redes + regiao

# Design matrices (agora sem exigir 'voto_A' no model.matrix)
mm_treino <- model.matrix(form_x, data = treino)
mm_teste  <- model.matrix(form_x, data = teste)
y_treino  <- treino$voto_A
y_teste   <- teste$voto_A

# Padronização (exceto intercepto)
scale_train <- function(X){
  m <- apply(X[,-1,drop=FALSE], 2, mean)
  s <- apply(X[,-1,drop=FALSE], 2, sd)
  s[s<1e-8] <- 1e-8
  Xs <- X
  Xs[,-1] <- sweep(sweep(X[,-1,drop=FALSE],2,m,"-"),2,s,"/")
  list(X=xs <- Xs, mean=m, sd=s)
}
sc  <- scale_train(mm_treino)
Xtr <- sc$X
Xte <- mm_teste
Xte[,-1] <- sweep(sweep(Xte[,-1,drop=FALSE],2,sc$mean,"-"),2,sc$sd,"/")

# ============================
# 4) Logística L1 (LASSO) do zero + K-fold CV
# ============================
sigm <- function(z) 1/(1+exp(-z))

loglik <- function(beta, X, y, w=NULL){
  if(is.null(w)) w <- rep(1, length(y))
  eta <- as.vector(X %*% beta)
  p <- sigm(eta)
  sum(w*(y*log(p+1e-12) + (1-y)*log(1-p+1e-12)))
}

lasso_logit_cd <- function(X, y, lambda, maxit=300, tol=1e-6, w=NULL){
  if(is.null(w)) w <- rep(1, length(y))
  p <- ncol(X)
  beta <- rep(0, p)
  for(it in 1:maxit){
    beta_old <- beta
    eta <- as.vector(X %*% beta)
    pvec <- sigm(eta)
    z <- eta + (y - pvec)/(pvec*(1-pvec) + 1e-8)
    W <- (pvec*(1-pvec))*w
    # Intercepto sem penalização
    xj <- X[,1]; rj <- z - (as.vector(X %*% beta) - xj*beta[1])
    beta[1] <- sum(W*xj*rj)/sum(W*xj*xj + 1e-12)
    # Demais coeficientes com soft-threshold
    for(j in 2:p){
      xj <- X[,j]
      rj <- z - (as.vector(X %*% beta) - xj*beta[j])
      aj <- sum(W*xj*xj) + 1e-12
      cj <- sum(W*xj*rj)
      beta[j] <- sign(cj/aj) * pmax(0, abs(cj/aj) - lambda/aj)
    }
    if(max(abs(beta - beta_old)) < tol) break
  }
  list(beta=beta, it=it)
}

AUC <- function(p, y){
  r <- rank(p, ties.method="average"); pos <- y==1
  n1 <- sum(pos); n0 <- sum(!pos)
  if(n1==0 || n0==0) return(NA_real_)
  (sum(r[pos]) - n1*(n1+1)/2)/(n1*n0)
}

roc_curve <- function(p,y){
  th <- sort(unique(c(0, p, 1)))
  out <- matrix(NA, nrow=length(th), ncol=3,
                dimnames=list(NULL, c("th","tpr","fpr")))
  for(i in seq_along(th)){
    t <- th[i]; yhat <- as.integer(p>=t)
    TP <- sum(yhat==1 & y==1); FP <- sum(yhat==1 & y==0)
    FN <- sum(yhat==0 & y==1); TN <- sum(yhat==0 & y==0)
    out[i,] <- c(t,
                 ifelse((TP+FN)==0, 0, TP/(TP+FN)),
                 ifelse((FP+TN)==0, 0, FP/(FP+TN)))
  }
  as.data.frame(out)
}

calibracao <- function(p,y,bins=10){
  br <- seq(0,1,length.out=bins+1)
  mids <- (head(br,-1)+tail(br,-1))/2
  obs  <- rep(NA,bins); nbin <- integer(bins)
  for(i in 1:bins){
    idx <- which(p>=br[i] & p<br[i+1])
    nbin[i] <- length(idx)
    obs[i]  <- if(nbin[i]==0) NA else mean(y[idx])
  }
  data.frame(midpoint=mids, observed_rate=obs, n=nbin)
}

kfold_cv_lasso <- function(X, y, lambdas, K=5, w=NULL, seed=123){
  if(is.null(w)) w <- rep(1, length(y))
  set.seed(seed)
  n <- length(y)
  fold_id <- sample(rep(1:K, length.out=n))
  aucs <- matrix(NA, nrow=length(lambdas), ncol=K)
  for(ki in 1:K){
    tr <- which(fold_id!=ki); va <- which(fold_id==ki)
    Xtr <- X[tr,,drop=FALSE]; ytr <- y[tr]; wtr <- w[tr]
    Xva <- X[va,,drop=FALSE]; yva <- y[va]
    for(li in seq_along(lambdas)){
      fit <- lasso_logit_cd(Xtr, ytr, lambda=lambdas[li], w=wtr)
      pv  <- sigm(as.vector(Xva %*% fit$beta))
      aucs[li, ki] <- {
        r <- rank(pv, ties.method="average")
        pos <- yva==1; n1 <- sum(pos); n0 <- sum(!pos)
        if(n1==0 || n0==0) NA else (sum(r[pos]) - n1*(n1+1)/2)/(n1*n0)
      }
    }
  }
  perf <- rowMeans(aucs, na.rm=TRUE)
  list(lambda_opt=lambdas[which.max(perf)], perf=perf, aucs=aucs)
}

lambdas <- exp(seq(log(0.001), log(1.5), length.out=40))
cvres   <- kfold_cv_lasso(Xtr, y_treino, lambdas, K=5, w=treino$peso)
lambda_star <- cvres$lambda_opt

fit_l1 <- lasso_logit_cd(Xtr, y_treino, lambda=lambda_star, w=treino$peso)
beta_l1 <- fit_l1$beta

pred_tr <- sigm(as.vector(Xtr %*% beta_l1))
pred_te <- sigm(as.vector(Xte %*% beta_l1))
auc_tr  <- AUC(pred_tr, y_treino)
auc_te  <- AUC(pred_te, y_teste)
acc_te  <- mean((pred_te>0.5)==y_teste)

cat("\n[Frequentista L1] AUC_treino =", round(auc_tr,3),
    "| AUC_teste =", round(auc_te,3),
    "| ACC_teste =", round(acc_te,3), "\n")

# Gráficos ROC / Calibração
roc_te <- roc_curve(pred_te, y_teste)
plot(roc_te$fpr, roc_te$tpr, type="l", lwd=2,
     xlab="Falso-Positivo", ylab="Verdadeiro-Positivo",
     main="ROC — Logística L1 (teste)"); abline(0,1,lty=2)

cal_te <- calibracao(pred_te, y_teste, bins=10)
plot(cal_te$midpoint, cal_te$observed_rate, pch=19,
     xlab="Prob. prevista (bin midpoint)", ylab="Taxa observada",
     main="Calibração — Logística L1 (teste)", ylim=c(0,1))
lines(cal_te$midpoint, cal_te$observed_rate); abline(0,1,lty=2)

# Importância (magnitude de coeficientes)
ord <- order(abs(beta_l1[-1]), decreasing=TRUE)
imp <- data.frame(var=colnames(Xtr)[-1][ord], coef=beta_l1[-1][ord])
print(head(imp, 15))

# ============================
# 5) Bayes Hierárquico (MH) — u ~ N(0, sigma_u^2)
# ============================
Xb_tr <- Xtr; Xb_te <- Xte
reg_tr <- as.integer(treino$regiao)
reg_te <- as.integer(teste$regiao)
J <- length(regioes)

logpost <- function(beta, u, logsigma, X, y, reg_idx, s2_beta=25, m0=log(0.5), s2_0=1){
  eta <- as.vector(X %*% beta) + u[reg_idx]
  p <- 1/(1+exp(-eta))
  ll <- sum(y*log(p+1e-12) + (1-y)*log(1-p+1e-12))
  lp_beta <- sum(dnorm(beta, 0, sqrt(s2_beta), log=TRUE))
  sigma <- exp(logsigma)
  lp_u <- sum(dnorm(u, 0, sigma, log=TRUE))
  lp_sig <- dnorm(logsigma, m0, sqrt(s2_0), log=TRUE) + logsigma
  ll + lp_beta + lp_u + lp_sig
}

mh_sampler <- function(X, y, reg_idx, iter=5000, burn=2500,
                       s2_beta=25, m0=log(0.5), s2_0=1,
                       step_beta=0.05, step_u=0.07, step_lsig=0.05, adapt=TRUE, seed=123){
  set.seed(seed)
  p <- ncol(X); J <- max(reg_idx)
  beta <- rep(0, p); u <- rep(0, J); logsigma <- log(0.5)
  keep <- list(beta=matrix(NA, nrow=iter-burn, ncol=p),
               u=matrix(NA, nrow=iter-burn, ncol=J),
               logsigma=numeric(iter-burn))
  acc_b <- acc_u <- acc_s <- 0
  lp <- logpost(beta,u,logsigma,X,y,reg_idx,s2_beta,m0,s2_0)
  
  for(t in 1:iter){
    # beta
    beta_prop <- beta + rnorm(p, 0, step_beta)
    lp_prop <- logpost(beta_prop,u,logsigma,X,y,reg_idx,s2_beta,m0,s2_0)
    if(log(runif(1)) < (lp_prop - lp)){ beta<-beta_prop; lp<-lp_prop; acc_b<-acc_b+1 }
    
    # u_j
    for(j in 1:J){
      u_prop <- u; u_prop[j] <- u[j] + rnorm(1,0,step_u)
      lp_prop <- logpost(beta,u_prop,logsigma,X,y,reg_idx,s2_beta,m0,s2_0)
      if(log(runif(1)) < (lp_prop - lp)){ u<-u_prop; lp<-lp_prop; acc_u<-acc_u+1 }
    }
    
    # logsigma
    ls_prop <- logsigma + rnorm(1,0,step_lsig)
    lp_prop <- logpost(beta,u,ls_prop,X,y,reg_idx,s2_beta,m0,s2_0)
    if(log(runif(1)) < (lp_prop - lp)){ logsigma<-ls_prop; lp<-lp_prop; acc_s<-acc_s+1 }
    
    # adaptação simples
    if(adapt && t%%200==0 && t<=burn){
      ar_b <- acc_b/(200*1); ar_u <- acc_u/(200*J); ar_s <- acc_s/(200*1)
      step_beta <- step_beta * ifelse(ar_b>0.3, 1.2, 0.8)
      step_u    <- step_u    * ifelse(ar_u>0.3, 1.2, 0.8)
      step_lsig <- step_lsig * ifelse(ar_s>0.3, 1.2, 0.8)
      acc_b<-acc_u<-acc_s<-0
    }
    
    if(t>burn){
      keep$beta[t-burn,]    <- beta
      keep$u[t-burn,]       <- u
      keep$logsigma[t-burn] <- logsigma
    }
    if(t%%500==0) cat("Iter",t,"| steps:", round(step_beta,3), round(step_u,3), round(step_lsig,3), "\n")
  }
  keep
}

cat("\n[Bayes MH] Amostrando...\n")
draws <- mh_sampler(Xb_tr, y_treino, reg_tr,
                    iter=5000, burn=2500,
                    step_beta=0.05, step_u=0.07, step_lsig=0.05, adapt=TRUE)

beta_post <- draws$beta
u_post    <- draws$u
ls_post   <- draws$logsigma

post_sigma_u <- mean(exp(ls_post))
cat("[Bayes MH] sigma_u (média posterior) ~", round(post_sigma_u,3), "\n")

pred_bayes_mean <- function(Xte, reg_te, beta_post, u_post, ndraw=1000){
  ndraw <- min(ndraw, nrow(beta_post))
  ss <- sample(1:nrow(beta_post), ndraw)
  P <- numeric(nrow(Xte))
  for(k in ss){
    eta <- as.vector(Xte %*% beta_post[k,]) + u_post[k, reg_te]
    P <- P + 1/(1+exp(-eta))
  }
  P/ndraw
}

p_bayes_te <- pred_bayes_mean(Xb_te, reg_te, beta_post, u_post, ndraw=1200)
auc_bayes  <- AUC(p_bayes_te, y_teste)
cat("[Bayes MH] AUC_teste =", round(auc_bayes,3), "\n")

# ROC Bayes
roc_b <- roc_curve(p_bayes_te, y_teste)
plot(roc_b$fpr, roc_b$tpr, type="l", lwd=2,
     xlab="Falso-Positivo", ylab="Verdadeiro-Positivo",
     main="ROC — Bayes Hierárquico (teste)"); abline(0,1,lty=2)

# IC empíricos dos u_j
ci <- t(apply(u_post, 2, quantile, probs=c(.025,.5,.975)))
colnames(ci) <- c("Q2.5","Med","Q97.5"); rownames(ci) <- regioes
print(round(ci,3))

# ============================
# 6) Efeitos marginais (diferenças finitas) e cenários
# ============================
ef_marginal <- function(Xrow, reg_idx, beta_post, u_post, var_idx, h=0.2, ndraw=800){
  ndraw <- min(ndraw, nrow(beta_post))
  ss <- sample(1:nrow(beta_post), ndraw)
  f <- function(xrow){
    mean( 1/(1+exp(-(as.vector(xrow %*% t(beta_post[ss,])) + u_post[ss, reg_idx]))) )
  }
  x0 <- Xrow; x1 <- Xrow; x1[var_idx] <- x1[var_idx] + h
  (f(x1) - f(x0))/h
}

x_mean <- colMeans(Xb_tr)
vars_cont <- c("idade", "log(renda_mensal)", "interesse_politica",
               "satisf_governo", "religiosidade", "uso_redes")
idx_vars  <- match(vars_cont, colnames(Xb_tr))
reg_idx_centro <- which(regioes=="Centro")

ems <- sapply(idx_vars, function(j) ef_marginal(x_mean, reg_idx_centro, beta_post, u_post, j, h=0.2))
names(ems) <- vars_cont
cat("\nEfeitos marginais (~ponto médio, região Centro):\n")
print(round(ems,4))

cenarios <- data.frame(
  idade = c(25,45,65),
  renda_mensal = c(1500, 5000, 12000),
  escolaridade = c("Médio","Superior","Superior"),
  genero = c("Feminino","Masculino","Masculino"),
  interesse_politica = c(4,7,8),
  satisf_governo = c(3,6,8),
  religiosidade = c(5,6,7),
  uso_redes = c(6,4,2),
  regiao = c("Zona Norte","Centro","Zona Sul")
)
# ajusta níveis para casar com treino
cenarios$regiao       <- factor(cenarios$regiao,       levels=levels(treino$regiao))
cenarios$genero       <- factor(cenarios$genero,       levels=levels(treino$genero))
cenarios$escolaridade <- factor(cenarios$escolaridade, levels=levels(treino$escolaridade))

mm_cen <- model.matrix(form_x, data=cenarios)
mm_cen[,-1] <- sweep(sweep(mm_cen[,-1,drop=FALSE],2,sc$mean,"-"),2,sc$sd,"/")
reg_cen <- as.integer(cenarios$regiao)

# prob médias + IC por amostras
set.seed(7)
S <- min(1500, nrow(beta_post))
ss <- sample(1:nrow(beta_post), S)
pp <- sapply(1:nrow(mm_cen), function(i){
  eta <- mm_cen[rep(i,S),] %*% t(beta_post[ss,]) + u_post[ss, reg_cen[i]]
  as.numeric(1/(1+exp(-eta)))
})
IC <- t(apply(pp, 2, quantile, probs=c(.025,.5,.975)))
colnames(IC) <- c("IC2.5","Mediana","IC97.5")
out_cen <- cbind(cenarios[,c("regiao","idade","renda_mensal","escolaridade","genero")], round(IC,3))
print(out_cen, row.names=FALSE)

# ============================
# 7) Resumo
# ============================
cat("\n=== Resumo interpretativo ===\n")
cat("* Logística L1 — AUC teste:", round(auc_te,3),
    " | calibração visual próxima da diagonal.\n")
cat("* Bayes MH — AUC teste:", round(auc_bayes,3),
    " | sigma_u posterior média:", round(post_sigma_u,3), "\n")
cat("* Sinais consistentes: +log(renda), +interesse, +satisfação; -uso intenso de redes.\n")
cat("* Interceptos aleatórios por região com shrinkage (ICs listados).\n")
cat("* Efeitos marginais por diferenças finitas no ponto médio (região Centro).\n")


