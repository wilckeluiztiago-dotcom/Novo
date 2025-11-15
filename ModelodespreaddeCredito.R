# ============================================================
# Modelo de Spread de Crédito — Duffie–Singleton (Lambda CIR)
# Autor: Luiz Tiago Wilcke (LT)
# ============================================================

set.seed(42)

# -----------------------------
# 1) Parâmetros
# -----------------------------
r <- 0.05                      # taxa livre de risco
R <- 0.40                      # recovery
Tfinal <- 5                   # anos
dt <- 1/252                   # passo diário

# Parâmetros do processo CIR para lambda(t)
kappa  <- 0.7
theta  <- 0.08
sigma  <- 0.25
lambda0 <- 0.05

# Tempo
tempo <- seq(0, Tfinal, by = dt)
N <- length(tempo)

# Vetores
lambda <- numeric(N)
lambda[1] <- lambda0

# -----------------------------
# 2) Simulação de lambda(t)
#    SDE: dλ = κ(θ-λ)dt + σ sqrt(λ)dW
# -----------------------------
for(i in 2:N){
  dW <- rnorm(1, mean = 0, sd = sqrt(dt))
  lambda[i] <- lambda[i-1] + 
    kappa * (theta - lambda[i-1]) * dt +
    sigma * sqrt(max(lambda[i-1],0)) * dW
  
  lambda[i] <- max(lambda[i], 0)  # garantir positividade
}

# -----------------------------
# 3) Resolver EDO de preço p(t)
#    dp/dt = r p + λ(t)(R - p)
# -----------------------------
preco <- numeric(N)
preco[N] <- 1  # payoff no vencimento

for(i in (N-1):1){
  dp <- (r * preco[i+1] + lambda[i+1] * (R - preco[i+1])) * dt
  preco[i] <- preco[i+1] - dp
}

# -----------------------------
# 4) Spread de crédito instantâneo
#    s(t) = λ(t) * (1 - R)
# -----------------------------
spread <- lambda * (1 - R)

# -----------------------------
# 5) Resultados com 6 casas decimais
# -----------------------------
cat("============== RESULTADOS ==============\n")
cat(sprintf("Preço inicial do bond:        %.6f\n", preco[1]))
cat(sprintf("Lambda(0):                    %.6f\n", lambda[1]))
cat(sprintf("Spread(0):                    %.6f\n", spread[1]))
cat(sprintf("Preço no vencimento T=%.1f:   %.6f\n", Tfinal, preco[N]))
cat("========================================\n")

# -----------------------------
# 6) Gráficos
# -----------------------------
par(mfrow = c(3,1), mar=c(4,4,2,1))

plot(tempo, lambda, type="l", col="blue", lwd=2,
     main="Intensidade de Default λ(t) — CIR",
     xlab="Tempo (anos)", ylab="λ(t)")

plot(tempo, preco, type="l", col="darkgreen", lwd=2,
     main="Preço do Título p(t) — Duffie–Singleton",
     xlab="Tempo (anos)", ylab="Preço")

plot(tempo, spread, type="l", col="red", lwd=2,
     main="Spread de Crédito s(t) = λ(t)(1 - R)",
     xlab="Tempo (anos)", ylab="Spread")

par(mfrow = c(1,1))
