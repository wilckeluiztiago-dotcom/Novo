# ============================================================
# Índices da B3 + Black–Scholes 
# Autor: Luiz Tiago Wilcke (LT)
# ============================================================

# ------------------------------------------------------------
# 1) Dados de exemplo dos principais índices (substitua por dados reais)
# ------------------------------------------------------------

# Datas simuladas (10 dias úteis)
datas <- as.Date("2025-01-02") + 0:9

precos_IBOV    <- c(133000, 134500, 132800, 135200, 136000, 137500, 138200, 137900, 139000, 140500)
precos_IBRX100 <- c(71000, 71500, 70800, 72000, 72500, 73000, 73250, 73100, 73500, 74000)
precos_IFIX    <- c(3200, 3210, 3195, 3220, 3230, 3240, 3255, 3260, 3270, 3285)

# ------------------------------------------------------------
# 2) Função para calcular retornos logarítmicos
# ------------------------------------------------------------
calcular_retorno_log <- function(precos) {
  n <- length(precos)
  if (n < 2) {
    stop("É necessário pelo menos 2 preços para calcular retornos.")
  }
  retornos <- numeric(n - 1)
  i <- 2
  while (i <= n) {
    retornos[i - 1] <- log(precos[i] / precos[i - 1])
    i <- i + 1
  }
  return(retornos)
}

# ------------------------------------------------------------
# 3) Volatilidade anualizada (base: 252 dias úteis)
# ------------------------------------------------------------
calcular_vol_anual <- function(precos) {
  retornos <- calcular_retorno_log(precos)
  # Desvio padrão amostral dos retornos
  desvio <- sd(retornos)
  vol_anual <- desvio * sqrt(252)
  return(vol_anual)
}

vol_IBOV    <- calcular_vol_anual(precos_IBOV)
vol_IBRX100 <- calcular_vol_anual(precos_IBRX100)
vol_IFIX    <- calcular_vol_anual(precos_IFIX)

# Também guardamos os retornos para plotar
retornos_IBOV    <- calcular_retorno_log(precos_IBOV)
retornos_IBRX100 <- calcular_retorno_log(precos_IBRX100)
retornos_IFIX    <- calcular_retorno_log(precos_IFIX)

# ------------------------------------------------------------
# 4) Black–Scholes para call europeia 
# ------------------------------------------------------------
black_scholes_call <- function(preco_inicial, preco_exercicio,
                               taxa_juros_anual, volatilidade_anual,
                               tempo_anos) {
  if (preco_inicial <= 0 || volatilidade_anual <= 0 || tempo_anos <= 0) {
    return(0)
  }
  
  raiz_T <- sqrt(tempo_anos)
  d1 <- (log(preco_inicial / preco_exercicio) +
           (taxa_juros_anual + 0.5 * volatilidade_anual^2) * tempo_anos) /
    (volatilidade_anual * raiz_T)
  d2 <- d1 - volatilidade_anual * raiz_T
  
  # pnorm e exp são funções de base R (pacote stats/base)
  valor_call <- preco_inicial * pnorm(d1) -
    preco_exercicio * exp(-taxa_juros_anual * tempo_anos) * pnorm(d2)
  return(valor_call)
}

# ------------------------------------------------------------
# 5) Exemplo: precificar calls sobre IBOV, IBRX100, IFIX
# ------------------------------------------------------------

# Suposições (ajuste conforme sua análise)
taxa_SELIC_anual <- 0.11        # 11% ao ano (exemplo)
tempo_anos <- 0.5               # 6 meses até o vencimento

# Preços atuais (última observação do vetor)
preco_atual_IBOV    <- tail(precos_IBOV, 1)
preco_atual_IBRX100 <- tail(precos_IBRX100, 1)
preco_atual_IFIX    <- tail(precos_IFIX, 1)

# Vamos fixar strikes (exemplo: at-the-money ou levemente OTM)
strike_IBOV    <- preco_atual_IBOV    * 1.02
strike_IBRX100 <- preco_atual_IBRX100 * 1.02
strike_IFIX    <- preco_atual_IFIX    * 1.02

# Cálculo dos preços das calls
preco_call_IBOV <- black_scholes_call(
  preco_inicial      = preco_atual_IBOV,
  preco_exercicio    = strike_IBOV,
  taxa_juros_anual   = taxa_SELIC_anual,
  volatilidade_anual = vol_IBOV,
  tempo_anos         = tempo_anos
)

preco_call_IBRX100 <- black_scholes_call(
  preco_inicial      = preco_atual_IBRX100,
  preco_exercicio    = strike_IBRX100,
  taxa_juros_anual   = taxa_SELIC_anual,
  volatilidade_anual = vol_IBRX100,
  tempo_anos         = tempo_anos
)

preco_call_IFIX <- black_scholes_call(
  preco_inicial      = preco_atual_IFIX,
  preco_exercicio    = strike_IFIX,
  taxa_juros_anual   = taxa_SELIC_anual,
  volatilidade_anual = vol_IFIX,
  tempo_anos         = tempo_anos
)

# ------------------------------------------------------------
# 6) Exibir resultados numéricos
# ------------------------------------------------------------
cat("=============== ÍNDICES B3 + BLACK–SCHOLES (R BASE) ===============\n")
cat("Volatilidade anual IBOV   :", vol_IBOV,    "\n")
cat("Volatilidade anual IBRX100:", vol_IBRX100, "\n")
cat("Volatilidade anual IFIX   :", vol_IFIX,    "\n\n")

cat("Preço atual IBOV   :", preco_atual_IBOV,    " | Strike:", strike_IBOV,
    " | Call (BS):", preco_call_IBOV,    "\n")
cat("Preço atual IBRX100:", preco_atual_IBRX100, " | Strike:", strike_IBRX100,
    " | Call (BS):", preco_call_IBRX100, "\n")
cat("Preço atual IFIX   :", preco_atual_IFIX,    " | Strike:", strike_IFIX,
    " | Call (BS):", preco_call_IFIX,    "\n")
cat("=================================================================\n")

# ------------------------------------------------------------
# 7) GRÁFICOS EM R BASE (sem pacotes externos)
# ------------------------------------------------------------

# Ajustar layout: 2 linhas x 2 colunas
par(mfrow = c(2, 2))

# ---- Gráfico 1: Preços dos índices ao longo do tempo ----
plot(datas, precos_IBOV, type = "l", lwd = 2,
     main = "Índices da B3 - Preços",
     xlab = "Data", ylab = "Nível do índice")
lines(datas, precos_IBRX100, lwd = 2, lty = 2)
lines(datas, precos_IFIX, lwd = 2, lty = 3)
legend("topleft",
       legend = c("IBOV", "IBRX100", "IFIX"),
       lty    = c(1, 2, 3),
       lwd    = 2,
       bty    = "n")

# ---- Gráfico 2: Retornos logarítmicos ----
datas_ret <- datas[-1]  # retornos começam na segunda data
plot(datas_ret, retornos_IBOV, type = "l", lwd = 2,
     main = "Retornos Logarítmicos Diários",
     xlab = "Data", ylab = "Retorno logarítmico")
lines(datas_ret, retornos_IBRX100, lwd = 2, lty = 2)
lines(datas_ret, retornos_IFIX, lwd = 2, lty = 3)
abline(h = 0, lty = 3)
legend("topleft",
       legend = c("IBOV", "IBRX100", "IFIX"),
       lty    = c(1, 2, 3),
       lwd    = 2,
       bty    = "n")

# ---- Gráfico 3: Volatilidades anualizadas ----
volat <- c(vol_IBOV, vol_IBRX100, vol_IFIX)
nomes_vol <- c("IBOV", "IBRX100", "IFIX")
barplot(volat,
        names.arg = nomes_vol,
        main = "Volatilidade Anualizada (252 dias)",
        ylab = "Volatilidade",
        ylim = c(0, max(volat) * 1.2))

# ---- Gráfico 4: Preços das calls de Black–Scholes ----
precos_calls <- c(preco_call_IBOV, preco_call_IBRX100, preco_call_IFIX)
nomes_calls  <- c("Call IBOV", "Call IBRX100", "Call IFIX")
barplot(precos_calls,
        names.arg = nomes_calls,
        main = "Preço da Call (Black–Scholes)",
        ylab = "Preço teórico da call")