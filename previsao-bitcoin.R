# ============================================================
# Previsão de Preço do Bitcoin (BTC-USD) — ARIMA + Regressão Linear
# Autor: Luiz Tiago Wilcke (LT)
# ============================================================

pkgs <- c("quantmod", "forecast", "TTR", "ggplot2", "dplyr", "lubridate")
instalar <- setdiff(pkgs, rownames(installed.packages()))
if(length(instalar)) install.packages(instalar)

library(quantmod)
library(forecast)
library(TTR)
library(ggplot2)
library(dplyr)
library(lubridate)

# ---------- Coleta de dados ----------
simbolo <- "BTC-USD"
data_inicial <- "2018-01-01"
getSymbols(simbolo, src = "yahoo", from = data_inicial, auto.assign = TRUE)
precos <- Cl(get(simbolo))

dados <- data.frame(
  data = as.Date(index(precos)),
  preco = as.numeric(precos)
)

# ---------- Criação de variáveis ----------
dados <- dados %>%
  mutate(
    retorno = c(NA, diff(log(preco))),
    mm7 = SMA(preco, n = 7),
    mm21 = SMA(preco, n = 21),
    rsi14 = RSI(preco, n = 14)
  )

# Calcula MACD fora do mutate()
macd_vals <- MACD(dados$preco, nFast = 12, nSlow = 26, nSig = 9)
dados$macd  <- macd_vals[,1]
dados$sinal <- macd_vals[,2]

# Remove valores iniciais com NA
dados <- na.omit(dados)

# ---------- Modelo ARIMA nos retornos ----------
modelo_arima <- auto.arima(dados$retorno)
prev_retornos <- forecast(modelo_arima, h = 14)$mean

# Reconstruir preços previstos
ultimo_preco <- tail(dados$preco, 1)
precos_previstos_arima <- ultimo_preco * exp(cumsum(prev_retornos))

# ---------- Modelo de regressão linear ----------
dados <- dados %>%
  mutate(preco_futuro = lead(preco, 1)) %>%
  na.omit()

modelo_lm <- lm(preco_futuro ~ mm7 + mm21 + rsi14 + macd + sinal, data = dados)

# Previsão para o próximo dia
novo <- tail(dados, 1)
previsao_lm <- predict(modelo_lm, newdata = novo)

# ---------- Resultados ----------
cat("=== Modelo ARIMA (Retornos) ===\n")
print(summary(modelo_arima))

cat("\n=== Modelo Linear com Indicadores ===\n")
print(summary(modelo_lm))

cat("\nPreço atual:", round(ultimo_preco,2), "USD")
cat("\nPrevisão ARIMA para 14 dias à frente (último valor):", round(tail(precos_previstos_arima,1),2), "USD")
cat("\nPrevisão Linear (1 dia à frente):", round(previsao_lm,2), "USD\n")

# ---------- Gráfico ----------
df_prev <- data.frame(
  data = seq(max(dados$data)+1, by="day", length.out=14),
  previsto = precos_previstos_arima
)

ggplot() +
  geom_line(data=dados, aes(x=data, y=preco), color="steelblue") +
  geom_line(data=df_prev, aes(x=data, y=previsto), color="red", linetype="dashed") +
  labs(
    title="Previsão de Preço do Bitcoin (ARIMA + Regressão)",
    x="Data", y="Preço (USD)"
  ) +
  theme_minimal()
