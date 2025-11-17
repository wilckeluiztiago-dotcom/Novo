###############################################################
# MODELO  DE ANÁLISE DE CRÉDITO
# EDE com saltos de Lévy + Probabilidade de Default
# Autor: Luiz Tiago Wilcke (LT)
###############################################################

# ------------------------------------------------------------
# 1) Função geradora de distribuição de Lévy (positiva)
#    X ~ Levy(mu, c) via transformação de variável normal
#    X = mu + c^2 / Z^2, com Z ~ Normal(0,1)
# ------------------------------------------------------------
rlevy <- function(n, c = 1, mu = 0) {
  z <- rnorm(n)
  x <- mu + (c^2) / (z^2)
  return(x)
}

# ------------------------------------------------------------
# 2) Simulação do processo de qualidade de crédito Q_t
#
# Modelo contínuo (EDE com saltos de Lévy):
#   dQ_t = kappa * (mu_q - Q_t) * dt + sigma * dW_t - dL_t
#
#   - Termo de reversão à média: kappa * (mu_q - Q_t) * dt
#   - Difusão Browniana: sigma * dW_t
#   - Saltos negativos de Lévy: - dL_t
#
# Probabilidade de default (logística):
#   PD_t = 1 / (1 + exp(-(alpha_pd + beta_pd * Q_t)))
#
# Default é absorvente: após default, qualidade e PD congelam.
# ------------------------------------------------------------
simular_modelo_credito <- function(
    n_clientes      = 500,
    horizonte_anos  = 5,
    passos_por_ano  = 252,
    kappa           = 1.5,
    media_qualidade = 0,
    volatilidade    = 0.40,
    lambda_salto    = 0.7,   # intensidade de Poisson (saltos por ano)
    escala_levy     = 0.5,   # parâmetro c da Lévy (tamanho típico de salto)
    deslocamento_levy = 0.0, # parâmetro mu da Lévy
    alpha_pd        = -3.0,
    beta_pd         = -1.1   # negativo: menor Q => maior PD
) {
  # passo de tempo (anos)
  dt <- 1 / passos_por_ano
  n_passos <- as.integer(horizonte_anos * passos_por_ano)
  
  # matrizes: linhas = tempo, colunas = clientes
  qualidade     <- matrix(NA_real_, nrow = n_passos + 1, ncol = n_clientes)
  prob_default  <- matrix(NA_real_, nrow = n_passos + 1, ncol = n_clientes)
  estado_default <- matrix(0L,      nrow = n_passos + 1, ncol = n_clientes)
  
  # Condições iniciais: distribuição de qualidade no tempo 0
  qualidade[1, ] <- rnorm(n_clientes, mean = media_qualidade, sd = 0.7)
  
  # Probabilidade inicial de default
  logito_pd_inicial <- alpha_pd + beta_pd * qualidade[1, ]
  prob_default[1, ] <- 1 / (1 + exp(-logito_pd_inicial))
  
  # Alguns clientes já podem estar em default no instante inicial
  estado_default[1, ] <- rbinom(n_clientes, size = 1, prob = prob_default[1, ])
  
  # Loop temporal
  for (t in 1:n_passos) {
    q_t <- qualidade[t, ]
    
    # Incremento Browniano
    dW <- rnorm(n_clientes, mean = 0, sd = sqrt(dt))
    
    # Número de saltos de Lévy em cada cliente no intervalo dt
    n_saltos <- rpois(n_clientes, lambda = lambda_salto * dt)
    
    # Tamanho agregado de saltos de Lévy (somando todos os saltos em dt)
    tamanho_saltos <- numeric(n_clientes)
    indices_com_salto <- which(n_saltos > 0)
    if (length(indices_com_salto) > 0) {
      for (i in indices_com_salto) {
        tamanho_saltos[i] <- sum(
          rlevy(n_saltos[i], c = escala_levy, mu = deslocamento_levy)
        )
      }
    }
    
    # Incremento da qualidade de crédito
    dQ <- kappa * (media_qualidade - q_t) * dt +
      volatilidade * dW -
      tamanho_saltos
    
    q_novo <- q_t + dQ
    
    # Clientes que já estavam em default têm trajetórias congeladas
    ja_default <- estado_default[t, ] == 1
    q_novo[ja_default] <- q_t[ja_default]
    
    qualidade[t + 1, ] <- q_novo
    
    # Probabilidade de default no novo tempo
    logito_pd <- alpha_pd + beta_pd * q_novo
    pd <- 1 / (1 + exp(-logito_pd))
    
    # Clientes em default ficam com PD = 1
    pd[ja_default] <- 1
    
    prob_default[t + 1, ] <- pd
    
    # Simulação de novos defaults neste intervalo
    # Probabilidade aproximada: PD_t * dt
    novos_default <- (estado_default[t, ] == 0) & (runif(n_clientes) < pd * dt)
    estado_default[t + 1, ] <- as.integer(estado_default[t, ] | novos_default)
  }
  
  lista <- list(
    qualidade      = qualidade,
    prob_default   = prob_default,
    estado_default = estado_default,
    dt             = dt,
    passos_por_ano = passos_por_ano,
    horizonte_anos = horizonte_anos
  )
  return(lista)
}

# ------------------------------------------------------------
# 3) Cálculo de perdas de crédito no portfólio
#
# Exposição e LGD são sorteadas para ilustrar:
#   Exposicao_i ~ Uniforme(1000, 50000)
#   LGD_i       ~ Uniforme(0.3, 0.9)
#
# Perda por cliente no horizonte:
#   Loss_i = Exposicao_i * LGD_i * Indicador_default_final_i
# ------------------------------------------------------------
calcular_perdas_portfolio <- function(resultados) {
  estado_default <- resultados$estado_default
  n_passos   <- nrow(estado_default) - 1
  n_clientes <- ncol(estado_default)
  
  # Exposição ao risco de crédito e LGD
  exposicao <- runif(n_clientes, min = 1000, max = 50000)
  lgd       <- runif(n_clientes, min = 0.3,  max = 0.9)
  
  # Perda por cliente no horizonte final
  perda_cliente_horizonte <- exposicao * lgd * estado_default[n_passos + 1, ]
  perda_total_horizonte   <- sum(perda_cliente_horizonte)
  
  # Perda acumulada ao longo do tempo
  perda_esperada_t <- numeric(n_passos + 1)
  for (t in 1:(n_passos + 1)) {
    perda_esperada_t[t] <- sum(exposicao * lgd * estado_default[t, ])
  }
  
  lista <- list(
    exposicao               = exposicao,
    lgd                     = lgd,
    perda_cliente_horizonte = perda_cliente_horizonte,
    perda_total_horizonte   = perda_total_horizonte,
    perda_esperada_t        = perda_esperada_t
  )
  return(lista)
}

# ------------------------------------------------------------
# 4) Geração de gráficos para análise de crédito
#    - Trajetórias da qualidade de crédito
#    - Probabilidade média de default
#    - Distribuição da qualidade no horizonte
#    - Distribuição da perda por cliente
#    - Evolução da perda acumulada do portfólio
# ------------------------------------------------------------
gerar_graficos_credito <- function(resultados, perdas) {
  qualidade      <- resultados$qualidade
  prob_default   <- resultados$prob_default
  estado_default <- resultados$estado_default
  dt             <- resultados$dt
  passos_por_ano <- resultados$passos_por_ano
  horizonte_anos <- resultados$horizonte_anos
  
  n_passos   <- nrow(qualidade) - 1
  n_clientes <- ncol(qualidade)
  tempo_anos <- (0:n_passos) / passos_por_ano
  
  op <- par(no.readonly = TRUE)
  on.exit(par(op))
  
  par(mfrow = c(2, 2))
  
  # 4.1) Trajetórias da qualidade de crédito (primeiros clientes)
  n_plot <- min(20, n_clientes)
  matplot(
    tempo_anos,
    qualidade[, 1:n_plot],
    type = "l", lty = 1,
    xlab = "Tempo (anos)",
    ylab = "Qualidade de crédito",
    main = "Trajetórias de qualidade (primeiros clientes)"
  )
  
  # 4.2) Probabilidade média de default ao longo do tempo
  pd_media <- rowMeans(prob_default)
  plot(
    tempo_anos,
    pd_media,
    type = "l",
    xlab = "Tempo (anos)",
    ylab = "Probabilidade média de default",
    main = "PD média do portfólio",
    ylim = c(0, 1)
  )
  
  # 4.3) Distribuição da qualidade de crédito no horizonte final
  hist(
    qualidade[n_passos + 1, ],
    breaks = 30,
    main = "Distribuição da qualidade no horizonte",
    xlab = "Qualidade de crédito final",
    col = "lightgray",
    border = "white"
  )
  
  # 4.4) Distribuição das perdas por cliente
  hist(
    perdas$perda_cliente_horizonte,
    breaks = 30,
    main = "Distribuição de perda por cliente",
    xlab = "Perda no horizonte",
    col = "lightgray",
    border = "white"
  )
  
  # Nova janela gráfica para perda acumulada do portfólio (se disponível)
  # Se estiver em ambiente sem suporte a múltiplas janelas, comente dev.new()
  dev.new()
  plot(
    tempo_anos,
    perdas$perda_esperada_t,
    type = "l",
    xlab = "Tempo (anos)",
    ylab = "Perda acumulada esperada",
    main = "Perda acumulada do portfólio"
  )
}

# ------------------------------------------------------------
# 5) Execução do modelo completo
# ------------------------------------------------------------
set.seed(123)

resultados <- simular_modelo_credito(
  n_clientes      = 500,
  horizonte_anos  = 5,
  passos_por_ano  = 252,
  kappa           = 1.5,
  media_qualidade = 0,
  volatilidade    = 0.40,
  lambda_salto    = 0.7,
  escala_levy     = 0.5,
  deslocamento_levy = 0.0,
  alpha_pd        = -3.0,
  beta_pd         = -1.1
)

perdas <- calcular_perdas_portfolio(resultados)

# Resumo numérico
n_clientes <- ncol(resultados$qualidade)
taxa_default_final <- mean(resultados$estado_default[nrow(resultados$estado_default), ])

cat("============================================================\n")
cat("RESUMO DO PORTFÓLIO DE CRÉDITO\n")
cat("Número de clientes:", n_clientes, "\n")
cat("Taxa de default no horizonte:", round(taxa_default_final, 4), "\n")
cat("Perda total do portfólio:", round(perdas$perda_total_horizonte, 2), "\n")
cat("============================================================\n")

# Gráficos
gerar_graficos_credito(resultados, perdas)
