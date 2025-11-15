## ============================================================
## Regressão de Poisson — Chamados de Emergência
## Autor: Luiz Tiago Wilcke (LT) — Exemplo em R
## ============================================================

set.seed(123)

## -----------------------------
## 1) Simulação de um banco de dados complexo
## -----------------------------

# Número de dias e bairros
n_dias    <- 365        # 1 ano
n_bairros <- 20
n         <- n_dias * n_bairros

# Criar grade dia x bairro
dia       <- rep(1:n_dias, each = n_bairros)
bairro_id <- rep(1:n_bairros, times = n_dias)

# Dia da semana (1 = segunda, ..., 7 = domingo)
dia_semana_num <- ((dia - 1) %% 7) + 1
rotulos_semana <- c("seg", "ter", "qua", "qui", "sex", "sab", "dom")
dia_semana     <- factor(rotulos_semana[dia_semana_num],
                         levels = rotulos_semana)

# Fim de semana (sab/dom)
fim_semana <- ifelse(dia_semana %in% c("sab", "dom"), 1, 0)

# Feriados (simplificado: alguns dias ao ano)
feriados_dias <- c(1, 90, 120, 150, 200, 250, 300, 359) # exemplo
feriado <- ifelse(dia %in% feriados_dias, 1, 0)

# População exposta por bairro (fixa no ano, mas varia entre bairros)
pop_bairro      <- round(runif(n_bairros, min = 5000, max = 50000))
pop_exposta     <- pop_bairro[bairro_id]

# Chuva (mm): mais chuva no verão (dias 300–365 e 1–90)
chuva_base <- ifelse(dia >= 300 | dia <= 90,
                     runif(n, 5, 40),  # estação chuvosa
                     runif(n, 0, 10))  # estação seca
# Pequena variabilidade aleatória
chuva_mm <- pmax(0, chuva_base + rnorm(n, 0, 3))

# Temperatura: mais quente no verão, mais fria no inverno
temp_base <- ifelse(dia >= 300 | dia <= 90,
                    runif(n, 25, 34),  # verão
                    runif(n, 14, 26))  # demais estações
temperatura <- temp_base + rnorm(n, 0, 1.5)

# -----------------------------
# 2) Gerar a taxa verdadeira (lambda) com estrutura complexa
# -----------------------------
# Modelo "verdadeiro" usado para simular os dados

beta0 <- -8.0           # intercepto
beta_chuva <- 0.03      # efeito da chuva (aumenta chamados)
beta_temp  <- 0.04      # efeito da temperatura
beta_fer   <- 0.50      # feriado aumenta chamados
beta_fds   <- 0.30      # fim de semana aumenta chamados
beta_int   <- 0.015     # interação chuva * feriado

# Efeito aleatório de bairro (heterogeneidade entre bairros)
efeito_bairro <- rnorm(n_bairros, mean = 0, sd = 0.4)
efeito_bairro_i <- efeito_bairro[bairro_id]

# Prever log(lambda)
log_lambda <- beta0 +
  beta_chuva * chuva_mm +
  beta_temp  * temperatura +
  beta_fer   * feriado +
  beta_fds   * fim_semana +
  beta_int   * (chuva_mm * feriado) +
  efeito_bairro_i +
  log(pop_exposta)    # offset "verdadeiro"

lambda <- exp(log_lambda)

# Gerar contagens de chamados (Poisson)
chamados <- rpois(n, lambda)

# Banco de dados final
dados <- data.frame(
  dia          = dia,
  bairro_id    = factor(bairro_id),
  dia_semana   = dia_semana,
  fim_semana   = fim_semana,
  feriado      = feriado,
  chuva_mm     = chuva_mm,
  temperatura  = temperatura,
  pop_exposta  = pop_exposta,
  chamados     = chamados
)

head(dados)
summary(dados$chamados)

## -----------------------------
## 3) Ajuste do modelo de Regressão de Poisson
## -----------------------------

# Vamos ajustar um modelo relativamente complexo:
# chamados ~ chuva_mm * feriado + temperatura + dia_semana + offset(log(pop_exposta))

modelo_poisson <- glm(
  chamados ~ chuva_mm * feriado + temperatura + dia_semana +
    offset(log(pop_exposta)),
  family = poisson(link = "log"),
  data   = dados
)

summary(modelo_poisson)

## -----------------------------
## 4) Diagnóstico de superdispersão
## -----------------------------

# Dispersão (phi) ~ 1 indica que o Poisson está ok
res_pearson <- residuals(modelo_poisson, type = "pearson")
phi <- sum(res_pearson^2) / modelo_poisson$df.residual
phi

cat("Fator de dispersão (phi) =", round(phi, 3), "\n")

# Se phi >> 1, temos superdispersão; poderíamos usar quasi-Poisson ou binomial negativa
# (mas aqui só verificamos)

## -----------------------------
## 5) Análise de deviance e significância global
## -----------------------------

anova(modelo_poisson, test = "Chisq")

## -----------------------------
## 6) Interpretação de alguns coeficientes (em termos de razão de taxas)
## -----------------------------

coeficientes <- coef(modelo_poisson)
exp_coef     <- exp(coeficientes)

round(cbind(Estimativa = coeficientes, Razao_taxa = exp_coef), 4)

## Exemplo de interpretação:
## - exp(beta_chuva) indica a multiplicação esperada na taxa de chamados
##   para +1 mm de chuva, mantendo outras variáveis constantes.
##
## - exp(beta_feriado) indica o fator multiplicativo na taxa em feriados,
##   comparado a dias não feriados (condicional às demais covariáveis).


## -----------------------------
## 7) Predição em cenários específicos
## -----------------------------

# Construir dois cenários para o MESMO bairro e população:
# Cenário A: dia de semana, sem chuva, não feriado
# Cenário B: feriado chuvoso de fim de semana

bairro_ref <- 1
pop_ref    <- 20000

novo <- data.frame(
  dia          = c(10, 20),
  bairro_id    = factor(c(bairro_ref, bairro_ref),
                        levels = levels(dados$bairro_id)),
  dia_semana   = factor(c("qua", "sab"), levels = levels(dados$dia_semana)),
  fim_semana   = c(0, 1),
  feriado      = c(0, 1),
  chuva_mm     = c(0, 35),
  temperatura  = c(22, 30),
  pop_exposta  = c(pop_ref, pop_ref)
)

pred_log <- predict(modelo_poisson, newdata = novo, type = "link")
pred_mu  <- exp(pred_log)

cbind(novo, lambda_esperada = round(pred_mu, 2))

## -----------------------------
## 8) Gráfico: efeito da chuva em dia comum vs feriado
## -----------------------------

chuva_seq <- seq(0, 50, by = 1)

novo_grid <- data.frame(
  dia          = 1,
  bairro_id    = factor(bairro_ref, levels = levels(dados$bairro_id)),
  dia_semana   = factor("qua", levels = levels(dados$dia_semana)),
  fim_semana   = 0,
  feriado      = rep(c(0, 1), each = length(chuva_seq)),
  chuva_mm     = rep(chuva_seq, times = 2),
  temperatura  = 25,
  pop_exposta  = pop_ref
)

lambda_grid <- exp(predict(modelo_poisson, newdata = novo_grid, type = "link"))

# Separar curvas: sem feriado (primeira parte), com feriado (segunda parte)
lambda_sem_feriado <- lambda_grid[novo_grid$feriado == 0]
lambda_com_feriado <- lambda_grid[novo_grid$feriado == 1]

# Gráfico base R
plot(chuva_seq, lambda_sem_feriado, type = "l",
     xlab = "Chuva (mm)",
     ylab = "Chamados esperados",
     main = "Efeito da chuva na taxa de chamados\nDia de semana, bairro fixo",
     lwd = 2)
lines(chuva_seq, lambda_com_feriado, lwd = 2, lty = 2)
legend("topleft",
       legend = c("Sem feriado", "Feriado"),
       lwd = 2, lty = c(1, 2), bty = "n")
