# ============================================================
# Raiz cúbica de 8.91 com 50 dígitos — Newton–Raphson (PURO)
# Autor: Luiz Tiago Wilcke (LT)
# ============================================================
# Método:
#   f(x) = x^3 - 8.91  →  x_{k+1} = (2*x_k + 8.91/x_k^2)/3
#   Tudo em PONTO-FIXO com inteiros. Se S = 10^p e x_int = round(x*S),
#   então o termo (8.91/x^2) na mesma escala de x fica:
#        T_int = A_escalado * S^2 / x_int^2,
#   onde A_escalado = 8.91 * S (exato pois p≥2).
#   Assim x_{k+1,int} = (2*x_int + T_int)/3  (arredondado).
# ============================================================

# ---------------------------
# Utilidades de ponto fixo
# ---------------------------
def pot10(n: int) -> int:
    """Retorna 10**n (n >= 0) como inteiro (rápido e sem float)."""
    r = 1
    for _ in range(n):
        r *= 10
    return r

def formatar_decimal(valor_escalado: int, p: int, casas: int) -> str:
    """
    Converte 'valor_escalado' (valor_real * 10^p) em string com 'casas' casas,
    com arredondamento inteiro correto.
    """
    neg = valor_escalado < 0
    v = -valor_escalado if neg else valor_escalado

    if p > casas:
        fator = pot10(p - casas)
        v = (v + fator // 2) // fator
        p_final = casas
    else:
        v = v * pot10(casas - p)
        p_final = casas

    base = pot10(p_final)
    inteiro = v // base
    frac    = v %  base
    s = f"{inteiro}.{str(frac).rjust(p_final, '0')}"
    return "-" + s if neg else s

def cubo_escalado(x_int: int, p: int) -> int:
    """
    Devolve (x_real^3)*S na escala S (i.e., x_int^3 / S^2).
    """
    S = pot10(p)
    return ((x_int * x_int) // S * x_int) // S

# -----------------------------------------
# Newton–Raphson (passo-a-passo “à mão”)
# -----------------------------------------
def raiz_cubica_891_newton(passos_max=30, p_interno=120, casas_saida=50,
                           imprimir=True):
    """
    Calcula (8.91)^(1/3) com aritmética inteira de alta precisão.
      - p_interno: dígitos internos (S = 10^p_interno). Use bem > casas_saida.
      - casas_saida: casas no resultado final.
      - passos_max: número máximo de iterações de Newton.
    """
    assert p_interno >= 2, "Use p_interno >= 2 para representar 8.91 exatamente."
    S = pot10(p_interno)

    # 8.91 na escala S (exato para p>=2)
    A_escalado = (891 * S) // 100

    # Chute inicial: 2 (pois 2^3 = 8)
    x = 2 * S

    if imprimir:
        print("=== Newton–Raphson em ponto fixo (inteiros) ===")
        print(f"Alvo: raiz cúbica de 8.91; p_interno={p_interno} (S=10^p).")
        print(f"Chute x0 = {formatar_decimal(x, p_interno, 20)}\n")
        cab = "{:>3s} | {:>60s} | {:>28s} | {:>28s}"
        print(cab.format("k", "x_k (decimal)", "resíduo |x^3-8.91|", "variação |Δx|"))
        print("-"*3 + "-+-" + "-"*60 + "-+-" + "-"*28 + "-+-" + "-"*28)

    for k in range(1, passos_max + 1):
        # termo = A_escalado * S^2 / x^2   (mantém a escala de x)
        den   = x * x
        termo = (A_escalado * S * S + den // 2) // den  # arredondamento inteiro

        # x_{k+1} = (2*x + termo)/3 (ainda na escala S)
        x_novo = (2 * x + termo + 1) // 3  # +1 para arredondar ao mais próximo

        # Métricas (na escala S)
        residuo = cubo_escalado(x_novo, p_interno) - A_escalado
        if residuo < 0: residuo = -residuo
        variacao = x_novo - x
        if variacao < 0: variacao = -variacao

        if imprimir:
            print(f"{k:3d} | {formatar_decimal(x_novo, p_interno, 60):>60s} | "
                  f"{formatar_decimal(residuo, p_interno, 28):>28s} | "
                  f"{formatar_decimal(variacao, p_interno, 28):>28s}")

        # Parada prática: quando Δx = 0 em escala interna (estabilizou)
        if variacao == 0:
            x = x_novo
            break

        x = x_novo

    # Resultado final com 50 casas
    resultado = formatar_decimal(x, p_interno, casas_saida)

    if imprimir:
        print("\n=== Resultado final ===")
        print(f"raiz_cubica(8.91) ≈ {resultado}  (com {casas_saida} casas decimais)")
        cubo_final = cubo_escalado(x, p_interno)
        erro_final = cubo_final - A_escalado
        if erro_final < 0: erro_final = -erro_final
        print("Checagem: |x^3 - 8.91| ≈", formatar_decimal(erro_final, p_interno, 60))

    return resultado

# -----------------
# Execução direta
# -----------------
if __name__ == "__main__":
    # p_interno = 120 dá folga generosa para imprimir 50 casas estáveis
    raiz_cubica_891_newton(passos_max=30, p_interno=120, casas_saida=50, imprimir=True)
