# ============================================================
# PI via Fórmula de Ramanujan (Série 1/pi)
# Autor: Luiz Tiago Wilcke (LT)
# ============================================================
# Fórmula clássica de Ramanujan:
#
#   1/π = (2*sqrt(2) / 9801) * Σ_{k=0}^{∞} [ (4k)! * (1103 + 26390k) ]
#                                       / [ (k!)^4 * 396^(4k) ]
#
# Nesta implementação:
#   - Usamos o módulo 'decimal' para ter alta precisão (1000+ dígitos)
#   - Somamos a série até que o termo seja menor que uma tolerância-alvo
#   - Utilizamos variáveis em português e uma estrutura mais sofisticada
# ============================================================

from dataclasses import dataclass
from decimal import Decimal, getcontext

@dataclass
class ConfiguracaoRamanujan:
    casas_decimais: int = 1000      # número de casas decimais desejadas para π
    margem_precisao: int = 20       # "folga" de dígitos extras para evitar erro numérico
    termos_maximos: int = 200       # limite de termos da série (Ramanujan converge muito rápido)
    mostrar_progresso: bool = True  # se True, mostra alguns logs simples

def calcular_pi_ramanujan(cfg: ConfiguracaoRamanujan) -> Decimal:
    """
    Calcula π usando a fórmula de Ramanujan para 1/π,
    com precisão configurável via 'ConfiguracaoRamanujan'.
    """
    # --------------------------------------------------------
    # 1) Ajustar precisão global do Decimal
    # --------------------------------------------------------
    precisao_interna = cfg.casas_decimais + cfg.margem_precisao
    getcontext().prec = precisao_interna

    # --------------------------------------------------------
    # 2) Constantes da fórmula
    # --------------------------------------------------------
    dois = Decimal(2)
    raiz2 = dois.sqrt()
    multiplicador = (dois * raiz2) / Decimal(9801)  # 2*sqrt(2)/9801

    # Série: Σ (4k)! (1103 + 26390k) / (k!^4 * 396^(4k))
    soma_serie = Decimal(0)

    # Pré-cálculo da base 396^4 como inteiro
    base_potencia = 396 ** 4

    # Fatoriais e potências controlados iterativamente
    fatorial_k = 1          # k!
    fatorial_4k = 1         # (4k)!
    potencia_396_4k = 1     # 396^(4k)

    # Tolerância alvo: algo como 10^(-casas_decimais - 5)
    tolerancia = Decimal(10) ** Decimal(-(cfg.casas_decimais + 5))

    for k in range(cfg.termos_maximos):
        if k == 0:
            # k = 0 => (4k)! = 0! = 1, (k!)^4 = 1, 396^(4k) = 1
            fatorial_4k = 1
            fatorial_k = 1
            potencia_396_4k = 1
        else:
            # Atualiza (4k)! a partir de (4(k-1))!
            # (4k)! = (4k)*(4k-1)*(4k-2)*(4k-3)*(4(k-1))!
            n1 = 4 * k
            fatorial_4k *= n1 * (n1 - 1) * (n1 - 2) * (n1 - 3)

            # Atualiza k! a partir de (k-1)! => k! = k * (k-1)!
            fatorial_k *= k

            # Atualiza 396^(4k) a partir de 396^(4(k-1))
            potencia_396_4k *= base_potencia

        # (k!)^4
        k_fatorial_quarta = fatorial_k ** 4

        # Termo inteiro da fração
        termo_numerador_int = fatorial_4k * (1103 + 26390 * k)
        termo_denominador_int = k_fatorial_quarta * potencia_396_4k

        # Converte para Decimal na divisão
        termo = Decimal(termo_numerador_int) / Decimal(termo_denominador_int)

        soma_serie += termo

        # Critério de parada: termo muito pequeno
        if cfg.mostrar_progresso and k % 10 == 0:
            print(f"k = {k:3d}, termo ≈ {termo:.3E}")

        if abs(termo) < tolerancia:
            if cfg.mostrar_progresso:
                print(f"Parando em k = {k}, termo < tolerância ({tolerancia})")
            break

    # --------------------------------------------------------
    # 3) Fechar fórmula: 1/π = multiplicador * soma_serie
    # --------------------------------------------------------
    inverso_pi = multiplicador * soma_serie
    pi_aproximado = Decimal(1) / inverso_pi

    return pi_aproximado

def formatar_pi(pi_valor: Decimal, casas_decimais: int) -> str:
    """
    Formata π com um número específico de casas decimais.
    Ex: casas_decimais = 1000 -> 1 dígito antes da vírgula + 1000 depois.
    """
    formato = f".{casas_decimais}f"
    return format(pi_valor, formato)

if __name__ == "__main__":
    # Configuração-alvo: 1000 casas decimais
    cfg = ConfiguracaoRamanujan(
        casas_decimais=1000,
        margem_precisao=20,
        termos_maximos=200,
        mostrar_progresso=True  # mude para False se não quiser logs
    )

    print("Calculando π via fórmula de Ramanujan com alta precisão...")
    pi_decimal = calcular_pi_ramanujan(cfg)
    pi_formatado = formatar_pi(pi_decimal, cfg.casas_decimais)

    print("\nπ aproximado (1000 casas decimais):\n")
    print(pi_formatado)
