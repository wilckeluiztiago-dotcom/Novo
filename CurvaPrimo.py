# ============================================================
# Criptografia de Curvas Elípticas 
# Autor: Luiz Tiago Wilcke 
# ============================================================
# Recursos:
#  - Curva elíptica sobre corpo primo F_p
#  - Representação de pontos, ponto no infinito
#  - Soma de pontos e multiplicação escalar (double-and-add)
#  - Geração de chave privada/pública
#  - ECDH (Elliptic Curve Diffie–Hellman)
#  - ECDSA (assinatura digital e verificação)
# ============================================================

import secrets
import hashlib
from dataclasses import dataclass
from typing import Optional, Tuple

# ------------------------------------------------------------
# 1. Estruturas básicas: Curva e Ponto
# ------------------------------------------------------------

@dataclass
class CurvaEliptica:
    """
    Representa uma curva elíptica do tipo:
        y^2 ≡ x^3 + a*x + b  (mod p)
    com um ponto gerador G de ordem n.
    """
    nome: str
    p: int   # primo do corpo finito
    a: int
    b: int
    gx: int  # coordenada x do ponto gerador G
    gy: int  # coordenada y do ponto gerador G
    n: int   # ordem do ponto gerador

    def __post_init__(self):
        # Verificação simples de não singularidade: 4a^3 + 27b^2 ≠ 0 (mod p)
        discriminante = (4 * pow(self.a, 3, self.p) + 27 * pow(self.b, 2, self.p)) % self.p
        if discriminante == 0:
            raise ValueError("Curva elíptica singular (discriminante = 0).")
        # Ponto gerador como propriedade
        self.G = PontoEliptico(self, self.gx, self.gy)


class PontoEliptico:
    """
    Representa um ponto em uma curva elíptica.
    O ponto no infinito é representado por x=y=None.
    """
    def __init__(self, curva: CurvaEliptica, x: Optional[int], y: Optional[int]):
        self.curva = curva
        self.x = x
        self.y = y

    def eh_infinito(self) -> bool:
        return self.x is None and self.y is None

    def __repr__(self):
        if self.eh_infinito():
            return "PontoEliptico(Infinito)"
        return f"PontoEliptico(x={self.x}, y={self.y})"

    # ---------------- Soma de pontos na curva ----------------

    def __neg__(self):
        """Retorna o oposto de um ponto: (x, -y mod p)."""
        if self.eh_infinito():
            return self
        return PontoEliptico(self.curva, self.x, (-self.y) % self.curva.p)

    def __add__(self, outro: "PontoEliptico") -> "PontoEliptico":
        """Soma de dois pontos na curva elíptica."""
        if self.curva is not outro.curva:
            raise ValueError("Pontos pertencem a curvas diferentes.")

        p = self.curva.p

        # Casos com ponto no infinito
        if self.eh_infinito():
            return outro
        if outro.eh_infinito():
            return self

        # Caso P + (-P) = O
        if self.x == outro.x and (self.y != outro.y or self.y == 0):
            return PontoEliptico(self.curva, None, None)  # infinito

        # Cálculo da inclinação (lambda)
        if self.x == outro.x and self.y == outro.y:
            # Ponto duplicado: fórmula de tangente
            numerador = (3 * self.x * self.x + self.curva.a) % p
            denominador = inverso_modular(2 * self.y % p, p)
        else:
            # Pontos distintos
            numerador = (outro.y - self.y) % p
            denominador = inverso_modular((outro.x - self.x) % p, p)

        lam = (numerador * denominador) % p

        # Fórmulas de soma em curva elíptica
        x3 = (lam * lam - self.x - outro.x) % p
        y3 = (lam * (self.x - x3) - self.y) % p
        return PontoEliptico(self.curva, x3, y3)

    # ---------------- Multiplicação escalar (k * P) ----------------

    def __rmul__(self, escalar: int) -> "PontoEliptico":
        """
        Multiplicação escalar via algoritmo double-and-add.
        Retorna escalar * P.
        """
        if escalar < 0:
            return (-escalar) * (-self)

        resultado = PontoEliptico(self.curva, None, None)  # infinito
        adicionando = self

        k = escalar
        while k > 0:
            if k & 1:
                resultado = resultado + adicionando
            adicionando = adicionando + adicionando
            k >>= 1

        return resultado


# ------------------------------------------------------------
# 2. Aritmética modular: inverso (para divisão modular)
# ------------------------------------------------------------

def inverso_modular(x: int, p: int) -> int:
    """
    Calcula o inverso modular de x (mod p) usando o algoritmo
    estendido de Euclides. Garante (x * inv) % p == 1.
    """
    if x == 0:
        raise ZeroDivisionError("Não existe inverso modular de zero.")
    # Algoritmo estendido de Euclides
    a, b = p, x % p
    u0, u1 = 0, 1
    while b != 0:
        q = a // b
        a, b = b, a - q * b
        u0, u1 = u1, u0 - q * u1
    if a != 1:
        raise ValueError("x não é inversível módulo p.")
    return u0 % p


# ------------------------------------------------------------
# 3. Definição de uma curva real (ex.: secp256k1)
# ------------------------------------------------------------
# Parâmetros da curva secp256k1 (usada no Bitcoin) – apenas PARA ESTUDO
# y^2 = x^3 + 0*x + 7 (mod p)
# ------------------------------------------------------------

curva_secp256k1 = CurvaEliptica(
    nome="secp256k1_didatica",
    p=0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F,
    a=0,
    b=7,
    gx=0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
    gy=0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8,
    n=0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
)


# ------------------------------------------------------------
# 4. Geração de chaves ECC
# ------------------------------------------------------------

def gerar_chave_privada(curva: CurvaEliptica) -> int:
    """
    Gera uma chave privada aleatória no intervalo [1, n-1].
    """
    while True:
        candidato = secrets.randbelow(curva.n)
        if 1 <= candidato < curva.n:
            return candidato


def gerar_par_de_chaves(curva: CurvaEliptica) -> Tuple[int, PontoEliptico]:
    """
    Retorna (chave_privada, chave_publica) para a curva dada.
    chave_publica = chave_privada * G
    """
    chave_privada = gerar_chave_privada(curva)
    chave_publica = chave_privada * curva.G
    return chave_privada, chave_publica


# ------------------------------------------------------------
# 5. ECDH – Segredo compartilhado
# ------------------------------------------------------------

def derivar_segredo_compartilhado(
    curva: CurvaEliptica,
    chave_privada_local: int,
    chave_publica_remota: PontoEliptico
) -> bytes:
    """
    Implementa o ECDH (Elliptic Curve Diffie–Hellman):
        S = d_local * Q_remoto
    Depois condensa S.x em um hash (SHA-256) para usar como chave simétrica.
    """
    if chave_publica_remota.eh_infinito():
        raise ValueError("Chave pública remota inválida (ponto no infinito).")

    ponto_segredo = chave_privada_local * chave_publica_remota
    if ponto_segredo.eh_infinito():
        raise ValueError("Falha ao derivar segredo (ponto no infinito).")

    x_bytes = ponto_segredo.x.to_bytes(32, "big")
    # Deriva chave simétrica usando SHA-256
    chave_simetrica = hashlib.sha256(x_bytes).digest()
    return chave_simetrica


# ------------------------------------------------------------
# 6. ECDSA – Assinatura digital
# ------------------------------------------------------------

def hash_mensagem(mensagem: bytes) -> int:
    """
    Calcula z = int(hash_sha256(mensagem)).
    """
    digest = hashlib.sha256(mensagem).digest()
    return int.from_bytes(digest, "big")


def assinar_mensagem(
    curva: CurvaEliptica,
    chave_privada: int,
    mensagem: bytes
) -> Tuple[int, int]:
    """
    Implementa ECDSA:
      - z = hash(mensagem)
      - escolher k aleatório em [1, n-1]
      - P = k*G = (x1, y1)
      - r = x1 mod n (se r==0, repetir)
      - s = (k^-1 * (z + r*d)) mod n (se s==0, repetir)
    Retorna assinatura (r, s).
    """
    n = curva.n
    z = hash_mensagem(mensagem) % n

    while True:
        k = gerar_chave_privada(curva)  # k aleatório
        P = k * curva.G
        r = P.x % n
        if r == 0:
            continue
        k_inv = inverso_modular(k, n)
        s = (k_inv * (z + r * chave_privada)) % n
        if s == 0:
            continue
        return r, s


def verificar_assinatura(
    curva: CurvaEliptica,
    chave_publica: PontoEliptico,
    mensagem: bytes,
    assinatura: Tuple[int, int]
) -> bool:
    """
    Verifica assinatura ECDSA:
      - z = hash(mensagem)
      - (r, s) assinatura
      - w = s^-1 mod n
      - u1 = z*w mod n
      - u2 = r*w mod n
      - P = u1*G + u2*Q
      - assinatura válida se P != O e (P.x mod n) == r
    """
    r, s = assinatura
    n = curva.n

    if not (1 <= r < n and 1 <= s < n):
        return False

    z = hash_mensagem(mensagem) % n
    w = inverso_modular(s, n)
    u1 = (z * w) % n
    u2 = (r * w) % n

    P = u1 * curva.G + u2 * chave_publica
    if P.eh_infinito():
        return False

    x_verif = P.x % n
    return x_verif == r


# ------------------------------------------------------------
# 7. Demonstração de uso
# ------------------------------------------------------------

def demonstracao():
    print("=== DEMONSTRAÇÃO ECC (Curva secp256k1 didática) ===")
    curva = curva_secp256k1

    # 1) Geração de chaves
    print("\n[1] Gerando par de chaves ECC (usuário A)...")
    chave_priv_A, chave_pub_A = gerar_par_de_chaves(curva)
    print("Chave privada A:", hex(chave_priv_A))
    print("Chave pública A.x:", hex(chave_pub_A.x))
    print("Chave pública A.y:", hex(chave_pub_A.y))

    print("\n[2] Gerando par de chaves ECC (usuário B)...")
    chave_priv_B, chave_pub_B = gerar_par_de_chaves(curva)
    print("Chave privada B:", hex(chave_priv_B))
    print("Chave pública B.x:", hex(chave_pub_B.x))
    print("Chave pública B.y:", hex(chave_pub_B.y))

    # 2) ECDH – Segredo compartilhado
    print("\n[3] Derivando segredo compartilhado via ECDH...")
    segredo_AB = derivar_segredo_compartilhado(curva, chave_priv_A, chave_pub_B)
    segredo_BA = derivar_segredo_compartilhado(curva, chave_priv_B, chave_pub_A)
    print("Segredo A->B (SHA-256):", segredo_AB.hex())
    print("Segredo B->A (SHA-256):", segredo_BA.hex())
    print("Segredos iguais?", segredo_AB == segredo_BA)

    # 3) ECDSA – Assinatura de mensagem
    mensagem = b"Petroleo quantico, Neural Vasicek e IA estatistica - LT"
    print("\n[4] Assinando mensagem com chave privada A...")
    assinatura = assinar_mensagem(curva, chave_priv_A, mensagem)
    r, s = assinatura
    print("Assinatura (r, s):")
    print("  r =", hex(r))
    print("  s =", hex(s))

    print("\n[5] Verificando assinatura com chave pública A...")
    ok = verificar_assinatura(curva, chave_pub_A, mensagem, assinatura)
    print("Assinatura válida?", ok)

    # Teste de falha (mensagem adulterada)
    mensagem_falsa = b"Petroleo classico, nada de quantico - LT"
    ok_falsa = verificar_assinatura(curva, chave_pub_A, mensagem_falsa, assinatura)
    print("Assinatura na mensagem adulterada é válida?", ok_falsa)


if __name__ == "__main__":
    demonstracao()
