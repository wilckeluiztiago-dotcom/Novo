# ============================================================
# GO 19x19 
# Autor: Luiz Tiago Wilcke (LT)
# ============================================================

import math, random, threading, time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set

# --------------------- Dependências -------------------------
try:
    import pygame
except Exception as e:
        raise SystemExit("Precisa de pygame: pip install pygame") from e

try:
    import numpy as np
except Exception as e:
        raise SystemExit("Precisa de numpy: pip install numpy") from e

_torch_ok = True
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    _torch_ok = False

# --------------------- Parâmetros UI ------------------------
LADO = 19
TAM_CASA = 36
MARGEM = 60
LARGURA = MARGEM*2 + TAM_CASA*(LADO-1)
ALTURA  = MARGEM*2 + TAM_CASA*(LADO-1) + 160  # HUD maior
FPS = 60

# Paleta
COR_BG = (235, 228, 200)
COR_LINHA = (60, 50, 35)
COR_PRETO = (25, 25, 25)
COR_BRANCO = (240, 240, 240)
COR_TEXTO = (20, 28, 40)
COR_AZUL = (60, 130, 255)
COR_HUD = (255, 255, 255)

# --------------------- Auxílios de grid ---------------------
NEIGH = [(1,0),(-1,0),(0,1),(0,-1)]

def dentro(i,j): return 0 <= i < LADO and 0 <= j < LADO

def vizinhos(i,j):
    for di,dj in NEIGH:
        ni, nj = i+di, j+dj
        if dentro(ni,nj): yield (ni,nj)

def coord_para_px(i,j):
    x = MARGEM + j*TAM_CASA
    y = MARGEM + i*TAM_CASA
    return x, y

# --------------------- Zobrist hashing ----------------------
def init_zobrist(seed=1234):
    rng = random.Random(seed)
    table = np.zeros((LADO, LADO, 2), dtype=np.uint64)
    for i in range(LADO):
        for j in range(LADO):
            for p in range(2):
                table[i,j,p] = np.uint64(rng.getrandbits(64))
    turno_hash = np.uint64(rng.getrandbits(64))
    return table, turno_hash

ZOB_TABLE, ZOB_TURNO = init_zobrist()

def zob_hash(tab: np.ndarray, vez: int) -> int:
    h = np.uint64(0)
    pret_idx = (tab == 1)
    bran_idx = (tab == -1)
    ii, jj = np.where(pret_idx)
    for a,b in zip(ii, jj):
        h = np.bitwise_xor(h, ZOB_TABLE[a,b,0])
    ii, jj = np.where(bran_idx)
    for a,b in zip(ii, jj):
        h = np.bitwise_xor(h, ZOB_TABLE[a,b,1])
    if vez == 1:
        h = np.bitwise_xor(h, ZOB_TURNO)
    return int(h)

# --------------------- Regras do Go -------------------------
@dataclass
class Estado:
    tab: np.ndarray          # (19,19) em {0,1,-1}
    vez: int                 # 1=preto (IA), -1=branco (humano)
    ko_hashes: Set[int]
    capturas_preto: int = 0
    capturas_branco: int = 0
    passes_consecutivos: int = 0
    ultimo_lance: Optional[Tuple[int,int]] = None

def novo_estado():
    tab = np.zeros((LADO,LADO), dtype=np.int8)
    h = zob_hash(tab, 1)
    return Estado(tab=tab, vez=1, ko_hashes={h})

def grupo_e_liberdades(tab: np.ndarray, i: int, j: int) -> Tuple[List[Tuple[int,int]], Set[Tuple[int,int]]]:
    cor = tab[i,j]; assert cor != 0
    visit = {(i,j)}
    pilha = [(i,j)]
    libs: Set[Tuple[int,int]] = set()
    grupo: List[Tuple[int,int]] = []
    while pilha:
        a,b = pilha.pop()
        grupo.append((a,b))
        for ni,nj in vizinhos(a,b):
            if tab[ni,nj] == 0:
                libs.add((ni,nj))
            elif tab[ni,nj] == cor and (ni,nj) not in visit:
                visit.add((ni,nj)); pilha.append((ni,nj))
    return grupo, libs

def remover_grupo(tab: np.ndarray, grupo: List[Tuple[int,int]]) -> None:
    for (a,b) in grupo: tab[a,b] = 0

def legal(estado: Estado, i: Optional[int], j: Optional[int]) -> bool:
    if i is None and j is None:
        return True
    if not dentro(i,j) or estado.tab[i,j] != 0:
        return False
    tab2 = estado.tab.copy()
    cor = estado.vez
    tab2[i,j] = cor
    # capturas adjacentes
    capturas = 0
    for ni,nj in vizinhos(i,j):
        if tab2[ni,nj] == -cor:
            g, libs = grupo_e_liberdades(tab2, ni, nj)
            if len(libs) == 0:
                remover_grupo(tab2, g); capturas += len(g)
    # anti-suicídio
    if tab2[i,j] == cor:
        _, libs_self = grupo_e_liberdades(tab2, i, j)
        if len(libs_self) == 0 and capturas == 0:
            return False
    # ko simples
    novo_h = zob_hash(tab2, -cor)
    if novo_h in estado.ko_hashes:
        return False
    return True

def jogar(estado: Estado, i: Optional[int], j: Optional[int]) -> Estado:
    novo = Estado(
        tab = estado.tab.copy(),
        vez = estado.vez,
        ko_hashes = set(estado.ko_hashes),
        capturas_preto = estado.capturas_preto,
        capturas_branco = estado.capturas_branco,
        passes_consecutivos = estado.passes_consecutivos,
        ultimo_lance = estado.ultimo_lance
    )
    if i is None and j is None:
        novo.passes_consecutivos += 1
        novo.vez = -novo.vez
        novo.ultimo_lance = None
        novo.ko_hashes.add(zob_hash(novo.tab, novo.vez))
        return novo

    assert legal(estado, i, j)
    cor = novo.vez
    novo.tab[i,j] = cor

    capt = 0
    for ni,nj in vizinhos(i,j):
        if novo.tab[ni,nj] == -cor:
            g, libs = grupo_e_liberdades(novo.tab, ni, nj)
            if len(libs) == 0:
                remover_grupo(novo.tab, g); capt += len(g)
    if cor == 1: novo.capturas_preto += capt
    else:        novo.capturas_branco += capt

    novo.passes_consecutivos = 0
    novo.vez = -novo.vez
    novo.ultimo_lance = (i,j)
    novo.ko_hashes.add(zob_hash(novo.tab, novo.vez))
    return novo

def fim_de_jogo(estado: Estado) -> bool:
    return estado.passes_consecutivos >= 2

def estimar_territorio(tab: np.ndarray) -> Tuple[int,int]:
    visit = np.zeros_like(tab, dtype=bool)
    pret_pts = int(np.count_nonzero(tab == 1))
    bran_pts = int(np.count_nonzero(tab == -1))
    for i in range(LADO):
        for j in range(LADO):
            if tab[i,j] != 0 or visit[i,j]: continue
            fila = [(i,j)]; reg = []; visit[i,j] = True; toque: Set[int]=set()
            while fila:
                a,b = fila.pop()
                reg.append((a,b))
                for ni,nj in vizinhos(a,b):
                    if tab[ni,nj] == 0 and not visit[ni,nj]:
                        visit[ni,nj] = True; fila.append((ni,nj))
                    elif tab[ni,nj] != 0:
                        toque.add(int(tab[ni,nj]))
            if len(toque) == 1:
                dono = next(iter(toque))
                if dono == 1: pret_pts += len(reg)
                else: bran_pts += len(reg)
    return pret_pts, bran_pts

# --------------------- IA: Rede + MCTS ----------------------
POLICY_DIM = LADO*LADO + 1  # +1 = passar

class BlocoRes(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)
    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return F.relu(x + h)

class RedePV(nn.Module):
    def __init__(self, canais=64, blocos=5):
        super().__init__()
        self.inp = nn.Conv2d(5, canais, 3, padding=1)
        self.bn  = nn.BatchNorm2d(canais)
        self.res = nn.Sequential(*[BlocoRes(canais) for _ in range(blocos)])
        self.cp1 = nn.Conv2d(canais, 2, 1)
        self.cpb = nn.BatchNorm2d(2)
        self.cpf = nn.Linear(2*LADO*LADO, POLICY_DIM)
        self.cv1 = nn.Conv2d(canais, 1, 1)
        self.cvb = nn.BatchNorm2d(1)
        self.cvf1 = nn.Linear(LADO*LADO, 64)
        self.cvf2 = nn.Linear(64, 1)
    def forward(self, x):
        h = F.relu(self.bn(self.inp(x)))
        h = self.res(h)
        hp = F.relu(self.cpb(self.cp1(h)))
        hp = torch.flatten(hp, 1)
        logits = self.cpf(hp)
        hv = F.relu(self.cvb(self.cv1(h)))
        hv = torch.flatten(hv, 1)
        hv = F.relu(self.cvf1(hv))
        val = torch.tanh(self.cvf2(hv))
        return logits, val.squeeze(1)

def codificar_estado(estado: Estado) -> np.ndarray:
    tab = estado.tab; vez = estado.vez
    me = (tab == vez).astype(np.float32)
    op = (tab == -vez).astype(np.float32)
    vazio = (tab == 0).astype(np.float32)
    ult = np.zeros_like(tab, dtype=np.float32)
    if estado.ultimo_lance is not None:
        a,b = estado.ultimo_lance; ult[a,b] = 1.0
    vezp = np.full_like(tab, 1.0 if vez==1 else 0.0, dtype=np.float32)
    return np.stack([me,op,vazio,ult,vezp], axis=0)

@dataclass
class NoMCTS:
    P: float; N: int; W: float; Q: float; filhos: Dict[int,'NoMCTS']

def softmax_np(x, temp=1.0):
    x = x / max(1e-8, temp)
    x = x - np.max(x)
    e = np.exp(x)
    s = e.sum()
    return e / (s if s>0 else 1.0)

def idx_para_acao(a_idx: int) -> Tuple[Optional[int], Optional[int]]:
    if a_idx == POLICY_DIM-1: return None, None
    return a_idx // LADO, a_idx % LADO

def acao_para_idx(i: Optional[int], j: Optional[int]) -> int:
    return POLICY_DIM-1 if (i is None and j is None) else (i*LADO + j)

def movimentos_legais(estado: Estado) -> List[int]:
    acoes = [POLICY_DIM-1]  # passar sempre legal
    for i in range(LADO):
        for j in range(LADO):
            if estado.tab[i,j] == 0 and legal(estado, i, j):
                acoes.append(acao_para_idx(i,j))
    return acoes

def aplicar_acao_idx(estado: Estado, a_idx: int) -> Estado:
    i,j = idx_para_acao(a_idx)
    return jogar(estado, None, None) if i is None else jogar(estado, i, j)

def utilidade_terminal(estado: Estado, perspectiva_vez_inicial: int) -> float:
    pret, bran = estimar_territorio(estado.tab)
    pret += estado.capturas_preto
    bran += estado.capturas_branco + 6.5  # komi
    diff = (pret - bran) if perspectiva_vez_inicial==1 else (bran - pret)
    return 1.0 if diff > 0 else -1.0

def rollout_avaliacao(estado: Estado, passos=20) -> float:
    cur = estado
    for _ in range(passos):
        if fim_de_jogo(cur): break
        a = random.choice(movimentos_legais(cur))
        cur = aplicar_acao_idx(cur, a)
    return utilidade_terminal(cur, estado.vez)

class MCTS:
    def __init__(self, modelo: Optional[RedePV], sims=120, c_puct=1.4, temperatura=1.0, fallback=False, dispositivo="cpu"):
        self.modelo = modelo; self.sims = sims; self.c = c_puct
        self.temp = temperatura; self.fallback = fallback
        self.dispositivo = dispositivo
        self.cache: Dict[int, NoMCTS] = {}

    def prior_politica(self, estado: Estado) -> np.ndarray:
        legais = movimentos_legais(estado)
        mask = np.zeros(POLICY_DIM, dtype=np.float32); mask[legais] = 1.0
        if self.fallback or (self.modelo is None):
            s = mask.sum()
            return mask / (s if s>0 else 1.0)
        plano = codificar_estado(estado)
        tens = torch.from_numpy(plano).unsqueeze(0).to(self.dispositivo)
        with torch.no_grad():
            logits, _ = self.modelo(tens)
            logits = logits[0].cpu().numpy()
        logits = logits - 1e9*(1-mask)
        p = np.exp(logits - logits.max()); s = p.sum()
        return p / (s if s>0 else 1.0)

    def avaliar_valor(self, estado: Estado) -> float:
        if self.fallback or (self.modelo is None):
            return rollout_avaliacao(estado, passos=20)
        plano = codificar_estado(estado)
        tens = torch.from_numpy(plano).unsqueeze(0).to(self.dispositivo)
        with torch.no_grad():
            _, v = self.modelo(tens)
        return float(v.item())

    def selecionar(self, raiz_est: Estado) -> Tuple[List[Tuple[Estado,int]], float]:
        caminho: List[Tuple[Estado,int]] = []
        estado = raiz_est
        while True:
            h = zob_hash(estado.tab, estado.vez)
            if h not in self.cache:
                p = self.prior_politica(estado)
                no = NoMCTS(P=0.0, N=0, W=0.0, Q=0.0, filhos={})
                self.cache[h] = no
                for a in movimentos_legais(estado):
                    self.cache[h].filhos[a] = NoMCTS(P=float(p[a]), N=0, W=0.0, Q=0.0, filhos={})
                v = self.avaliar_valor(estado)
                return caminho, v
            no = self.cache[h]
            melhor_a, melhor_ucb = None, -1e18
            sqrtN = math.sqrt(max(1, no.N))
            for a, filho in no.filhos.items():
                u = filho.Q + self.c * filho.P * (sqrtN / (1 + filho.N))
                if u > melhor_ucb:
                    melhor_ucb, melhor_a = u, a
            caminho.append((estado, melhor_a))
            estado = aplicar_acao_idx(estado, melhor_a)
            if fim_de_jogo(estado):
                return caminho, utilidade_terminal(estado, raiz_est.vez)

    def retropropagar(self, caminho: List[Tuple[Estado,int]], valor: float):
        for est, a in reversed(caminho):
            h = zob_hash(est.tab, est.vez)
            no = self.cache[h]; filho = no.filhos[a]
            filho.N += 1; filho.W += valor; filho.Q = filho.W / filho.N
            no.N += 1
            valor = -valor

    def buscar(self, raiz: Estado) -> np.ndarray:
        for _ in range(self.sims):
            cam, v = self.selecionar(raiz)
            self.retropropagar(cam, v)
        h = zob_hash(raiz.tab, raiz.vez)
        no = self.cache.get(h)
        pi = np.zeros(POLICY_DIM, dtype=np.float32)
        if (no is None) or (len(no.filhos)==0):
            legais = movimentos_legais(raiz)
            for a in legais: pi[a] = 1.0
            s = pi.sum(); return pi/(s if s>0 else 1.0)
        for a, filho in no.filhos.items(): pi[a] = filho.N
        return softmax_np(np.log(pi + 1e-8), temp=self.temp)

# --------------------- IA em thread -------------------------
@dataclass
class Config:
    sims_mcts: int = 120
    temperatura: float = 1.0
    c_puct: float = 1.4
    uso_gpu: bool = False

class IAWorker(threading.Thread):
    def __init__(self, estado: Estado, conf: Config, modelo: Optional[RedePV]):
        super().__init__(daemon=True)
        self.estado_in = estado
        self.conf = conf
        self.modelo = modelo
        self.resulto: Optional[int] = None
    def run(self):
        dispositivo = "cuda" if (self.conf.uso_gpu and _torch_ok and torch.cuda.is_available()) else "cpu"
        fallback = (not _torch_ok) or (self.modelo is None)
        mcts = MCTS(modelo=self.modelo, sims=self.conf.sims_mcts, c_puct=self.conf.c_puct,
                    temperatura=self.conf.temperatura, fallback=fallback, dispositivo=dispositivo)
        pi = mcts.buscar(self.estado_in)
        self.resulto = int(np.argmax(pi))

# --------------------- UI / desenho -------------------------
def desenhar_tabuleiro(tela, estado: Estado, fonte, fonte_mini, hover_ij: Optional[Tuple[int,int]], pensando: bool, conf: Config):
    tela.fill(COR_BG)
    desenhar_dragao(tela)
    s = pygame.Surface((LARGURA, ALTURA-160), pygame.SRCALPHA)
    pygame.draw.rect(s, (255,255,255,60), s.get_rect(), border_radius=18)
    tela.blit(s, (0,0))

    # linhas
    for k in range(LADO):
        x0, y = coord_para_px(k,0)
        x1, _ = coord_para_px(k,LADO-1)
        pygame.draw.line(tela, COR_LINHA, (MARGEM, y), (MARGEM+TAM_CASA*(LADO-1), y), 2)
        x, y0 = coord_para_px(0,k)
        _, y1 = coord_para_px(LADO-1,k)
        pygame.draw.line(tela, COR_LINHA, (x, MARGEM), (x, MARGEM+TAM_CASA*(LADO-1)), 2)

    # hoshi
    for a in [3,9,15]:
        for b in [3,9,15]:
            x,y = coord_para_px(a,b)
            pygame.draw.circle(tela, COR_LINHA, (x,y), 5)

    # pedras
    for i in range(LADO):
        for j in range(LADO):
            v = estado.tab[i,j]
            if v == 0: continue
            x,y = coord_para_px(i,j)
            cor = COR_PRETO if v==1 else COR_BRANCO
            pygame.draw.circle(tela, (0,0,0), (x+2,y+2), 15)
            pygame.draw.circle(tela, cor, (x,y), 15)
            if estado.ultimo_lance == (i,j):
                pygame.draw.circle(tela, (255,0,0), (x,y), 4)

    # hover da jogada do humano (brancas)
    if hover_ij and estado.vez == -1 and estado.tab[hover_ij[0], hover_ij[1]] == 0:
        xh,yh = coord_para_px(*hover_ij)
        pygame.draw.circle(tela, (200,200,255), (xh,yh), 15, 2)

    # HUD
    pygame.draw.rect(tela, COR_HUD, (0, ALTURA-160, LARGURA, 160))
    pret_pts, bran_pts = estimar_territorio(estado.tab)
    linha1 = f"Capturas Preto(IA): {estado.capturas_preto} | Capturas Branco(Você): {estado.capturas_branco}"
    linha2 = f"Estimativa — Preto: {pret_pts} | Branco+komi: {bran_pts+6.5:.1f}"
    tela.blit(fonte.render(linha1, True, COR_TEXTO), (20, ALTURA-150))
    tela.blit(fonte.render(linha2, True, COR_TEXTO), (20, ALTURA-120))

    # status/teclas
    pensando_txt = "IA pensando" + "."*((pygame.time.get_ticks()//300)%4) if pensando else "Sua vez (brancas)"
    controles = "[P]=Passar  [R]=Reiniciar  [+/-]=Sims MCTS  [G]=GPU on/off"
    tela.blit(fonte_mini.render(pensando_txt, True, (60,70,95)), (20, ALTURA-90))
    tela.blit(fonte_mini.render(controles+f" | sims={conf.sims_mcts}", True, (60,70,95)), (20, ALTURA-65))

def desenhar_dragao(tela):
    base = pygame.Surface((LARGURA, ALTURA-160), pygame.SRCALPHA)
    pts = []
    for k in range(0, 720, 8):
        t = math.radians(k)
        x = LARGURA*0.15 + k*(LARGURA*0.7/720.0)
        y = MARGEM + (ALTURA-300)/2 + math.sin(t*1.5)*80 + math.sin(t*0.25)*20
        pts.append((x,y))
    for idx, (x,y) in enumerate(pts):
        r = int(10 + 6*math.sin(idx*0.2))
        pygame.draw.circle(base, (0,0,0,40), (int(x), int(y)), r+2)
        pygame.draw.circle(base, (50,40,30,140), (int(x), int(y)), r)
    for idx in range(0, len(pts), 12):
        if idx+8 < len(pts):
            x1,y1 = pts[idx]; x2,y2 = pts[idx+8]
            mx,my = (x1+x2)/2, (y1+y2)/2 - 24
            pygame.draw.polygon(base, (90, 20, 20, 160), [(x1,y1),(mx,my),(x2,y2)])
    hx, hy = pts[-1]
    pygame.draw.circle(base, (40, 35, 30, 200), (int(hx), int(hy)), 28)
    pygame.draw.circle(base, (255, 255, 255, 200), (int(hx+8), int(hy-6)), 5)
    pygame.draw.circle(base, (0, 0, 0, 220), (int(hx+8), int(hy-6)), 2)
    pygame.draw.line(base, (60,30,20,200), (hx-5,hy-15), (hx+24, hy-40), 5)
    pygame.draw.line(base, (60,30,20,200), (hx-8,hy-10), (hx+8, hy-38), 4)
    for ang in (-20, -8, 6):
        a = math.radians(ang)
        for s in range(0, 40, 5):
            x = hx + 30 + s*math.cos(a)
            y = hy - 10 + s*math.sin(a) + 4*math.sin(s*0.3)
            pygame.draw.circle(base, (50,40,30,160), (int(x),int(y)), 2)
    for i in range(0, LARGURA, 10):
        for j in range(0, ALTURA-160, 10):
            if (i//10 + j//10) % 2 == 0:
                base.set_at((i,j), (255,255,255,18))
    tela.blit(base, (0,0))

# --------------------- Main loop ----------------------------
def main():
    pygame.init()
    tela = pygame.display.set_mode((LARGURA, ALTURA))
    pygame.display.set_caption("Go 19x19 — IA (preto) vs Humano (branco)")
    relogio = pygame.time.Clock()
    fonte = pygame.font.SysFont("arial", 22)
    fonte_mini = pygame.font.SysFont("arial", 18)

    conf = Config()
    estado = novo_estado()
    fim = False

    # IA “morna”: cria rede (se tiver torch)
    modelo: Optional[RedePV] = None
    if _torch_ok:
        try:
            dispositivo = "cuda" if (conf.uso_gpu and torch.cuda.is_available()) else "cpu"
            modelo = RedePV().to(dispositivo)
            for m in modelo.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        except Exception:
            modelo = None

    trabalhador: Optional[IAWorker] = None
    pensando = False
    proxima_acao_idx: Optional[int] = None
    hover_ij: Optional[Tuple[int,int]] = None

    DIST_MAX2 = (0.45*TAM_CASA)**2  # limiar de clique

    rodando = True
    while rodando:
        # -------- eventos --------
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                rodando = False

            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_p and not fim and estado.vez == -1:
                    if legal(estado, None, None):
                        estado = jogar(estado, None, None)
                elif ev.key == pygame.K_r:
                    estado = novo_estado(); fim = False
                    trabalhador = None; pensando = False; proxima_acao_idx = None
                elif ev.key == pygame.K_PLUS or ev.key == pygame.K_EQUALS:
                    conf.sims_mcts = min(conf.sims_mcts+20, 400)
                elif ev.key == pygame.K_MINUS:
                    conf.sims_mcts = max(conf.sims_mcts-20, 20)
                elif ev.key == pygame.K_g:
                    conf.uso_gpu = not conf.uso_gpu

            elif ev.type == pygame.MOUSEMOTION:
                mx,my = ev.pos
                # detectar intersecção mais próxima, com limite
                best, bestd = None, 1e18
                for i in range(LADO):
                    for j in range(LADO):
                        x,y = coord_para_px(i,j)
                        d = (mx-x)**2 + (my-y)**2
                        if d < bestd:
                            bestd = d; best=(i,j)
                hover_ij = best if (best and bestd <= DIST_MAX2) else None

            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and not fim:
                # humano joga brancas
                if estado.vez == -1 and hover_ij is not None:
                    i,j = hover_ij
                    if legal(estado, i, j):
                        estado = jogar(estado, i, j)

        # -------- lógica de turno --------
        if fim_de_jogo(estado) and not fim:
            fim = True
            pret_pts, bran_pts = estimar_territorio(estado.tab)
            pret_pts += estado.capturas_preto
            bran_pts += estado.capturas_branco + 6.5
            vencedor = "Preto (IA)" if pret_pts > bran_pts else "Branco (Você)"
            print(f"Fim! Preto: {pret_pts} | Branco+komi: {bran_pts:.1f} → Vencedor: {vencedor}")

        # IA só inicia thread quando for a vez dela e não estiver pensando
        if not fim and estado.vez == 1:
            if not pensando and trabalhador is None:
                pensando = True
                trabalhador = IAWorker(estado, conf, modelo)
                trabalhador.start()
            # se já rodando, checa resultado
            if trabalhador is not None and not trabalhador.is_alive():
                pensando = False
                proxima_acao_idx = trabalhador.resulto
                trabalhador = None
                if proxima_acao_idx is not None:
                    estado = aplicar_acao_idx(estado, proxima_acao_idx)
                    proxima_acao_idx = None

        # -------- desenho --------
        desenhar_tabuleiro(tela, estado, fonte, fonte_mini, hover_ij, pensando, conf)

        # topo
        topo = pygame.Surface((LARGURA, 36), pygame.SRCALPHA)
        pygame.draw.rect(topo, (255,255,255,200), topo.get_rect())
        txt = "Preto (IA MCTS+Rede)  vs  Branco (Humano) — [P]=Passar  [R]=Reiniciar  [+/-]=Sims  [G]=GPU"
        tela.blit(pygame.font.SysFont("arial",18).render(txt, True, (10,15,35)), (12,6))

        pygame.display.flip()
        pygame.event.pump()
        pygame.time.delay(5)   # respira um pouco
        relogio.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()
