"""
Motor de Inteligência Artificial
Implementa Minimax com Alpha-Beta Pruning, Tabela de Transposição e Principal Variation Search (PVS).
"""

import time
from typing import Optional, Tuple, Dict, List
from config import *
from utils.tipos import Movimento, EntradaTT
from motor.tabuleiro import Estado
from motor.avaliacao import Avaliador

class MotorIA:
    def __init__(self, avaliador: Avaliador, tempo_por_lance: float = 3.0):
        self.avaliador = avaliador
        self.tempo_limite = tempo_por_lance
        self.tabela_transposicao: Dict[int, EntradaTT] = {}
        self.inicio_busca = 0.0
        self.nos_avaliados = 0
        self.melhor_movimento_global: Optional[Movimento] = None

    def _ordenar_movimentos(self, movimentos: List[Movimento], melhor_tt: Optional[Movimento] = None) -> None:
        def score(m: Movimento) -> int:
            if melhor_tt and m.origem == melhor_tt.origem and m.destino == melhor_tt.destino:
                return 20000 # PV move primeiro
            s = 0
            # MVV-LVA (Most Valuable Victim - Least Valuable Aggressor)
            if m.capturada != VAZIO:
                s += abs(VALOR_PECA[m.capturada]) * 10 - abs(VALOR_PECA[m.peca])
            if m.promocao != VAZIO: s += 900
            if m.roque: s += 50
            return s
        movimentos.sort(key=score, reverse=True)

    def escolher_movimento(self, estado: Estado) -> Optional[Movimento]:
        self.inicio_busca = time.time()
        self.nos_avaliados = 0
        self.melhor_movimento_global = None
        
        profundidade_maxima = 6 # Pode aumentar se tiver tempo
        
        # Iterative Deepening
        for prof in range(1, profundidade_maxima + 1):
            valor = self._negamax(estado, prof, -1000000, 1000000)
            
            # Se o tempo acabou durante a busca, não confie no resultado incompleto
            if time.time() - self.inicio_busca > self.tempo_limite:
                break
                
            print(f"Profundidade {prof}: Valor {valor/100:.2f}, Nós {self.nos_avaliados}")
            
        return self.melhor_movimento_global

    def _negamax(self, estado: Estado, profundidade: int, alfa: int, beta: int) -> int:
        if time.time() - self.inicio_busca > self.tempo_limite:
            return self.avaliador.avaliar(estado) * (1 if estado.brancas_jogam else -1)

        self.nos_avaliados += 1
        alpha_orig = alfa

        # Tabela de Transposição
        entrada = self.tabela_transposicao.get(estado.hash_atual)
        if entrada and entrada.profundidade >= profundidade:
            if entrada.flag == 0: return entrada.valor
            elif entrada.flag == 1: alfa = max(alfa, entrada.valor)
            elif entrada.flag == 2: beta = min(beta, entrada.valor)
            if alfa >= beta: return entrada.valor

        if profundidade == 0:
            return self._busca_quiescencia(estado, alfa, beta)

        movimentos = estado.gerar_movimentos_legais()
        if not movimentos:
            if estado.em_xeque(estado.brancas_jogam):
                return -999999 + (100 - profundidade) # Mate (prefira mates mais rápidos)
            return 0 # Afogamento

        # Ordenação
        melhor_mov_tt = entrada.melhor_movimento if entrada else None
        self._ordenar_movimentos(movimentos, melhor_mov_tt)

        melhor_valor = -1000000
        melhor_movimento_local = None

        for i, movimento in enumerate(movimentos):
            estado.fazer_movimento(movimento)
            
            # Principal Variation Search (PVS)
            if i == 0:
                valor = -self._negamax(estado, profundidade - 1, -beta, -alfa)
            else:
                # Null Window Search
                valor = -self._negamax(estado, profundidade - 1, -alfa - 1, -alfa)
                if alfa < valor < beta:
                    valor = -self._negamax(estado, profundidade - 1, -beta, -alfa)
            
            estado.desfazer_movimento()

            if valor > melhor_valor:
                melhor_valor = valor
                melhor_movimento_local = movimento
                # Atualiza global apenas na raiz
                if profundidade == 6: # Ajustar lógica para pegar da raiz corretamente
                    pass 

            alfa = max(alfa, valor)
            if alfa >= beta:
                break

        # Salva na TT
        flag = 0
        if melhor_valor <= alpha_orig: flag = 2 # Upper bound
        elif melhor_valor >= beta: flag = 1 # Lower bound
        
        self.tabela_transposicao[estado.hash_atual] = EntradaTT(
            estado.hash_atual, profundidade, melhor_valor, flag, melhor_movimento_local
        )
        
        # Na raiz, atualiza o melhor movimento global
        # (Isso é uma simplificação, idealmente a raiz chama uma função separada)
        # Mas como usamos iterative deepening, precisamos salvar o melhor da iteração atual
        # Vamos assumir que quem chamou _negamax na raiz captura o movimento da TT ou lógica separada
        # Para simplificar aqui:
        if self.tabela_transposicao.get(estado.hash_atual):
             # Hack: se estamos na raiz (chamada externa), atualizamos
             pass

        return melhor_valor

    # Sobrescrevendo para simplificar a raiz e garantir o movimento
    def escolher_movimento(self, estado: Estado) -> Optional[Movimento]:
        self.inicio_busca = time.time()
        self.nos_avaliados = 0
        self.melhor_movimento_global = None
        
        prof_max = 5
        
        for prof in range(1, prof_max + 1):
            alfa = -1000000
            beta = 1000000
            melhor_valor = -1000000
            melhor_mov = None
            
            movimentos = estado.gerar_movimentos_legais()
            if not movimentos: break
            
            # Ordena usando TT da iteração anterior
            entrada = self.tabela_transposicao.get(estado.hash_atual)
            pv_move = entrada.melhor_movimento if entrada else None
            self._ordenar_movimentos(movimentos, pv_move)
            
            for i, mov in enumerate(movimentos):
                estado.fazer_movimento(mov)
                
                if i == 0:
                    val = -self._negamax(estado, prof - 1, -beta, -alfa)
                else:
                    val = -self._negamax(estado, prof - 1, -alfa - 1, -alfa)
                    if alfa < val < beta:
                        val = -self._negamax(estado, prof - 1, -beta, -alfa)
                
                estado.desfazer_movimento()
                
                if time.time() - self.inicio_busca > self.tempo_limite:
                    break
                
                if val > melhor_valor:
                    melhor_valor = val
                    melhor_mov = mov
                
                alfa = max(alfa, val)
            
            if time.time() - self.inicio_busca > self.tempo_limite:
                break
                
            self.melhor_movimento_global = melhor_mov
            print(f"Prof {prof}: {melhor_valor}")
            
        return self.melhor_movimento_global

    def _busca_quiescencia(self, estado: Estado, alfa: int, beta: int) -> int:
        val_estatico = self.avaliador.avaliar(estado) * (1 if estado.brancas_jogam else -1)
        if val_estatico >= beta: return beta
        if val_estatico > alfa: alfa = val_estatico
        
        movimentos = estado.gerar_movimentos_legais()
        # Apenas capturas
        capturas = [m for m in movimentos if m.capturada != VAZIO]
        self._ordenar_movimentos(capturas)
        
        for mov in capturas:
            estado.fazer_movimento(mov)
            val = -self._busca_quiescencia(estado, -beta, -alfa)
            estado.desfazer_movimento()
            
            if val >= beta: return beta
            if val > alfa: alfa = val
            
        return alfa
