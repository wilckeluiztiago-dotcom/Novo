import pygame
import sys
import copy

pygame.init()

# Constantes e dimensões
LARGURA_TABULEIRO = 640
ALTURA_TABULEIRO = 640
LARGURA_JANELA = 1000  # Espaço extra para informações
ALTURA_JANELA = 640
TAMANHO_QUADRADO = LARGURA_TABULEIRO // 8

# Cores
BRANCO = (255, 255, 255)
PRETO = (0, 0, 0)
AZUL = (0, 0, 255)
VERMELHO = (255, 0, 0)
CINZA = (128, 128, 128)
VERDE = (0, 255, 0)
AMARELO = (255, 255, 0)
AZUL_LEGENDA = (0, 0, 255)

# Criação da janela
tela = pygame.display.set_mode((LARGURA_JANELA, ALTURA_JANELA))
pygame.display.set_caption('Jogo de Xadrez - Versão Melhorada')

# Inicialização das fontes (tentando encontrar uma fonte que suporte os símbolos Unicode de xadrez)
pygame.font.init()
fontes_disponiveis = ['Segoe UI Symbol', 'Arial Unicode MS', 'DejaVu Sans', 'FreeSerif', 'Symbola']
for nome in fontes_disponiveis:
    try:
        FONTE_PECA = pygame.font.SysFont(nome, TAMANHO_QUADRADO - 10)
        teste_texto = FONTE_PECA.render('\u2654', True, PRETO)
        if teste_texto:
            break
    except:
        continue
else:
    print("Nenhuma fonte adequada encontrada. Certifique-se de ter uma fonte que suporte os símbolos Unicode de xadrez.")
    pygame.quit()
    sys.exit()

FONTE_INFO = pygame.font.SysFont(None, 24)
FONTE_LEGENDA = pygame.font.SysFont(None, 30)
FONTE_MENU = pygame.font.SysFont(None, 40)

# Mapeamento dos símbolos Unicode das peças
SIMBOLOS_PECAS = {
    'rei_azul': '\u2654',     
    'rainha_azul': '\u2655',  
    'torre_azul': '\u2656',   
    'bispo_azul': '\u2657',   
    'cavalo_azul': '\u2658',  
    'peao_azul': '\u2659',    
    'rei_vermelho': '\u265A',     
    'rainha_vermelho': '\u265B',  
    'torre_vermelho': '\u265C',   
    'bispo_vermelho': '\u265D',   
    'cavalo_vermelho': '\u265E',  
    'peao_vermelho': '\u265F',    
}

# Classe que representa uma peça de xadrez
class Peca:
    def __init__(self, tipo, cor):
        self.tipo = tipo  # 'rei', 'rainha', 'bispo', 'cavalo', 'torre', 'peao'
        self.cor = cor    # 'azul' ou 'vermelho'
        self.simbolo = SIMBOLOS_PECAS[f'{tipo}_{cor}']
        self.movimentos_realizados = 0

    def get_movimentos_validos(self, x, y, tabuleiro, roque_disponivel):
        movimentos = []
        if self.tipo == 'peao':
            direcao = -1 if self.cor == 'azul' else 1
            novo_y = y + direcao
            # Movimento para frente
            if 0 <= novo_y < 8:
                if tabuleiro[novo_y][x] is None:
                    movimentos.append((x, novo_y))
                    # Duplo avanço se for o primeiro movimento
                    if (self.cor == 'azul' and y == 6) or (self.cor == 'vermelho' and y == 1):
                        novo_y2 = y + 2 * direcao
                        if tabuleiro[novo_y2][x] is None:
                            movimentos.append((x, novo_y2))
                # Capturas diagonais
                for dx in [-1, 1]:
                    novo_x = x + dx
                    if 0 <= novo_x < 8:
                        destino = tabuleiro[novo_y][novo_x]
                        if destino and destino.cor != self.cor:
                            movimentos.append((novo_x, novo_y))
                        # A implementação de en passant pode ser acrescentada aqui
        elif self.tipo == 'torre':
            direcoes = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for dx, dy in direcoes:
                nx, ny = x + dx, y + dy
                while 0 <= nx < 8 and 0 <= ny < 8:
                    destino = tabuleiro[ny][nx]
                    if destino is None:
                        movimentos.append((nx, ny))
                    elif destino.cor != self.cor:
                        movimentos.append((nx, ny))
                        break
                    else:
                        break
                    nx += dx
                    ny += dy
            # As condições para roque podem ser implementadas aqui
        elif self.tipo == 'bispo':
            direcoes = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
            for dx, dy in direcoes:
                nx, ny = x + dx, y + dy
                while 0 <= nx < 8 and 0 <= ny < 8:
                    destino = tabuleiro[ny][nx]
                    if destino is None:
                        movimentos.append((nx, ny))
                    elif destino.cor != self.cor:
                        movimentos.append((nx, ny))
                        break
                    else:
                        break
                    nx += dx
                    ny += dy
        elif self.tipo == 'rainha':
            direcoes = [(-1, 0), (1, 0), (0, -1), (0, 1),
                        (-1, -1), (-1, 1), (1, -1), (1, 1)]
            for dx, dy in direcoes:
                nx, ny = x + dx, y + dy
                while 0 <= nx < 8 and 0 <= ny < 8:
                    destino = tabuleiro[ny][nx]
                    if destino is None:
                        movimentos.append((nx, ny))
                    elif destino.cor != self.cor:
                        movimentos.append((nx, ny))
                        break
                    else:
                        break
                    nx += dx
                    ny += dy
        elif self.tipo == 'rei':
            direcoes = [(-1, -1), (-1, 0), (-1, 1),
                        (0, -1),         (0, 1),
                        (1, -1), (1, 0), (1, 1)]
            for dx, dy in direcoes:
                nx, ny = x + dx, y + dy
                if 0 <= nx < 8 and 0 <= ny < 8:
                    destino = tabuleiro[ny][nx]
                    if destino is None or destino.cor != self.cor:
                        movimentos.append((nx, ny))
            # O roque também pode ser implementado aqui
        elif self.tipo == 'cavalo':
            possiveis = [(x+1, y+2), (x+1, y-2), (x-1, y+2), (x-1, y-2),
                         (x+2, y+1), (x+2, y-1), (x-2, y+1), (x-2, y-1)]
            for nx, ny in possiveis:
                if 0 <= nx < 8 and 0 <= ny < 8:
                    destino = tabuleiro[ny][nx]
                    if destino is None or destino.cor != self.cor:
                        movimentos.append((nx, ny))
        return movimentos

# Classe que representa o estado do jogo
class Jogo:
    def __init__(self):
        self.tabuleiro = [[None for _ in range(8)] for _ in range(8)]
        self.jogador_atual = 'azul'  # 'azul' é o jogador; 'vermelho' será a IA
        self.historico = []  # Histórico de movimentos
        self.roque_disponivel = {
            'azul': {'roque_menos': True, 'roque_mais': True},
            'vermelho': {'roque_menos': True, 'roque_mais': True}
        }
        self.iniciar_tabuleiro()

    def iniciar_tabuleiro(self):
        # Peças do jogador (azul)
        for i in range(8):
            self.tabuleiro[6][i] = Peca('peao', 'azul')
        self.tabuleiro[7][0] = Peca('torre', 'azul')
        self.tabuleiro[7][1] = Peca('cavalo', 'azul')
        self.tabuleiro[7][2] = Peca('bispo', 'azul')
        self.tabuleiro[7][3] = Peca('rainha', 'azul')
        self.tabuleiro[7][4] = Peca('rei', 'azul')
        self.tabuleiro[7][5] = Peca('bispo', 'azul')
        self.tabuleiro[7][6] = Peca('cavalo', 'azul')
        self.tabuleiro[7][7] = Peca('torre', 'azul')

        # Peças da IA (vermelho)
        for i in range(8):
            self.tabuleiro[1][i] = Peca('peao', 'vermelho')
        self.tabuleiro[0][0] = Peca('torre', 'vermelho')
        self.tabuleiro[0][1] = Peca('cavalo', 'vermelho')
        self.tabuleiro[0][2] = Peca('bispo', 'vermelho')
        self.tabuleiro[0][3] = Peca('rainha', 'vermelho')
        self.tabuleiro[0][4] = Peca('rei', 'vermelho')
        self.tabuleiro[0][5] = Peca('bispo', 'vermelho')
        self.tabuleiro[0][6] = Peca('cavalo', 'vermelho')
        self.tabuleiro[0][7] = Peca('torre', 'vermelho')

    def desenhar_tabuleiro(self):
        for y in range(8):
            for x in range(8):
                cor = BRANCO if (x + y) % 2 == 0 else CINZA
                pygame.draw.rect(tela, cor, (x * TAMANHO_QUADRADO, y * TAMANHO_QUADRADO, TAMANHO_QUADRADO, TAMANHO_QUADRADO))
                peca = self.tabuleiro[y][x]
                if peca:
                    texto = FONTE_PECA.render(peca.simbolo, True, PRETO)
                    pos_texto = texto.get_rect(center=(x * TAMANHO_QUADRADO + TAMANHO_QUADRADO // 2,
                                                        y * TAMANHO_QUADRADO + TAMANHO_QUADRADO // 2))
                    tela.blit(texto, pos_texto)

    def desenhar_info(self):
        # Área de informações à direita do tabuleiro
        pygame.draw.line(tela, PRETO, (LARGURA_TABULEIRO, 0), (LARGURA_TABULEIRO, ALTURA_TABULEIRO), 2)
        titulo = FONTE_INFO.render('Histórico de Movimentos:', True, PRETO)
        tela.blit(titulo, (LARGURA_TABULEIRO + 20, 10))
        y_offset = 40
        for cor_jogador, desc in self.historico[-25:]:
            linhas = self.dividir_texto(desc, 300, FONTE_INFO)
            for linha in linhas:
                cor_texto = AZUL if cor_jogador == 'azul' else VERMELHO
                texto = FONTE_INFO.render(linha, True, cor_texto)
                tela.blit(texto, (LARGURA_TABULEIRO + 20, y_offset))
                y_offset += 20
                if y_offset > ALTURA_TABULEIRO - 60:
                    break
        legenda = FONTE_LEGENDA.render('Autor: Luiz Tiago Wilcke', True, AZUL_LEGENDA)
        tela.blit(legenda, (LARGURA_TABULEIRO + 20, ALTURA_TABULEIRO - 40))

    def dividir_texto(self, texto, largura_max, fonte):
        palavras = texto.split(' ')
        linhas = []
        linha_atual = ""
        for palavra in palavras:
            if fonte.size(linha_atual + palavra + " ")[0] < largura_max:
                linha_atual += palavra + " "
            else:
                linhas.append(linha_atual)
                linha_atual = palavra + " "
        if linha_atual:
            linhas.append(linha_atual)
        return linhas

    def mover_peca(self, origem, destino, is_ai_move=False, eval_score=None):
        x1, y1 = origem
        x2, y2 = destino
        peca = self.tabuleiro[y1][x1]
        self.tabuleiro[y2][x2] = peca
        self.tabuleiro[y1][x1] = None
        peca.movimentos_realizados += 1

        # Promoção de peão
        if peca.tipo == 'peao' and (y2 == 0 or y2 == 7):
            self.promocao_peao(x2, y2, peca.cor)

        # Atualiza disponibilidade para roque (implementação simplificada)
        if peca.tipo == 'rei':
            self.roque_disponivel[peca.cor]['roque_mais'] = False
            self.roque_disponivel[peca.cor]['roque_menos'] = False
        if peca.tipo == 'torre':
            if x1 == 0:
                self.roque_disponivel[peca.cor]['roque_menos'] = False
            elif x1 == 7:
                self.roque_disponivel[peca.cor]['roque_mais'] = False

        if is_ai_move:
            desc = f"IA move {peca.tipo.capitalize()} de ({x1},{y1}) para ({x2},{y2}) | Eval: {eval_score}"
            self.historico.append(('vermelho', desc))
        else:
            desc = f"Jogador move {peca.tipo.capitalize()} de ({x1},{y1}) para ({x2},{y2})"
            self.historico.append(('azul', desc))

    def promocao_peao(self, x, y, cor):
        promovido = False
        while not promovido:
            tela.fill(BRANCO)
            fonte_promocao = pygame.font.SysFont(None, 40)
            texto = fonte_promocao.render('Escolha: (R)ainha, (B)ispo, (C)avalo ou (T)orre:', True, PRETO)
            tela.blit(texto, (20, ALTURA_JANELA // 2 - 50))
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        nova = Peca('rainha', cor)
                        promovido = True
                    elif event.key == pygame.K_b:
                        nova = Peca('bispo', cor)
                        promovido = True
                    elif event.key == pygame.K_c:
                        nova = Peca('cavalo', cor)
                        promovido = True
                    elif event.key == pygame.K_t:
                        nova = Peca('torre', cor)
                        promovido = True
                    if promovido:
                        self.tabuleiro[y][x] = nova

    def esta_em_xeque(self, cor):
        rei_pos = None
        for y in range(8):
            for x in range(8):
                peca = self.tabuleiro[y][x]
                if peca and peca.tipo == 'rei' and peca.cor == cor:
                    rei_pos = (x, y)
                    break
            if rei_pos:
                break
        if not rei_pos:
            return False
        cor_oponente = 'vermelho' if cor == 'azul' else 'azul'
        for y in range(8):
            for x in range(8):
                peca = self.tabuleiro[y][x]
                if peca and peca.cor == cor_oponente:
                    if rei_pos in peca.get_movimentos_validos(x, y, self.tabuleiro, self.roque_disponivel):
                        return True
        return False

    def esta_em_xeque_mate(self, cor):
        if not self.esta_em_xeque(cor):
            return False
        movimentos = self.obter_movimentos_validos(cor)
        for mov in movimentos:
            copia = copy.deepcopy(self)
            copia.mover_peca(*mov)
            if not copia.esta_em_xeque(cor):
                return False
        return True

    def obter_movimentos_validos(self, cor):
        movimentos = []
        for y in range(8):
            for x in range(8):
                peca = self.tabuleiro[y][x]
                if peca and peca.cor == cor:
                    possiveis = peca.get_movimentos_validos(x, y, self.tabuleiro, self.roque_disponivel)
                    for dest in possiveis:
                        movimentos.append(((x, y), dest))
        legais = []
        for origem, destino in movimentos:
            copia = copy.deepcopy(self)
            copia.mover_peca(origem, destino)
            if not copia.esta_em_xeque(cor):
                legais.append((origem, destino))
        return legais

    def valor_peca(self, peca):
        valores = {'peao': 10, 'cavalo': 30, 'bispo': 30, 'torre': 50, 'rainha': 90, 'rei': 900}
        return valores.get(peca.tipo, 0)

    def avaliar_tabuleiro(self):
        total = 0
        for y in range(8):
            for x in range(8):
                peca = self.tabuleiro[y][x]
                if peca:
                    if peca.cor == 'vermelho':
                        total += self.valor_peca(peca)
                    else:
                        total -= self.valor_peca(peca)
        return total

    def minimax(self, profundidade, maximizando, alpha=float('-inf'), beta=float('inf')):
        cor = 'vermelho' if maximizando else 'azul'
        if profundidade == 0 or self.esta_em_xeque_mate('azul') or self.esta_em_xeque_mate('vermelho'):
            return self.avaliar_tabuleiro(), None
        movimentos = self.obter_movimentos_validos(cor)
        if not movimentos:
            return self.avaliar_tabuleiro(), None
        melhor_movimento = None
        if maximizando:
            max_eval = float('-inf')
            for mov in movimentos:
                copia = copy.deepcopy(self)
                copia.mover_peca(*mov)
                eval_atual, _ = copia.minimax(profundidade - 1, False, alpha, beta)
                if eval_atual > max_eval:
                    max_eval = eval_atual
                    melhor_movimento = mov
                alpha = max(alpha, eval_atual)
                if beta <= alpha:
                    break
            return max_eval, melhor_movimento
        else:
            min_eval = float('inf')
            for mov in movimentos:
                copia = copy.deepcopy(self)
                copia.mover_peca(*mov)
                eval_atual, _ = copia.minimax(profundidade - 1, True, alpha, beta)
                if eval_atual < min_eval:
                    min_eval = eval_atual
                    melhor_movimento = mov
                beta = min(beta, eval_atual)
                if beta <= alpha:
                    break
            return min_eval, melhor_movimento

# Tela de menu para seleção da dificuldade da IA
def menu_inicial():
    selecionado = None
    while selecionado is None:
        tela.fill(BRANCO)
        titulo = FONTE_MENU.render("Selecione a Dificuldade", True, PRETO)
        tela.blit(titulo, (LARGURA_TABULEIRO // 2 - titulo.get_width() // 2, 100))
        btn_facil = FONTE_MENU.render("Fácil", True, AZUL)
        btn_medio = FONTE_MENU.render("Médio", True, AZUL)
        btn_dificil = FONTE_MENU.render("Difícil", True, AZUL)
        pos_facil = btn_facil.get_rect(center=(LARGURA_TABULEIRO // 2, 200))
        pos_medio = btn_medio.get_rect(center=(LARGURA_TABULEIRO // 2, 300))
        pos_dificil = btn_dificil.get_rect(center=(LARGURA_TABULEIRO // 2, 400))
        tela.blit(btn_facil, pos_facil)
        tela.blit(btn_medio, pos_medio)
        tela.blit(btn_dificil, pos_dificil)
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                if pos_facil.collidepoint(mx, my):
                    selecionado = 2   # Profundidade 2 para nível Fácil
                elif pos_medio.collidepoint(mx, my):
                    selecionado = 3   # Profundidade 3 para nível Médio
                elif pos_dificil.collidepoint(mx, my):
                    selecionado = 4   # Profundidade 4 para nível Difícil
    return selecionado

def main():
    # Seleção do nível de dificuldade (define a profundidade da IA)
    ai_depth = menu_inicial()

    jogo = Jogo()
    selecionado = None
    rodando = True
    fim_de_jogo = False

    while rodando:
        jogo.desenhar_tabuleiro()
        jogo.desenhar_info()
        pygame.display.flip()

        if fim_de_jogo:
            fonte_fim = pygame.font.SysFont(None, 50)
            texto_fim = fonte_fim.render('Xeque-mate!', True, PRETO)
            pos_texto = texto_fim.get_rect(center=(LARGURA_TABULEIRO // 2, ALTURA_TABULEIRO // 2))
            tela.blit(texto_fim, pos_texto)
            pygame.display.flip()
            pygame.time.wait(3000)
            rodando = False
            continue

        if jogo.jogador_atual == 'vermelho':
            # Turno da IA
            eval_score, melhor_mov = jogo.minimax(ai_depth, True)
            if melhor_mov:
                jogo.mover_peca(*melhor_mov, is_ai_move=True, eval_score=eval_score)
                print(f"IA move de {melhor_mov[0]} para {melhor_mov[1]} | Eval: {eval_score}")
            if jogo.esta_em_xeque_mate('azul'):
                fim_de_jogo = True
            jogo.jogador_atual = 'azul'
            continue

        # Turno do jogador (azul)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                rodando = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                if mx >= LARGURA_TABULEIRO or my >= ALTURA_TABULEIRO:
                    continue
                x = mx // TAMANHO_QUADRADO
                y = my // TAMANHO_QUADRADO
                if selecionado:
                    if (x, y) in selecionado[2]:
                        jogo.mover_peca((selecionado[0], selecionado[1]), (x, y))
                        if jogo.esta_em_xeque_mate('vermelho'):
                            fim_de_jogo = True
                        jogo.jogador_atual = 'vermelho'
                        selecionado = None
                    else:
                        selecionado = None
                else:
                    peca = jogo.tabuleiro[y][x]
                    if peca and peca.cor == 'azul':
                        possiveis = peca.get_movimentos_validos(x, y, jogo.tabuleiro, jogo.roque_disponivel)
                        validos = []
                        for dest in possiveis:
                            copia = copy.deepcopy(jogo)
                            copia.mover_peca((x, y), dest)
                            if not copia.esta_em_xeque('azul'):
                                validos.append(dest)
                        if validos:
                            selecionado = (x, y, validos)
                            # Destaque a peça selecionada e os movimentos possíveis
                            jogo.desenhar_tabuleiro()
                            jogo.desenhar_info()
                            pygame.draw.rect(tela, AMARELO, (x * TAMANHO_QUADRADO, y * TAMANHO_QUADRADO, TAMANHO_QUADRADO, TAMANHO_QUADRADO), 3)
                            for mov in validos:
                                pygame.draw.circle(tela, VERDE, (mov[0] * TAMANHO_QUADRADO + TAMANHO_QUADRADO // 2,
                                                                  mov[1] * TAMANHO_QUADRADO + TAMANHO_QUADRADO // 2), 10)
                            pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()