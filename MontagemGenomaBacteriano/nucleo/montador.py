import networkx as nx

class Montador:
    """
    Classe responsável por montar os contigs a partir do Grafo de Bruijn.
    """
    def __init__(self, grafo_bruijn):
        self.grafo_bruijn = grafo_bruijn
        self.grafo = grafo_bruijn.grafo
        self.k = grafo_bruijn.k

    def encontrar_caminhos_nao_ramificados(self):
        """
        Encontra caminhos maximais não ramificados no grafo.
        Estes caminhos correspondem aos contigs.
        """
        contigs = []
        
        # Identifica nós que são início de caminhos (in-degree != 1 ou out-degree != 1)
        # Para simplificar, vamos iterar sobre todos os nós e encontrar caminhos simples
        
        # Uma abordagem comum é encontrar todos os caminhos simples maximal
        # Aqui usamos uma heurística simplificada:
        # 1. Encontrar nós com grau de entrada != 1 ou grau de saída != 1 (nós de ramificação)
        # 2. Iniciar caminhos a partir desses nós
        
        # Para um grafo de Bruijn, contigs são caminhos onde in_degree == 1 e out_degree == 1
        # exceto nas pontas.
        
        visitados = set()
        
        # Primeiro, processa nós de ramificação
        nos_ramificacao = [n for n in self.grafo.nodes() if self.grafo.in_degree(n) != 1 or self.grafo.out_degree(n) != 1]
        
        for no_inicial in nos_ramificacao:
            for vizinho in self.grafo.successors(no_inicial):
                caminho = [no_inicial, vizinho]
                atual = vizinho
                
                while self.grafo.in_degree(atual) == 1 and self.grafo.out_degree(atual) == 1:
                    sucessores = list(self.grafo.successors(atual))
                    if not sucessores:
                        break
                    proximo = sucessores[0]
                    caminho.append(proximo)
                    atual = proximo
                    if atual in visitados: # Evitar ciclos infinitos simples
                        break
                    visitados.add(atual)
                
                contig = self.reconstruir_sequencia(caminho)
                contigs.append(contig)
                
        # TODO: Tratar ciclos isolados que não passam por nós de ramificação
        
        return contigs

    def reconstruir_sequencia(self, caminho_kmers):
        """
        Reconstrói a sequência de DNA a partir de uma lista de k-mers (nós do grafo).
        A sequência é o primeiro k-mer + a última base de cada k-mer subsequente.
        """
        if not caminho_kmers:
            return ""
            
        sequencia = caminho_kmers[0]
        for kmer in caminho_kmers[1:]:
            sequencia += kmer[-1]
            
        return sequencia
