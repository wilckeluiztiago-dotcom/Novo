import networkx as nx
from nucleo.kmer import KmerUtils
from config import COBERTURA_MINIMA

class GrafoBruijn:
    """
    Implementação de um Grafo de Bruijn para montagem de genomas.
    """
    def __init__(self, k):
        self.k = k
        self.grafo = nx.DiGraph()

    def construir_de_reads(self, reads):
        """
        Constrói o grafo a partir de uma lista de reads.
        Cada k-mer (k-1)-mer -> (k-1)-mer é uma aresta.
        """
        print(f"Construindo grafo com k={self.k}...")
        for _, seq, _ in reads:
            for kmer in KmerUtils.gerar_kmers(seq, self.k):
                kmer_can = KmerUtils.kmer_canonico(kmer)
                
                prefixo = kmer_can[:-1]
                sufixo = kmer_can[1:]
                
                # Adiciona aresta (cria nós se não existirem)
                if self.grafo.has_edge(prefixo, sufixo):
                    self.grafo[prefixo][sufixo]['cobertura'] += 1
                else:
                    self.grafo.add_edge(prefixo, sufixo, cobertura=1)
        
        print(f"Grafo construído: {self.grafo.number_of_nodes()} nós, {self.grafo.number_of_edges()} arestas.")

    def remover_erros(self, cobertura_minima=COBERTURA_MINIMA):
        """
        Remove arestas com cobertura abaixo do limiar (prováveis erros de sequenciamento).
        """
        print(f"Removendo arestas com cobertura < {cobertura_minima}...")
        arestas_para_remover = []
        for u, v, dados in self.grafo.edges(data=True):
            if dados['cobertura'] < cobertura_minima:
                arestas_para_remover.append((u, v))
        
        self.grafo.remove_edges_from(arestas_para_remover)
        
        # Remove nós isolados
        nos_isolados = list(nx.isolates(self.grafo))
        self.grafo.remove_nodes_from(nos_isolados)
        
        print(f"Grafo após limpeza: {self.grafo.number_of_nodes()} nós, {self.grafo.number_of_edges()} arestas.")

    def simplificar_bolhas(self):
        """
        Simplifica bolhas no grafo (caminhos alternativos curtos causados por SNPs ou erros).
        (Implementação simplificada para fins demonstrativos)
        """
        # TODO: Implementar algoritmo de detecção e colapso de bolhas
        pass

    def obter_estatisticas(self):
        """
        Retorna estatísticas básicas do grafo.
        """
        return {
            "num_nos": self.grafo.number_of_nodes(),
            "num_arestas": self.grafo.number_of_edges(),
            "densidade": nx.density(self.grafo)
        }
