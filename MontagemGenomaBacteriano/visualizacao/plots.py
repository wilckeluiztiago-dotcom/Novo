import matplotlib.pyplot as plt
import networkx as nx

class Visualizador:
    """
    Classe para geração de gráficos e visualizações.
    """
    
    @staticmethod
    def plotar_distribuicao_cobertura(coberturas, arquivo_saida="distribuicao_cobertura.png"):
        """
        Plota o histograma da distribuição de cobertura dos k-mers.
        """
        if not coberturas:
            print("Sem dados de cobertura para plotar.")
            return

        plt.figure(figsize=(10, 6))
        plt.hist(coberturas, bins=50, color='skyblue', edgecolor='black')
        plt.title("Distribuição de Cobertura de K-mers")
        plt.xlabel("Cobertura")
        plt.ylabel("Frequência")
        plt.grid(True, alpha=0.3)
        plt.savefig(arquivo_saida)
        print(f"Gráfico de cobertura salvo em: {arquivo_saida}")
        plt.close()

    @staticmethod
    def visualizar_grafo_simplificado(grafo, arquivo_saida="grafo_simplificado.png"):
        """
        Gera uma visualização simplificada do grafo (apenas para grafos pequenos).
        """
        if grafo.number_of_nodes() > 1000:
            print("Grafo muito grande para visualização completa. Pulando plotagem do grafo.")
            return

        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(grafo, k=0.15, iterations=20)
        nx.draw(grafo, pos, node_size=20, node_color="blue", alpha=0.5, with_labels=False)
        plt.title("Visualização do Grafo de Bruijn (Simplificado)")
        plt.savefig(arquivo_saida)
        print(f"Imagem do grafo salva em: {arquivo_saida}")
        plt.close()
