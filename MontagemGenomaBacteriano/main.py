import sys
import os
from config import *
from dados.leitor_fastq import LeitorFASTQ
from dados.pre_processamento import PreProcessador
from nucleo.grafo_bruijn import GrafoBruijn
from nucleo.montador import Montador
from estatistica.distribuicao import ModeloCobertura
from estatistica.metricas import MetricasMontagem
from visualizacao.plots import Visualizador
from identificacao.identificador import IdentificadorBacteriano

def main():
    """
    Função principal do Montador de Genoma Bacteriano.
    """
    print("=== Montador de Genoma Bacteriano ===")
    print(f"Configuração: K-mer={TAMANHO_KMER}, Cobertura Mínima={COBERTURA_MINIMA}")

    # Caminho do arquivo de entrada (pode ser passado por argumento ou hardcoded para teste)
    arquivo_fastq = "exemplo.fastq" 
    if len(sys.argv) > 1:
        arquivo_fastq = sys.argv[1]

    if not os.path.exists(arquivo_fastq):
        print(f"Erro: Arquivo '{arquivo_fastq}' não encontrado.")
        print("Por favor, forneça um arquivo FASTQ válido.")
        return

    # 1. Leitura e Pré-processamento
    print("\n--- 1. Leitura e Pré-processamento ---")
    leitor = LeitorFASTQ(arquivo_fastq)
    reads_processados = PreProcessador.processar_reads(leitor)
    print(f"Total de reads processados e aprovados: {len(reads_processados)}")

    if not reads_processados:
        print("Nenhum read passou no controle de qualidade. Abortando.")
        return

    # 2. Construção do Grafo
    print("\n--- 2. Construção do Grafo de Bruijn ---")
    grafo_bruijn = GrafoBruijn(k=TAMANHO_KMER)
    grafo_bruijn.construir_de_reads(reads_processados)
    
    # 3. Análise Estatística Preliminar
    print("\n--- 3. Análise Estatística ---")
    coberturas = [d['cobertura'] for _, _, d in grafo_bruijn.grafo.edges(data=True)]
    lambda_est = ModeloCobertura.estimar_lambda_poisson(coberturas)
    print(f"Cobertura média estimada (Lambda): {lambda_est:.2f}")
    
    Visualizador.plotar_distribuicao_cobertura(coberturas, "distribuicao_cobertura.png")

    # 4. Simplificação do Grafo
    print("\n--- 4. Simplificação do Grafo ---")
    grafo_bruijn.remover_erros(cobertura_minima=COBERTURA_MINIMA)
    # grafo_bruijn.simplificar_bolhas() # Implementação futura

    # 5. Montagem
    print("\n--- 5. Montagem de Contigs ---")
    montador = Montador(grafo_bruijn)
    contigs = montador.encontrar_caminhos_nao_ramificados()
    print(f"Total de contigs gerados: {len(contigs)}")

    # 6. Métricas Finais
    print("\n--- 6. Métricas da Montagem ---")
    n50 = MetricasMontagem.calcular_n50(contigs)
    l50 = MetricasMontagem.calcular_l50(contigs)
    maior_contig = max([len(c) for c in contigs]) if contigs else 0
    
    print(f"N50: {n50}")
    print(f"L50: {l50}")
    print(f"Maior Contig: {maior_contig}")
    
    # Salvar contigs
    with open("contigs.fasta", "w") as f:
        for i, contig in enumerate(contigs):
            f.write(f">contig_{i+1}_len_{len(contig)}\n")
            f.write(f"{contig}\n")
    print("Contigs salvos em 'contigs.fasta'.")
    
    # 7. Identificação Bacteriana
    print("\n--- 7. Identificação Bacteriana ---")
    tamanho_total = sum(len(c) for c in contigs)
    gc_valores = [MetricasMontagem.conteudo_gc(c) for c in contigs if c]
    gc_medio = sum(gc_valores) / len(gc_valores) if gc_valores else 0
    
    identificador = IdentificadorBacteriano()
    relatorio = identificador.gerar_relatorio(tamanho_total, gc_medio)
    print(relatorio)
    
    # Salvar relatório
    with open("identificacao_bacteriana.txt", "w") as f:
        f.write(relatorio)
    print("\nRelatório de identificação salvo em 'identificacao_bacteriana.txt'.")

    print("\nProcesso finalizado com sucesso.")

if __name__ == "__main__":
    main()
