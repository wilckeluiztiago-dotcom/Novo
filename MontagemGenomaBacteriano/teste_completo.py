"""
Script de teste completo com análise genômica avançada e visualização.
"""

import sys
from dados.leitor_fastq import LeitorFASTQ
from dados.pre_processamento import PreProcessador
from nucleo.grafo_bruijn import GrafoBruijn
from nucleo.montador import Montador
from estatistica.distribuicao import ModeloCobertura
from estatistica.metricas import MetricasMontagem
from identificacao.identificador import IdentificadorBacteriano
from analise.genomica import AnalisadorGenomica
from visualizacao.genoma_circular import VisualizadorGenoma
from config import *

def main():
    print("=== MONTADOR DE GENOMA BACTERIANO - VERSÃO AVANÇADA ===\n")
    
    arquivo_fastq = "exemplo.fastq"
    if len(sys.argv) > 1:
        arquivo_fastq = sys.argv[1]
    
    # 1-5: Pipeline padrão (leitura, grafo, montagem)
    print("Executando pipeline de montagem...")
    leitor = LeitorFASTQ(arquivo_fastq)
    reads_processados = PreProcessador.processar_reads(leitor)
    
    grafo = GrafoBruijn(k=TAMANHO_KMER)
    grafo.construir_de_reads(reads_processados)
    grafo.remover_erros(cobertura_minima=COBERTURA_MINIMA)
    
    montador = Montador(grafo)
    contigs = montador.encontrar_caminhos_nao_ramificados()
    
    print(f"✅ {len(contigs)} contigs montados\n")
    
    # 6: Análise genômica avançada
    print("--- ANÁLISE GENÔMICA AVANÇADA ---")
    orfs_por_contig = []
    for contig in contigs[:10]:  # Analisa primeiros 10
        orfs = AnalisadorGenomica.encontrar_orfs(contig, tamanho_minimo=100)
        orfs_por_contig.append(orfs)
    
    relatorio_genomico, total_orfs, total_genes = AnalisadorGenomica.gerar_relatorio_genomico(contigs)
    print(relatorio_genomico)
    
    # Salva relatório
    with open("analise_genomica.txt", "w") as f:
        f.write(relatorio_genomico)
    
    # 7: Identificação bacteriana
    print("\n--- IDENTIFICAÇÃO BACTERIANA ---")
    tamanho_total = sum(len(c) for c in contigs)
    gc_valores = [MetricasMontagem.conteudo_gc(c) for c in contigs if c]
    gc_medio = sum(gc_valores) / len(gc_valores) if gc_valores else 0
    
    identificador = IdentificadorBacteriano()
    print(f"Banco de dados: {len(identificador.banco)} espécies bacterianas")
    candidatos = identificador.identificar(tamanho_total, gc_medio, top_n=10)
    
    print(f"\nTop 10 candidatos:")
    for i, cand in enumerate(candidatos, 1):
        bact = cand['bacteria']
        print(f"{i}. {bact['nome']} - Score: {cand['score']:.1f}%")
    
    # 8: Visualização circular do genoma
    print("\n--- VISUALIZAÇÃO DO GENOMA ---")
    try:
        visualizador = VisualizadorGenoma()
        arquivo_viz = visualizador.criar_visualizacao(contigs[:5], orfs_por_contig[:5], 
                                                       "genoma_circular.png")
        print(f"✅ Visualização criada: {arquivo_viz}")
    except Exception as e:
        print(f"⚠️ Erro na visualização: {e}")
    
    print("\n" + "="*80)
    print("RESUMO FINAL:")
    print(f"  • Contigs: {len(contigs)}")
    print(f"  • ORFs detectados: {total_orfs}")
    print(f"  • Genes prováveis: {total_genes}")
    print(f"  • Tamanho total: {tamanho_total:,} bp")
    print(f"  • GC médio: {gc_medio:.1f}%")
    print(f"  • Melhor candidato: {candidatos[0]['bacteria']['nome']} ({candidatos[0]['score']:.1f}%)")
    print("="*80)
    
    print("\n✅ Análise completa finalizada!")

if __name__ == "__main__":
    main()
