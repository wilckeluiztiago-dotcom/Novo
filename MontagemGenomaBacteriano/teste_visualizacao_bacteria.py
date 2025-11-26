"""
Script de teste para visualização detalhada de bactéria.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualizacao.bacteria_detalhada import VisualizadorBacteriaAvancado
from analise.genomica import AnalisadorGenomica
from identificacao.banco_expandido import BACTERIAS_EXPANDIDO

def teste_visualizacao():
    """Testa visualização detalhada de diferentes bactérias."""
    
    # Criar contigs e ORFs de exemplo
    contigs = [
        "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG" * 20,
        "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAG" * 15,
        "TACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT" * 10
    ]
    
    # Analisar ORFs
    orfs_por_contig = []
    for contig in contigs:
        orfs = AnalisadorGenomica.encontrar_orfs(contig, tamanho_minimo=50)
        orfs_por_contig.append(orfs)
    
    # Testar diferentes formas de bactérias
    formas_teste = ['bacilo', 'coco', 'espiral']
    
    for forma in formas_teste:
        # Encontrar bactéria com essa forma
        bacteria = next((b for b in BACTERIAS_EXPANDIDO if b.get('forma') == forma), None)
        
        if bacteria:
            print(f"\n🦠 Gerando visualização: {bacteria['nome']} ({forma})")
            
            # Preparar informações
            tamanho_total = sum(len(c) for c in contigs)
            gc_valores = [sum(1 for base in c if base in 'GC') / len(c) * 100 for c in contigs if c]
            gc_medio = sum(gc_valores) / len(gc_valores) if gc_valores else 0
            
            bacteria_info = {
                'nome': bacteria['nome'],
                'forma': bacteria.get('forma', 'bacilo'),
                'gram': bacteria.get('gram', 'negativa'),
                'tamanho_genoma': tamanho_total,
                'gc': gc_medio,
                'patogenicidade': bacteria.get('patogenicidade', 'Desconhecida'),
                'aplicacoes': bacteria.get('aplicacoes', 'N/A')
            }
            
            # Criar visualização
            visualizador = VisualizadorBacteriaAvancado()
            arquivo = f"bacteria_{forma}.png"
            visualizador.criar_visualizacao(bacteria_info, contigs, orfs_por_contig, arquivo)
            
            print(f"✅ Salvo em: {arquivo}")
    
    print("\n" + "="*60)
    print("✅ Visualizações criadas com sucesso!")
    print("\nArquivos gerados:")
    print("  - bacteria_bacilo.png")
    print("  - bacteria_coco.png")
    print("  - bacteria_espiral.png")
    print("="*60)

if __name__ == "__main__":
    teste_visualizacao()
