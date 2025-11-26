"""
Script para gerar genomas FASTQ das 20 novas bactérias adicionadas.
Autor: Luiz Tiago Wilcke
Data: 2025-11-25
"""

import random
from identificacao.banco_expandido import BANCO_GENOMAS_EXPANDIDO

def gerar_sequencia_com_gc(tamanho, gc_percentual):
    """Gera sequência de DNA com GC% específico."""
    num_gc = int(tamanho * gc_percentual / 100)
    num_at = tamanho - num_gc
    
    bases = ['G', 'C'] * (num_gc // 2) + ['A', 'T'] * (num_at // 2)
    random.shuffle(bases)
    
    return ''.join(bases[:tamanho])

def gerar_reads_de_genoma(genoma, num_reads, tamanho_read, taxa_erro=0.01):
    """Gera reads a partir de um genoma."""
    reads = []
    tamanho_genoma = len(genoma)
    
    for i in range(num_reads):
        pos = random.randint(0, tamanho_genoma - tamanho_read)
        read = genoma[pos:pos + tamanho_read]
        
        # Adiciona erros
        read_list = list(read)
        for j in range(len(read_list)):
            if random.random() < taxa_erro:
                read_list[j] = random.choice(['A', 'C', 'G', 'T'])
        read = ''.join(read_list)
        
        qualidade = ''.join(chr(33 + min(40, max(20, int(random.gauss(30, 5))))) 
                           for _ in range(tamanho_read))
        reads.append((f"@read_{i+1}", read, qualidade))
    
    return reads

def salvar_fastq(reads, nome_arquivo):
    """Salva reads em formato FASTQ."""
    with open(nome_arquivo, "w") as f:
        for cabecalho, seq, qual in reads:
            f.write(f"{cabecalho}\n{seq}\n+\n{qual}\n")

# Novas bactérias para gerar genomas (as 20 adicionadas)
novas_bacterias = [
    "Aeromonas hydrophila",
    "Edwardsiella tarda",
    "Elizabethkingia meningoseptica",
    "Comamonas testosteroni",
    "Kluyvera ascorbata",
    "Ochrobactrum anthropi",
    "Weeksella virosa",
    "Roseomonas gilardii",
    "Methylobacterium mesophilicum",
    "Brevundimonas diminuta",
    "Sphingobacterium multivorum",
    "Cupriavidus pauculus",
    "Tsukamurella paurometabola",
    "Nocardia asteroides",
    "Gordonia bronchialis",
    "Rhodococcus equi",
    "Leuconostoc mesenteroides",
    "Pediococcus acidilactici",
    "Aerococcus viridans",
    "Micrococcus luteus"
]

print("="*70)
print("  GERAÇÃO DE GENOMAS FASTQ - 20 NOVAS BACTÉRIAS")
print("="*70)
print()

genomas_gerados = 0

for bacteria_info in BANCO_GENOMAS_EXPANDIDO:
    nome = bacteria_info['nome']
    
    # Gerar apenas para as novas bactérias
    if nome not in novas_bacterias:
        continue
    
    print(f"Gerando genoma: {nome}")
    
    # Obter parâmetros da bactéria
    tamanho_min, tamanho_max = bacteria_info['tamanho_genoma']
    gc_min, gc_max = bacteria_info['conteudo_gc']
    
    tamanho = int((tamanho_min + tamanho_max) / 2)
    gc = (gc_min + gc_max) / 2
    cobertura = 100  # 100x cobertura
    
    # Gerar genoma
    genoma = gerar_sequencia_com_gc(tamanho, gc)
    
    # Calcular número de reads
    tamanho_read = 100
    num_reads = int((tamanho * cobertura) / tamanho_read)
    
    # Gerar reads
    reads = gerar_reads_de_genoma(genoma, num_reads, tamanho_read)
    
    # Nome do arquivo (normalizar nome)
    nome_arquivo = nome.lower().replace(" ", "_").replace(".", "")
    arquivo = f"genoma_{nome_arquivo}.fastq"
    
    # Salvar
    salvar_fastq(reads, arquivo)
    
    print(f"  ✅ {arquivo}")
    print(f"     Tamanho: {tamanho:,} bp")
    print(f"     GC: {gc:.1f}%")
    print(f"     Reads: {num_reads:,}")
    print(f"     Cobertura: {cobertura}x")
    print()
    
    genomas_gerados += 1

print("="*70)
print(f"✅ GERAÇÃO CONCLUÍDA: {genomas_gerados} genomas FASTQ criados!")
print("="*70)
print()
print("Arquivos disponíveis:")
print("  - Use 'ls genoma_*.fastq' para listar todos")
print("  - Carregue na GUI com: python app_gui.py")
print()
