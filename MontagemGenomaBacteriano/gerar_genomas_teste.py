"""
Gerador de genomas sintéticos de diferentes bactérias para teste.
"""

import random

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

# Genomas sintéticos de diferentes bactérias
genomas_bacterianos = {
    "ecoli": {
        "tamanho": 5000,  # 5kb (simplificado)
        "gc": 50,
        "cobertura": 100,
        "descricao": "E. coli (gram-negativa, GC 50%)"
    },
    "bacillus": {
        "tamanho": 4200,
        "gc": 44,
        "cobertura": 100,
        "descricao": "Bacillus subtilis (gram-positiva, GC 44%)"
    },
    "staph": {
        "tamanho": 2800,
        "gc": 33,
        "cobertura": 150,
        "descricao": "Staphylococcus aureus (gram-positiva, GC 33%)"
    },
    "pseudomonas": {
        "tamanho": 6500,
        "gc": 66,
        "cobertura": 80,
        "descricao": "Pseudomonas aeruginosa (gram-negativa, GC 66%)"
    },
    "mycobacterium": {
        "tamanho": 4400,
        "gc": 65,
        "cobertura": 100,
        "descricao": "Mycobacterium tuberculosis (GC 65%)"
    }
}

print("Gerando genomas sintéticos de bactérias...\n")

for nome, params in genomas_bacterianos.items():
    print(f"Gerando {params['descricao']}...")
    
    # Gera genoma
    genoma = gerar_sequencia_com_gc(params['tamanho'], params['gc'])
    
    # Calcula número de reads para atingir cobertura desejada
    tamanho_read = 100
    num_reads = int((params['tamanho'] * params['cobertura']) / tamanho_read)
    
    # Gera reads
    reads = gerar_reads_de_genoma(genoma, num_reads, tamanho_read)
    
    # Salva
    arquivo = f"genoma_{nome}.fastq"
    salvar_fastq(reads, arquivo)
    
    print(f"  ✅ {arquivo}")
    print(f"     Tamanho: {params['tamanho']} bp")
    print(f"     GC: {params['gc']}%")
    print(f"     Reads: {num_reads}")
    print(f"     Cobertura: {params['cobertura']}x\n")

print("="*60)
print("Genomas sintéticos criados com sucesso!")
print("\nPara testar, execute:")
print("  python main.py genoma_ecoli.fastq")
print("  python app_gui.py  # e selecione um dos genomas")
