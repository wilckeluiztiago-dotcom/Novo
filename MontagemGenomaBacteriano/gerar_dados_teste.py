import random

def gerar_sequencia_aleatoria(tamanho):
    """Gera uma sequência de DNA aleatória."""
    bases = ['A', 'C', 'G', 'T']
    return ''.join(random.choice(bases) for _ in range(tamanho))

def gerar_qualidade(tamanho, qualidade_media=30):
    """Gera scores de qualidade Phred."""
    return ''.join(chr(33 + min(40, max(20, int(random.gauss(qualidade_media, 5))))) for _ in range(tamanho))

def gerar_reads_de_genoma(genoma, num_reads, tamanho_read):
    """Gera reads a partir de um genoma de referência."""
    reads = []
    tamanho_genoma = len(genoma)
    
    for i in range(num_reads):
        # Posição aleatória
        pos = random.randint(0, tamanho_genoma - tamanho_read)
        read = genoma[pos:pos + tamanho_read]
        
        # Adiciona alguns erros aleatórios (1% de erro)
        read_list = list(read)
        for j in range(len(read_list)):
            if random.random() < 0.01:
                read_list[j] = random.choice(['A', 'C', 'G', 'T'])
        read = ''.join(read_list)
        
        qualidade = gerar_qualidade(tamanho_read)
        reads.append((f"@read_{i+1}", read, qualidade))
    
    return reads

# Genoma sintético pequeno (500 bp)
genoma_referencia = gerar_sequencia_aleatoria(500)

# Gerar 1000 reads de 100 bp (cobertura ~200x)
reads = gerar_reads_de_genoma(genoma_referencia, 1000, 100)

# Salvar em FASTQ
with open("exemplo.fastq", "w") as f:
    for cabecalho, seq, qual in reads:
        f.write(f"{cabecalho}\n")
        f.write(f"{seq}\n")
        f.write("+\n")
        f.write(f"{qual}\n")

print("Arquivo 'exemplo.fastq' criado com sucesso!")
print(f"Genoma de referência: {len(genoma_referencia)} bp")
print(f"Número de reads: {len(reads)}")
print(f"Cobertura estimada: {len(reads) * 100 / len(genoma_referencia):.1f}x")
