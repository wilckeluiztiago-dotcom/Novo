"""
Módulo de análise genômica avançada: detecção de ORFs e anotação de genes.
"""

class AnalisadorGenomica:
    """
    Realiza análises genômicas avançadas nos contigs montados.
    """
    
    # Códons de início e parada
    CODONS_INICIO = ['ATG', 'GTG', 'TTG']
    CODONS_PARADA = ['TAA', 'TAG', 'TGA']
    
    def __init__(self):
        pass
    
    @staticmethod
    def encontrar_orfs(sequencia, tamanho_minimo=100):
        """
        Encontra Open Reading Frames (ORFs) na sequência.
        Retorna lista de ORFs com posição, tamanho e frame.
        """
        orfs = []
        
        # Procura em 3 frames de leitura (fita direta)
        for frame in range(3):
            i = frame
            while i < len(sequencia) - 2:
                codon = sequencia[i:i+3]
                
                # Encontrou códon de início
                if codon in AnalisadorGenomica.CODONS_INICIO:
                    inicio = i
                    j = i + 3
                    
                    # Procura códon de parada
                    while j < len(sequencia) - 2:
                        codon_atual = sequencia[j:j+3]
                        if codon_atual in AnalisadorGenomica.CODONS_PARADA:
                            tamanho = j - inicio + 3
                            if tamanho >= tamanho_minimo:
                                orfs.append({
                                    'inicio': inicio,
                                    'fim': j + 3,
                                    'tamanho': tamanho,
                                    'frame': frame + 1,
                                    'fita': '+',
                                    'sequencia': sequencia[inicio:j+3]
                                })
                            break
                        j += 3
                    i = j
                else:
                    i += 3
        
        # Procura na fita reversa complementar
        seq_reversa = AnalisadorGenomica.reverso_complementar(sequencia)
        for frame in range(3):
            i = frame
            while i < len(seq_reversa) - 2:
                codon = seq_reversa[i:i+3]
                
                if codon in AnalisadorGenomica.CODONS_INICIO:
                    inicio = i
                    j = i + 3
                    
                    while j < len(seq_reversa) - 2:
                        codon_atual = seq_reversa[j:j+3]
                        if codon_atual in AnalisadorGenomica.CODONS_PARADA:
                            tamanho = j - inicio + 3
                            if tamanho >= tamanho_minimo:
                                # Converte posição para coordenadas da fita original
                                pos_original_fim = len(sequencia) - inicio
                                pos_original_inicio = len(sequencia) - (j + 3)
                                
                                orfs.append({
                                    'inicio': pos_original_inicio,
                                    'fim': pos_original_fim,
                                    'tamanho': tamanho,
                                    'frame': -(frame + 1),
                                    'fita': '-',
                                    'sequencia': seq_reversa[inicio:j+3]
                                })
                            break
                        j += 3
                    i = j
                else:
                    i += 3
        
        # Ordena por posição
        orfs.sort(key=lambda x: x['inicio'])
        return orfs
    
    @staticmethod
    def reverso_complementar(sequencia):
        """Retorna o reverso complementar de uma sequência."""
        complemento = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
        return ''.join(complemento.get(base, 'N') for base in reversed(sequencia))
    
    @staticmethod
    def calcular_densidade_codificante(sequencia, orfs):
        """
        Calcula a densidade de regiões codificantes (% do genoma que é ORF).
        """
        if not sequencia:
            return 0.0
        
        bases_codificantes = sum(orf['tamanho'] for orf in orfs)
        return (bases_codificantes / len(sequencia)) * 100
    
    @staticmethod
    def prever_genes_funcionais(orfs):
        """
        Classifica ORFs em possíveis genes funcionais baseado em tamanho.
        """
        genes = []
        for orf in orfs:
            # Heurística simples: ORFs > 300bp têm maior chance de serem genes
            if orf['tamanho'] >= 300:
                genes.append({
                    **orf,
                    'tipo': 'gene_provavel',
                    'confianca': 'alta' if orf['tamanho'] >= 600 else 'media'
                })
            elif orf['tamanho'] >= 150:
                genes.append({
                    **orf,
                    'tipo': 'gene_possivel',
                    'confianca': 'baixa'
                })
        
        return genes
    
    @staticmethod
    def gerar_relatorio_genomico(contigs):
        """
        Gera relatório completo de análise genômica.
        """
        relatorio = "=" * 80 + "\n"
        relatorio += "ANÁLISE GENÔMICA AVANÇADA\n"
        relatorio += "=" * 80 + "\n\n"
        
        total_orfs = 0
        total_genes = 0
        tamanho_total = 0
        
        for i, contig in enumerate(contigs[:10], 1):  # Analisa primeiros 10 contigs
            if len(contig) < 100:
                continue
                
            orfs = AnalisadorGenomica.encontrar_orfs(contig, tamanho_minimo=100)
            genes = AnalisadorGenomica.prever_genes_funcionais(orfs)
            densidade = AnalisadorGenomica.calcular_densidade_codificante(contig, orfs)
            
            tamanho_total += len(contig)
            total_orfs += len(orfs)
            total_genes += len(genes)
            
            relatorio += f"Contig {i} ({len(contig)} bp):\n"
            relatorio += f"  • ORFs encontrados: {len(orfs)}\n"
            relatorio += f"  • Genes prováveis: {len(genes)}\n"
            relatorio += f"  • Densidade codificante: {densidade:.1f}%\n"
            
            if genes:
                relatorio += f"  • Genes principais:\n"
                for j, gene in enumerate(genes[:3], 1):
                    relatorio += f"    {j}. Posição {gene['inicio']}-{gene['fim']} "
                    relatorio += f"({gene['tamanho']} bp, fita {gene['fita']}, "
                    relatorio += f"confiança: {gene['confianca']})\n"
            relatorio += "\n"
        
        relatorio += "-" * 80 + "\n"
        relatorio += f"RESUMO GERAL:\n"
        relatorio += f"  • Total de ORFs: {total_orfs}\n"
        relatorio += f"  • Total de genes prováveis: {total_genes}\n"
        relatorio += f"  • Tamanho analisado: {tamanho_total:,} bp\n"
        relatorio += "=" * 80 + "\n"
        
        return relatorio, total_orfs, total_genes
