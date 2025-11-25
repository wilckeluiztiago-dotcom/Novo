from config import QUALIDADE_MINIMA_PHRED, TAMANHO_MINIMO_READ

class PreProcessador:
    """
    Classe para pré-processamento de reads (QC e trimagem).
    """
    
    @staticmethod
    def calcular_phred(char_qualidade):
        """
        Converte caractere ASCII para score Phred (Offset 33).
        """
        return ord(char_qualidade) - 33

    @staticmethod
    def filtrar_por_qualidade_media(sequencia, qualidade, limiar=QUALIDADE_MINIMA_PHRED):
        """
        Verifica se a qualidade média do read é superior ao limiar.
        """
        scores = [PreProcessador.calcular_phred(q) for q in qualidade]
        media = sum(scores) / len(scores)
        return media >= limiar

    @staticmethod
    def trimar_pontas(sequencia, qualidade, limiar=QUALIDADE_MINIMA_PHRED):
        """
        Remove bases de baixa qualidade das pontas (algoritmo simples de janela deslizante ou ponta a ponta).
        Aqui implementamos uma trimagem simples das pontas (3' e 5').
        """
        scores = [PreProcessador.calcular_phred(q) for q in qualidade]
        
        inicio = 0
        fim = len(scores)
        
        # Trimar do início (5')
        while inicio < fim and scores[inicio] < limiar:
            inicio += 1
            
        # Trimar do fim (3')
        while fim > inicio and scores[fim-1] < limiar:
            fim -= 1
            
        if fim - inicio < TAMANHO_MINIMO_READ:
            return None, None
            
        return sequencia[inicio:fim], qualidade[inicio:fim]

    @staticmethod
    def processar_reads(leitor_fastq):
        """
        Processa todos os reads de um leitor, aplicando filtros e trimagem.
        """
        reads_processados = []
        
        for cabecalho, seq, qual in leitor_fastq.ler_reads():
            # 1. Trimar pontas
            res = PreProcessador.trimar_pontas(seq, qual)
            if res[0] is None:
                continue
            
            seq_trim, qual_trim = res
            
            # 2. Filtrar por qualidade média
            if PreProcessador.filtrar_por_qualidade_media(seq_trim, qual_trim):
                reads_processados.append((cabecalho, seq_trim, qual_trim))
                
        return reads_processados
