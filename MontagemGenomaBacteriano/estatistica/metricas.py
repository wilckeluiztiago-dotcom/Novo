import numpy as np

class MetricasMontagem:
    """
    Cálculo de métricas de qualidade de montagem.
    """
    
    @staticmethod
    def calcular_n50(contigs):
        """
        Calcula o N50 dos contigs.
        N50 é o comprimento do menor contig tal que a soma dos contigs
        de comprimento maior ou igual a ele é pelo menos 50% do tamanho total da montagem.
        """
        if not contigs:
            return 0
            
        comprimentos = sorted([len(c) for c in contigs], reverse=True)
        tamanho_total = sum(comprimentos)
        soma_acumulada = 0
        
        for comp in comprimentos:
            soma_acumulada += comp
            if soma_acumulada >= tamanho_total / 2:
                return comp
        return 0

    @staticmethod
    def calcular_l50(contigs):
        """
        Calcula o L50: número de contigs necessários para atingir o N50.
        """
        if not contigs:
            return 0
            
        comprimentos = sorted([len(c) for c in contigs], reverse=True)
        tamanho_total = sum(comprimentos)
        soma_acumulada = 0
        
        for i, comp in enumerate(comprimentos):
            soma_acumulada += comp
            if soma_acumulada >= tamanho_total / 2:
                return i + 1
        return 0

    @staticmethod
    def conteudo_gc(sequencia):
        """
        Calcula a porcentagem de bases G e C na sequência.
        """
        if not sequencia:
            return 0.0
        g = sequencia.count('G')
        c = sequencia.count('C')
        return (g + c) / len(sequencia) * 100.0
