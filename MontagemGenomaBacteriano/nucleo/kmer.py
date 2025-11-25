class KmerUtils:
    """
    Utilitários para manipulação de k-mers e sequências de DNA.
    """
    
    COMPLEMENTO = str.maketrans("ACGTN", "TGCAN")

    @staticmethod
    def reverso_complementar(sequencia):
        """
        Retorna o reverso complementar de uma sequência de DNA.
        """
        return sequencia.translate(KmerUtils.COMPLEMENTO)[::-1]

    @staticmethod
    def kmer_canonico(kmer):
        """
        Retorna o k-mer canônico (o menor lexicograficamente entre o k-mer e seu reverso complementar).
        Isso é importante porque o sequenciamento lê ambas as fitas.
        """
        rc = KmerUtils.reverso_complementar(kmer)
        if kmer < rc:
            return kmer
        return rc

    @staticmethod
    def gerar_kmers(sequencia, k):
        """
        Gera todos os k-mers de uma sequência.
        """
        n = len(sequencia)
        if n < k:
            return
        for i in range(n - k + 1):
            yield sequencia[i:i+k]
