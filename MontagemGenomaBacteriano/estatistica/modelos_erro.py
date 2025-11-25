import math

class ModeloErro:
    """
    Modelos para probabilidade de erro de sequenciamento.
    """
    
    @staticmethod
    def phred_para_probabilidade(q):
        """
        Converte score Phred para probabilidade de erro.
        P = 10^(-Q/10)
        """
        return 10 ** (-q / 10.0)

    @staticmethod
    def probabilidade_kmer_correto(qualidades):
        """
        Calcula a probabilidade de um k-mer inteiro estar correto,
        dado os scores de qualidade de suas bases.
        P(kmer_correto) = Product(1 - P_erro_i)
        """
        prob_correto = 1.0
        for q in qualidades:
            p_erro = ModeloErro.phred_para_probabilidade(q)
            prob_correto *= (1.0 - p_erro)
        return prob_correto

    @staticmethod
    def kmer_confiavel(qualidades, limiar_confianca=0.90):
        """
        Verifica se a probabilidade do k-mer estar correto é superior ao limiar.
        """
        return ModeloErro.probabilidade_kmer_correto(qualidades) >= limiar_confianca
