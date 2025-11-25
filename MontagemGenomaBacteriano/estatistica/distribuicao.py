import numpy as np
from scipy.stats import poisson, norm

class ModeloCobertura:
    """
    Modelos estatísticos para distribuição de cobertura de k-mers.
    """
    
    @staticmethod
    def estimar_lambda_poisson(coberturas):
        """
        Estima o parâmetro lambda da distribuição de Poisson a partir das coberturas observadas.
        Lambda é a média da cobertura.
        """
        if not coberturas:
            return 0
        return np.mean(coberturas)

    @staticmethod
    def probabilidade_poisson(k, lam):
        """
        Calcula P(X=k) para uma distribuição de Poisson com média lambda.
        """
        return poisson.pmf(k, lam)

    @staticmethod
    def intervalo_confianca_poisson(lam, confianca=0.95):
        """
        Retorna o intervalo de confiança para a cobertura esperada.
        """
        return poisson.interval(confianca, lam)

    @staticmethod
    def classificar_kmer(cobertura, lam, limiar_erro=0.01):
        """
        Classifica um k-mer como 'Erro', 'Único' ou 'Repetitivo' baseado na probabilidade.
        """
        prob = ModeloCobertura.probabilidade_poisson(cobertura, lam)
        
        # Se a cobertura é muito baixa e a probabilidade é baixa -> Erro
        if cobertura < lam and prob < limiar_erro:
            return "Erro"
        
        # Se a cobertura é muito alta (ex: 2*lambda) -> Repetitivo
        if cobertura > 1.8 * lam: # Heurística simples
            return "Repetitivo"
            
        return "Único"
