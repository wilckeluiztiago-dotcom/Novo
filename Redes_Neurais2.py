import numpy as np
import scipy.special
from scipy import ndimage
import math
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')

class RedeNeuralProfunda:
    def __init__(self, arquitetura, taxa_aprendizado=0.001, 
                 regularizacao_l2=0.01, taxa_dropout=0.3, 
                 momentum=0.9, taxa_decainento_peso=0.0001):
        
        self.arquitetura = arquitetura
        self.taxa_aprendizado = taxa_aprendizado
        self.regularizacao_l2 = regularizacao_l2
        self.taxa_dropout = taxa_dropout
        self.momentum = momentum
        self.taxa_decainento_peso = taxa_decainento_peso
        
        # Inicialização avançada de parâmetros
        self.pesos = self._inicializar_pesos_avancada()
        self.vieses = self._inicializar_vieses()
        
        # Otimizadores e estados avançados
        self.velocidade_pesos = [np.zeros_like(w) for w in self.pesos]
        self.velocidade_vieses = [np.zeros_like(b) for b in self.vieses]
        self.estado_quadrados_pesos = [np.zeros_like(w) for w in self.pesos]
        self.estado_quadrados_vieses = [np.zeros_like(b) for b in self.vieses]
        
        # Cache para treinamento
        self.ativacoes = []
        self.mascaras_dropout = []
        self.medias_lote = []
        self.variancias_lote = []
        
        # Histórico de métricas
        self.historico_loss = []
        self.historico_acuracia = []
        
        # Configurações avançadas
        self.epsilon = 1e-8
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.iteracao = 0

    def _inicializar_pesos_avancada(self):
        """Inicialização sofisticada usando He Normal e Orthogonal"""
        pesos = []
        for i in range(len(self.arquitetura)-1):
            if i < len(self.arquitetura)-2:  # Camadas ocultas
                # He initialization para ReLU/Swish
                std_dev = math.sqrt(2.0 / self.arquitetura[i])
                peso_camada = np.random.normal(0, std_dev, 
                                             (self.arquitetura[i+1], self.arquitetura[i]))
                
                # Adicionar inicialização ortogonal para melhor condicionamento
                if peso_camada.shape[0] >= peso_camada.shape[1]:
                    U, _, Vt = np.linalg.svd(peso_camada, full_matrices=False)
                    peso_camada = U @ Vt
                else:
                    U, _, Vt = np.linalg.svd(peso_camada.T, full_matrices=False)
                    peso_camada = (U @ Vt).T
                    
            else:  # Camada de saída
                # Xavier/Glorot initialization para camada final
                limite = math.sqrt(6.0 / (self.arquitetura[i] + self.arquitetura[i+1]))
                peso_camada = np.random.uniform(-limite, limite, 
                                              (self.arquitetura[i+1], self.arquitetura[i]))
            
            pesos.append(peso_camada)
        return pesos

    def _inicializar_vieses(self):
        """Inicialização de vieses com pequeno ruído"""
        vieses = []
        for i in range(len(self.arquitetura)-1):
            # Inicialização com pequeno ruído positivo para evitar "dead neurons"
            bias_camada = np.random.normal(0.01, 0.001, (self.arquitetura[i+1], 1))
            vieses.append(bias_camada)
        return vieses

    def _funcao_ativacao(self, x, tipo='swish'):
        """Seletor sofisticado de funções de ativação"""
        if tipo == 'swish':
            return x * scipy.special.expit(x)
        elif tipo == 'relu':
            return np.maximum(0, x)
        elif tipo == 'leaky_relu':
            return np.where(x > 0, x, 0.01 * x)
        elif tipo == 'elu':
            return np.where(x > 0, x, 1.0 * (np.exp(x) - 1))
        elif tipo == 'gelu':
            return 0.5 * x * (1 + scipy.special.erf(x / np.sqrt(2)))
        else:
            return scipy.special.expit(x)  # sigmoid como fallback

    def _derivada_ativacao(self, x, tipo='swish'):
        """Derivadas das funções de ativação"""
        if tipo == 'swish':
            sigmoid = scipy.special.expit(x)
            return sigmoid + x * sigmoid * (1 - sigmoid)
        elif tipo == 'relu':
            return np.where(x > 0, 1, 0)
        elif tipo == 'leaky_relu':
            return np.where(x > 0, 1, 0.01)
        elif tipo == 'elu':
            return np.where(x > 0, 1, 1.0 * np.exp(x))
        elif tipo == 'gelu':
            return 0.5 * (1 + scipy.special.erf(x / np.sqrt(2))) + \
                   (x * np.exp(-0.5 * x ** 2)) / (np.sqrt(2 * np.pi))
        else:
            sigmoid = scipy.special.expit(x)
            return sigmoid * (1 - sigmoid)

    def _normalizacao_lote_avancada(self, x, camada_idx, treinamento=True):
        """Batch normalization com suporte a modo de treinamento e inferência"""
        if treinamento:
            media = np.mean(x, axis=1, keepdims=True)
            variancia = np.var(x, axis=1, keepdims=True)
            
            # Salvar para backward pass
            if len(self.medias_lote) <= camada_idx:
                self.medias_lote.append(media)
                self.variancias_lote.append(variancia)
            else:
                self.medias_lote[camada_idx] = media
                self.variancias_lote[camada_idx] = variancia
            
            # Normalizar
            x_normalizado = (x - media) / np.sqrt(variancia + self.epsilon)
        else:
            # Em inferência, usar médias móveis (não implementado aqui por simplicidade)
            x_normalizado = x
            
        return x_normalizado

    def _dropout_avancado(self, x, taxa, treinamento=True):
        """Dropout avançado com suporte a diferentes modalidades"""
        if treinamento and taxa > 0:
            # Dropout gaussiano para treinamento mais suave
            mascara = np.random.normal(1.0, taxa, size=x.shape)
            return x * mascara
        return x

    def _propagacao_direta(self, entrada, treinamento=True, tipo_ativacao='swish'):
        """Propagação direta sofisticada com múltiplas funcionalidades"""
        self.ativacoes = [entrada]
        self.mascaras_dropout = []
        self.medias_lote = []
        self.variancias_lote = []
        
        ativacao_atual = entrada
        
        for i, (peso, vies) in enumerate(zip(self.pesos, self.vieses)):
            # Transformação linear
            z = np.dot(peso, ativacao_atual) + vies
            
            # Batch normalization (exceto na camada de saída)
            if i < len(self.pesos) - 1:
                z = self._normalizacao_lote_avancada(z, i, treinamento)
            
            # Aplicar função de ativação
            if i == len(self.pesos) - 1:
                # Softmax para camada de saída
                ativacao_atual = self._softmax_estavel(z)
            else:
                ativacao_atual = self._funcao_ativacao(z, tipo_ativacao)
                
                # Aplicar dropout durante treinamento
                if treinamento:
                    ativacao_atual = self._dropout_avancado(ativacao_atual, self.taxa_dropout)
                    mascara = (ativacao_atual != 0).astype(float)
                    self.mascaras_dropout.append(mascara)
            
            self.ativacoes.append(ativacao_atual)
        
        return ativacao_atual

    def _propagacao_reversa(self, saida_esperada, tipo_ativacao='swish'):
        """Backpropagation avançado com regularizações"""
        m = saida_esperada.shape[1]
        gradientes_pesos = [np.zeros_like(w) for w in self.pesos]
        gradientes_vieses = [np.zeros_like(b) for b in self.vieses]
        
        # Gradiente da camada de saída (softmax + cross-entropy)
        dz = self.ativacoes[-1] - saida_esperada
        
        # Calcular gradientes com regularização L2
        gradientes_pesos[-1] = (1/m) * np.dot(dz, self.ativacoes[-2].T) + \
                              (self.regularizacao_l2/m) * self.pesos[-1]
        gradientes_vieses[-1] = (1/m) * np.sum(dz, axis=1, keepdims=True)
        
        # Propagação reversa pelas camadas ocultas
        for l in range(len(self.pesos)-2, -1, -1):
            # Gradiente em relação à ativação da camada anterior
            da = np.dot(self.pesos[l+1].T, dz)
            
            # Aplicar máscara de dropout reversa
            if l < len(self.mascaras_dropout):
                da *= self.mascaras_dropout[l]
            
            # Gradiente através da função de ativação
            z = np.dot(self.pesos[l], self.ativacoes[l]) + self.vieses[l]
            dz = da * self._derivada_ativacao(z, tipo_ativacao)
            
            # Gradientes com regularização L2 e weight decay
            gradientes_pesos[l] = (1/m) * np.dot(dz, self.ativacoes[l].T) + \
                                 (self.regularizacao_l2/m) * self.pesos[l] + \
                                 self.taxa_decainento_peso * self.pesos[l]
            gradientes_vieses[l] = (1/m) * np.sum(dz, axis=1, keepdims=True)
        
        return gradientes_pesos, gradientes_vieses

    def _otimizador_adam_avancado(self, gradientes_pesos, gradientes_vieses):
        """Implementação sofisticada do otimizador Adam com correções"""
        self.iteracao += 1
        pesos_atualizados = []
        vieses_atualizados = []
        
        for i, (peso, vies, grad_w, grad_b) in enumerate(zip(
            self.pesos, self.vieses, gradientes_pesos, gradientes_vieses)):
            
            # Atualizar momentos de primeira ordem (momentum)
            self.velocidade_pesos[i] = self.beta1 * self.velocidade_pesos[i] + \
                                      (1 - self.beta1) * grad_w
            self.velocidade_vieses[i] = self.beta1 * self.velocidade_vieses[i] + \
                                       (1 - self.beta1) * grad_b
            
            # Atualizar momentos de segunda ordem (RMSProp)
            self.estado_quadrados_pesos[i] = self.beta2 * self.estado_quadrados_pesos[i] + \
                                            (1 - self.beta2) * (grad_w ** 2)
            self.estado_quadrados_vieses[i] = self.beta2 * self.estado_quadrados_vieses[i] + \
                                             (1 - self.beta2) * (grad_b ** 2)
            
            # Correção de viés dos momentos
            velocidade_pesos_corrigida = self.velocidade_pesos[i] / \
                                       (1 - self.beta1 ** self.iteracao)
            velocidade_vieses_corrigida = self.velocidade_vieses[i] / \
                                        (1 - self.beta1 ** self.iteracao)
            estado_pesos_corrigido = self.estado_quadrados_pesos[i] / \
                                   (1 - self.beta2 ** self.iteracao)
            estado_vieses_corrigido = self.estado_quadrados_vieses[i] / \
                                    (1 - self.beta2 ** self.iteracao)
            
            # Atualização dos parâmetros
            novo_peso = peso - self.taxa_aprendizado * velocidade_pesos_corrigida / \
                       (np.sqrt(estado_pesos_corrigido) + self.epsilon)
            novo_vies = vies - self.taxa_aprendizado * velocidade_vieses_corrigida / \
                       (np.sqrt(estado_vieses_corrigido) + self.epsilon)
            
            pesos_atualizados.append(novo_peso)
            vieses_atualizados.append(novo_vies)
        
        return pesos_atualizados, vieses_atualizados

    def _softmax_estavel(self, x):
        """Implementação numericamente estável do softmax"""
        exp_shifted = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_shifted / np.sum(exp_shifted, axis=0, keepdims=True)

    def _calcular_perda(self, saida_prevista, saida_real):
        """Cálculo de perda com múltiplas componentes de regularização"""
        m = saida_real.shape[1]
        
        # Cross-entropy
        entropia_cruzada = -np.mean(np.sum(saida_real * np.log(saida_prevista + self.epsilon), axis=0))
        
        # Regularização L2
        penalidade_l2 = sum(np.sum(w**2) for w in self.pesos)
        perda_l2 = (self.regularizacao_l2 / (2 * m)) * penalidade_l2
        
        # Weight decay
        penalidade_decainento = sum(np.sum(w**2) for w in self.pesos)
        perda_decainento = self.taxa_decainento_peso * penalidade_decainento
        
        return entropia_cruzada + perda_l2 + perda_decainento

    def _aumento_dados_avancado(self, dados, rotulos=None):
        """Técnicas sofisticadas de aumento de dados"""
        dados_aumentados = dados.copy()
        
        if len(dados.shape) == 2:  # Assumindo dados de imagem
            # Adicionar ruído gaussiano
            ruido = np.random.normal(0, 0.05, dados.shape)
            dados_aumentados += ruido
            
            # Aplicar pequenas rotações (para imagens)
            for i in range(dados.shape[1]):
                imagem = dados[:, i].reshape(int(np.sqrt(dados.shape[0])), -1)
                angulo = np.random.uniform(-5, 5)
                imagem_rotacionada = ndimage.rotate(imagem, angulo, reshape=False, mode='reflect')
                dados_aumentados[:, i] = imagem_rotacionada.flatten()
        
        return dados_aumentados

    def _agendamento_taxa_aprendizado(self, epoca, epocas_totais):
        """Schedule sofisticado da taxa de aprendizado"""
        # Cosine annealing
        taxa_minima = self.taxa_aprendizado * 0.01
        ciclo = math.cos(math.pi * epoca / epocas_totais)
        return taxa_minima + 0.5 * (self.taxa_aprendizado - taxa_minima) * (1 + ciclo)

    def treinar(self, dados_treino, rotulos_treino, epocas=100, 
                tamanho_lote=32, dados_validacao=None, rotulos_validacao=None,
                paciencia_early_stopping=10):
        """Algoritmo de treinamento sofisticado com múltiplas funcionalidades"""
        
        melhor_perda = float('inf')
        contador_paciencia = 0
        
        for epoca in range(epocas):
            # Agendar taxa de aprendizado
            taxa_atual = self._agendamento_taxa_aprendizado(epoca, epocas)
            self.taxa_aprendizado = taxa_atual
            
            # Embaralhar dados
            indices_embaralhados = np.random.permutation(dados_treino.shape[1])
            dados_embaralhados = dados_treino[:, indices_embaralhados]
            rotulos_embaralhados = rotulos_treino[:, indices_embaralhados]
            
            perda_epoca = 0
            acuracia_epoca = 0
            lotes = 0
            
            for i in range(0, dados_treino.shape[1], tamanho_lote):
                # Mini-batch
                fim = min(i + tamanho_lote, dados_treino.shape[1])
                dados_lote = dados_embaralhados[:, i:fim]
                rotulos_lote = rotulos_embaralhados[:, i:fim]
                
                # Aumento de dados
                dados_lote = self._aumento_dados_avancado(dados_lote)
                
                # Forward e backward propagation
                saida_prevista = self._propagacao_direta(dados_lote, treinamento=True)
                gradientes_pesos, gradientes_vieses = self._propagacao_reversa(rotulos_lote)
                
                # Otimização com Adam
                novos_pesos, novos_vieses = self._otimizador_adam_avancado(
                    gradientes_pesos, gradientes_vieses)
                
                self.pesos = novos_pesos
                self.vieses = novos_vieses
                
                # Calcular métricas
                perda_lote = self._calcular_perda(saida_prevista, rotulos_lote)
                acuracia_lote = self._calcular_acuracia(saida_prevista, rotulos_lote)
                
                perda_epoca += perda_lote
                acuracia_epoca += acuracia_lote
                lotes += 1
            
            # Métricas médias da época
            perda_media = perda_epoca / lotes
            acuracia_media = acuracia_epoca / lotes
            
            self.historico_loss.append(perda_media)
            self.historico_acuracia.append(acuracia_media)
            
            # Validação
            if dados_validacao is not None:
                saida_validacao = self._propagacao_direta(dados_validacao, treinamento=False)
                perda_validacao = self._calcular_perda(saida_validacao, rotulos_validacao)
                
                # Early stopping
                if perda_validacao < melhor_perda:
                    melhor_perda = perda_validacao
                    contador_paciencia = 0
                    # Salvar melhor modelo (implementação simplificada)
                else:
                    contador_paciencia += 1
                    
                if contador_paciencia >= paciencia_early_stopping:
                    print(f"Early stopping na época {epoca}")
                    break
            
            if epoca % 10 == 0:
                print(f"Época {epoca}: Perda = {perda_media:.4f}, Acurácia = {acuracia_media:.4f}")

    def _calcular_acuracia(self, saida_prevista, saida_real):
        """Cálculo de acurácia com suporte a múltiplas classes"""
        predicoes = np.argmax(saida_prevista, axis=0)
        rotulos = np.argmax(saida_real, axis=0)
        return np.mean(predicoes == rotulos)

    def prever(self, dados):
        """Fazer previsões em modo de inferência"""
        return self._propagacao_direta(dados, treinamento=False)

    def obter_metricas(self):
        """Retornar histórico de métricas"""
        return {
            'perda': self.historico_loss,
            'acuracia': self.historico_acuracia
        }

# Exemplo de uso sofisticado:
if __name__ == "__main__":
    # Configuração avançada da rede
    arquitetura = [784, 512, 256, 128, 10]  # Deep network
    
    # Criar rede neural com hyperparâmetros otimizados
    rede = RedeNeuralProfunda(
        arquitetura=arquitetura,
        taxa_aprendizado=0.001,
        regularizacao_l2=0.01,
        taxa_dropout=0.2,
        momentum=0.9,
        taxa_decainento_peso=0.0001
    )
    
 
    X_treino = np.random.randn(784, 5000)
    y_treino = np.eye(10)[np.random.randint(0, 10, 5000)].T
    
    X_val = np.random.randn(784, 1000)
    y_val = np.eye(10)[np.random.randint(0, 10, 1000)].T
    
    # Treinamento avançado
    rede.treinar(
        dados_treino=X_treino,
        rotulos_treino=y_treino,
        dados_validacao=X_val,
        rotulos_validacao=y_val,
        epocas=200,
        tamanho_lote=128,
        paciencia_early_stopping=15
    )
    
    # Fazer previsões
    X_teste = np.random.randn(784, 100)
    previsoes = rede.prever(X_teste)
    
    print("Forma das previsões:", previsoes.shape)
    print("Rede neural sofisticada treinada com sucesso!")