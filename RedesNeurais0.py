import numpy as np

class RedeNeuralFeedforward:
    def __init__(self, arquitetura, taxa_aprendizado=0.01):
        self.arquitetura = arquitetura
        self.taxa_aprendizado = taxa_aprendizado
        self.pesos = []
        self.vieses = []
        
        for i in range(len(arquitetura) - 1):
            peso = np.random.randn(arquitetura[i], arquitetura[i + 1]) * 0.1
            vies = np.zeros((1, arquitetura[i + 1]))
            self.pesos.append(peso)
            self.vieses.append(vies)
    
    def funcao_ativacao(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def derivada_ativacao(self, x):
        return x * (1 - x)
    
    def propagacao_direta(self, X):
        self.ativacoes = [X]
        self.zs = []
        ativacao_atual = X
        
        for i in range(len(self.pesos)):
            z = np.dot(ativacao_atual, self.pesos[i]) + self.vieses[i]
            ativacao_atual = self.funcao_ativacao(z)
            self.zs.append(z)
            self.ativacoes.append(ativacao_atual)
        
        return ativacao_atual
    
    def retropropagacao(self, X, y, saida):
        m = X.shape[0]
        deltas = [None] * len(self.pesos)
        erro = saida - y
        delta = erro * self.derivada_ativacao(self.ativacoes[-1])
        deltas[-1] = delta
        
        for i in range(len(self.pesos) - 2, -1, -1):
            delta = np.dot(deltas[i + 1], self.pesos[i + 1].T) * self.derivada_ativacao(self.ativacoes[i + 1])
            deltas[i] = delta
        
        for i in range(len(self.pesos)):
            grad_peso = np.dot(self.ativacoes[i].T, deltas[i]) / m
            grad_vies = np.sum(deltas[i], axis=0, keepdims=True) / m
            self.pesos[i] -= self.taxa_aprendizado * grad_peso
            self.vieses[i] -= self.taxa_aprendizado * grad_vies
    
    def treinar(self, X, y, epocas, verbose=True):
        for epoca in range(epocas):
            saida = self.propagacao_direta(X)
            self.retropropagacao(X, y, saida)
            if verbose and epoca % 100 == 0:
                perda = np.mean((saida - y) ** 2)
                print(f"Época {epoca}, Perda: {perda:.4f}")
    
    def prever(self, X):
        return self.propagacao_direta(X)

class RedeConvolucional:
    def __init__(self, num_filtros=3, tamanho_filtro=3):
        self.num_filtros = num_filtros
        self.tamanho_filtro = tamanho_filtro
        self.filtros = np.random.randn(num_filtros, tamanho_filtro, tamanho_filtro) * 0.1
    
    def convolucao(self, imagem, filtro):
        altura_img, largura_img = imagem.shape
        altura_saida = altura_img - self.tamanho_filtro + 1
        largura_saida = largura_img - self.tamanho_filtro + 1
        saida = np.zeros((altura_saida, largura_saida))
        
        for i in range(altura_saida):
            for j in range(largura_saida):
                regiao = imagem[i:i + self.tamanho_filtro, j:j + self.tamanho_filtro]
                saida[i, j] = np.sum(regiao * filtro)
        
        return saida
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def pooling_max(self, mapa_caracteristicas, tamanho_janela=2):
        altura, largura = mapa_caracteristicas.shape
        altura_pool = altura // tamanho_janela
        largura_pool = largura // tamanho_janela
        saida = np.zeros((altura_pool, largura_pool))
        
        for i in range(altura_pool):
            for j in range(largura_pool):
                regiao = mapa_caracteristicas[
                    i * tamanho_janela:(i + 1) * tamanho_janela,
                    j * tamanho_janela:(j + 1) * tamanho_janela
                ]
                saida[i, j] = np.max(regiao)
        
        return saida
    
    def forward(self, imagem):
        self.mapas_caracteristicas = []
        mapa_ativado = []
        
        for i in range(self.num_filtros):
            mapa_convolucao = self.convolucao(imagem, self.filtros[i])
            mapa_ativacao = self.relu(mapa_convolucao)
            mapa_pooling = self.pooling_max(mapa_ativacao)
            self.mapas_caracteristicas.append(mapa_convolucao)
            mapa_ativado.append(mapa_pooling)
        
        return np.array(mapa_ativado)

class LSTM:
    def __init__(self, unidades_entrada, unidades_ocultas, unidades_saida):
        self.unidades_ocultas = unidades_ocultas
        
        # Pesos para portão de esquecimento
        self.Wf = np.random.randn(unidades_entrada + unidades_ocultas, unidades_ocultas) * 0.01
        self.bf = np.zeros((1, unidades_ocultas))
        
        # Pesos para portão de entrada
        self.Wi = np.random.randn(unidades_entrada + unidades_ocultas, unidades_ocultas) * 0.01
        self.bi = np.zeros((1, unidades_ocultas))
        
        # Pesos para célula candidata
        self.Wc = np.random.randn(unidades_entrada + unidades_ocultas, unidades_ocultas) * 0.01
        self.bc = np.zeros((1, unidades_ocultas))
        
        # Pesos para portão de saída
        self.Wo = np.random.randn(unidades_entrada + unidades_ocultas, unidades_ocultas) * 0.01
        self.bo = np.zeros((1, unidades_ocultas))
        
        # Pesos para camada de saída
        self.Wy = np.random.randn(unidades_ocultas, unidades_saida) * 0.01
        self.by = np.zeros((1, unidades_saida))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def tanh(self, x):
        return np.tanh(x)
    
    def forward(self, X):
        self.X = X
        batch_size, seq_len, input_dim = X.shape
        
        self.h = np.zeros((batch_size, seq_len + 1, self.unidades_ocultas))
        self.c = np.zeros((batch_size, seq_len + 1, self.unidades_ocultas))
        
        self.f = np.zeros((batch_size, seq_len, self.unidades_ocultas))
        self.i = np.zeros((batch_size, seq_len, self.unidades_ocultas))
        self.c_linha = np.zeros((batch_size, seq_len, self.unidades_ocultas))
        self.o = np.zeros((batch_size, seq_len, self.unidades_ocultas))
        
        saidas = []
        
        for t in range(seq_len):
            x_t = X[:, t, :]
            combinado = np.concatenate([self.h[:, t, :], x_t], axis=1)
            
            # Portão de esquecimento
            self.f[:, t, :] = self.sigmoid(np.dot(combinado, self.Wf) + self.bf)
            
            # Portão de entrada
            self.i[:, t, :] = self.sigmoid(np.dot(combinado, self.Wi) + self.bi)
            
            # Célula candidata
            self.c_linha[:, t, :] = self.tanh(np.dot(combinado, self.Wc) + self.bc)
            
            # Portão de saída
            self.o[:, t, :] = self.sigmoid(np.dot(combinado, self.Wo) + self.bo)
            
            # Estado da célula
            self.c[:, t + 1, :] = (self.f[:, t, :] * self.c[:, t, :] + 
                                  self.i[:, t, :] * self.c_linha[:, t, :])
            
            # Estado oculto
            self.h[:, t + 1, :] = self.o[:, t, :] * self.tanh(self.c[:, t + 1, :])
            
            # Saída
            y_t = np.dot(self.h[:, t + 1, :], self.Wy) + self.by
            saidas.append(y_t)
        
        return np.array(saidas).transpose(1, 0, 2)

class Autoencoder:
    def __init__(self, dimensao_entrada, dimensao_latente, taxa_aprendizado=0.01):
        self.taxa_aprendizado = taxa_aprendizado
        
        # Codificador
        self.W_encoder = np.random.randn(dimensao_entrada, dimensao_latente) * 0.01
        self.b_encoder = np.zeros((1, dimensao_latente))
        
        # Decodificador
        self.W_decoder = np.random.randn(dimensao_latente, dimensao_entrada) * 0.01
        self.b_decoder = np.zeros((1, dimensao_entrada))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def encoder(self, X):
        z = np.dot(X, self.W_encoder) + self.b_encoder
        return self.sigmoid(z)
    
    def decoder(self, z):
        X_reconstruido = np.dot(z, self.W_decoder) + self.b_decoder
        return self.sigmoid(X_reconstruido)
    
    def forward(self, X):
        self.codigo_latente = self.encoder(X)
        self.reconstrucao = self.decoder(self.codigo_latente)
        return self.reconstrucao
    
    def treinar(self, X, epocas):
        for epoca in range(epocas):
            # Forward pass
            reconstrucao = self.forward(X)
            
            # Calcular perda
            perda = np.mean((X - reconstrucao) ** 2)
            
            # Backward pass
            erro = reconstrucao - X
            d_reconstrucao = erro * reconstrucao * (1 - reconstrucao)
            
            # Gradientes do decodificador
            grad_W_decoder = np.dot(self.codigo_latente.T, d_reconstrucao) / X.shape[0]
            grad_b_decoder = np.sum(d_reconstrucao, axis=0, keepdims=True) / X.shape[0]
            
            # Gradientes do codificador
            d_codigo_latente = np.dot(d_reconstrucao, self.W_decoder.T) * self.codigo_latente * (1 - self.codigo_latente)
            grad_W_encoder = np.dot(X.T, d_codigo_latente) / X.shape[0]
            grad_b_encoder = np.sum(d_codigo_latente, axis=0, keepdims=True) / X.shape[0]
            
            # Atualizar pesos
            self.W_decoder -= self.taxa_aprendizado * grad_W_decoder
            self.b_decoder -= self.taxa_aprendizado * grad_b_decoder
            self.W_encoder -= self.taxa_aprendizado * grad_W_encoder
            self.b_encoder -= self.taxa_aprendizado * grad_b_encoder
            
            if epoca % 100 == 0:
                print(f"Época {epoca}, Perda: {perda:.4f}")

class RedeGenerativaAdversaria:
    def __init__(self, dimensao_latente, dimensao_dados, taxa_aprendizado=0.001):
        self.dimensao_latente = dimensao_latente
        self.taxa_aprendizado = taxa_aprendizado
        
        # Gerador
        self.W_g1 = np.random.randn(dimensao_latente, 128) * 0.01
        self.b_g1 = np.zeros((1, 128))
        self.W_g2 = np.random.randn(128, dimensao_dados) * 0.01
        self.b_g2 = np.zeros((1, dimensao_dados))
        
        # Discriminador
        self.W_d1 = np.random.randn(dimensao_dados, 128) * 0.01
        self.b_d1 = np.zeros((1, 128))
        self.W_d2 = np.random.randn(128, 1) * 0.01
        self.b_d2 = np.zeros((1, 1))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def gerador(self, z):
        camada1 = self.relu(np.dot(z, self.W_g1) + self.b_g1)
        saida = np.dot(camada1, self.W_g2) + self.b_g2
        return self.sigmoid(saida)
    
    def discriminador(self, x):
        camada1 = self.relu(np.dot(x, self.W_d1) + self.b_d1)
        saida = np.dot(camada1, self.W_d2) + self.b_d2
        return self.sigmoid(saida)
    
    def treinar(self, dados_reais, epocas, batch_size=32):
        num_amostras = dados_reais.shape[0]
        
        for epoca in range(epocas):
            # Treinar discriminador
            indices = np.random.randint(0, num_amostras, batch_size)
            batch_real = dados_reais[indices]
            
            z = np.random.randn(batch_size, self.dimensao_latente)
            batch_falso = self.gerador(z)
            
            # Forward pass do discriminador
            pred_real = self.discriminador(batch_real)
            pred_falso = self.discriminador(batch_falso)
            
            # Perda do discriminador
            perda_d = -np.mean(np.log(pred_real + 1e-8) + np.log(1 - pred_falso + 1e-8))
            
            # Treinar gerador
            z = np.random.randn(batch_size, self.dimensao_latente)
            batch_falso = self.gerador(z)
            pred_falso = self.discriminador(batch_falso)
            
            # Perda do gerador
            perda_g = -np.mean(np.log(pred_falso + 1e-8))
            
            if epoca % 100 == 0:
                print(f"Época {epoca}, Perda D: {perda_d:.4f}, Perda G: {perda_g:.4f}")

if __name__ == "__main__":
    print("=== Rede Neural Feedforward ===")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    rn = RedeNeuralFeedforward([2, 4, 1], 0.1)
    rn.treinar(X, y, 1000, verbose=False)
    predicoes = rn.prever(X)
    print("Predições XOR:", predicoes.flatten())
    
    print("\n=== Rede Convolucional ===")
    cnn = RedeConvolucional(num_filtros=2, tamanho_filtro=3)
    imagem_teste = np.random.randn(10, 10)
    saida_cnn = cnn.forward(imagem_teste)
    print(f"Formato da saída CNN: {saida_cnn.shape}")
    
    print("\n=== LSTM ===")
    lstm = LSTM(unidades_entrada=3, unidades_ocultas=5, unidades_saida=2)
    X_seq = np.random.randn(4, 6, 3)  # (batch, sequencia, características)
    saida_lstm = lstm.forward(X_seq)
    print(f"Formato da saída LSTM: {saida_lstm.shape}")
    
    print("\n=== Autoencoder ===")
    autoencoder = Autoencoder(dimensao_entrada=10, dimensao_latente=5)
    X_ae = np.random.randn(100, 10)
    autoencoder.treinar(X_ae, 500)
    
    print("\n=== GAN ===")
    gan = RedeGenerativaAdversaria(dimensao_latente=10, dimensao_dados=20)
    dados_reais = np.random.rand(1000, 20)
    gan.treinar(dados_reais, 500)