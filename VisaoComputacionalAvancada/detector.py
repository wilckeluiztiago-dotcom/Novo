import cv2
import numpy as np
from ultralytics import YOLO
import time

class DetectorDeObjetos:
    def __init__(self, modelo_path='yolov8n.pt'):
        """
        Inicializa o detector de objetos.
        
        Args:
            modelo_path (str): Caminho ou nome do modelo YOLO a ser usado.
                               'yolov8n.pt' é mais rápido, 'yolov8m.pt' é mais preciso.
        """
        print(f"Carregando modelo {modelo_path}...")
        self.modelo = YOLO(modelo_path)
        self.cores = np.random.uniform(0, 255, size=(100, 3))
        
        # Dicionário de tradução COCO (Inglês -> Português)
        self.traducoes = {
            'person': 'Pessoa', 'bicycle': 'Bicicleta', 'car': 'Carro', 'motorcycle': 'Moto',
            'airplane': 'Aviao', 'bus': 'Onibus', 'train': 'Trem', 'truck': 'Caminhao',
            'boat': 'Barco', 'traffic light': 'Semaforo', 'fire hydrant': 'Hidrante',
            'stop sign': 'Placa Pare', 'parking meter': 'Parquimetro', 'bench': 'Banco',
            'bird': 'Passaro', 'cat': 'Gato', 'dog': 'Cachorro', 'horse': 'Cavalo',
            'sheep': 'Ovelha', 'cow': 'Vaca', 'elephant': 'Elefante', 'bear': 'Urso',
            'zebra': 'Zebra', 'giraffe': 'Girafa', 'backpack': 'Mochila', 'umbrella': 'Guarda-chuva',
            'handbag': 'Bolsa', 'tie': 'Gravata', 'suitcase': 'Mala', 'frisbee': 'Frisbee',
            'skis': 'Esquis', 'snowboard': 'Snowboard', 'sports ball': 'Bola de Esportes',
            'kite': 'Pipa', 'baseball bat': 'Taco de Beisebol', 'baseball glove': 'Luva de Beisebol',
            'skateboard': 'Skate', 'surfboard': 'Prancha de Surf', 'tennis racket': 'Raquete de Tenis',
            'bottle': 'Garrafa', 'wine glass': 'Taca de Vinho', 'cup': 'Copo', 'fork': 'Garfo',
            'knife': 'Faca', 'spoon': 'Colher', 'bowl': 'Tigela', 'banana': 'Banana',
            'apple': 'Maca', 'sandwich': 'Sanduiche', 'orange': 'Laranja', 'broccoli': 'Brocolis',
            'carrot': 'Cenoura', 'hot dog': 'Cachorro Quente', 'pizza': 'Pizza', 'donut': 'Rosquinha',
            'cake': 'Bolo', 'chair': 'Cadeira', 'couch': 'Sofa', 'potted plant': 'Planta',
            'bed': 'Cama', 'dining table': 'Mesa de Jantar', 'toilet': 'Privada', 'tv': 'TV',
            'laptop': 'Notebook', 'mouse': 'Mouse', 'remote': 'Controle Remoto', 'keyboard': 'Teclado',
            'cell phone': 'Celular', 'microwave': 'Microondas', 'oven': 'Forno', 'toaster': 'Torradeira',
            'sink': 'Pia', 'refrigerator': 'Geladeira', 'book': 'Livro', 'clock': 'Relogio',
            'vase': 'Vaso', 'scissors': 'Tesoura', 'teddy bear': 'Urso de Pelucia',
            'hair drier': 'Secador de Cabelo', 'toothbrush': 'Escova de Dentes'
        }

    def detectar(self, quadro):
        """
        Realiza a detecção de objetos em um quadro (imagem).
        
        Args:
            quadro (numpy.ndarray): Imagem/Frame para processar.
            
        Returns:
            list: Lista de resultados da detecção.
        """
        resultados = self.modelo(quadro, verbose=False)
        return resultados

    def desenhar_caixas(self, quadro, resultados):
        """
        Desenha as caixas delimitadoras e rótulos no quadro.
        
        Args:
            quadro (numpy.ndarray): O quadro original.
            resultados (list): Resultados da detecção do YOLO.
            
        Returns:
            numpy.ndarray: Quadro com anotações.
        """
        quadro_anotado = quadro.copy()
        
        for r in resultados:
            caixas = r.boxes
            for caixa in caixas:
                # Coordenadas da caixa
                x1, y1, x2, y2 = caixa.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Confiança e Classe
                confianca = float(caixa.conf[0])
                cls = int(caixa.cls[0])
                nome_classe_ingles = self.modelo.names[cls]
                nome_classe_pt = self.traducoes.get(nome_classe_ingles, nome_classe_ingles)
                
                # Cor para a classe
                cor = self.cores[cls % len(self.cores)]
                
                # Desenhar retângulo
                cv2.rectangle(quadro_anotado, (x1, y1), (x2, y2), cor, 2)
                
                # Texto do rótulo
                texto = f"{nome_classe_pt} {confianca:.2f}"
                (w, h), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                
                # Fundo do texto para legibilidade
                cv2.rectangle(quadro_anotado, (x1, y1 - 20), (x1 + w, y1), cor, -1)
                cv2.putText(quadro_anotado, texto, (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                            
        return quadro_anotado
