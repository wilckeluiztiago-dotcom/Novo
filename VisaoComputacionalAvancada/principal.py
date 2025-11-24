import cv2
import argparse
import sys
from detector import DetectorDeObjetos

def main():
    # Configuração de argumentos da linha de comando
    parser = argparse.ArgumentParser(description='Sistema Avançado de Visão Computacional')
    parser.add_argument('--fonte', type=str, default='0', 
                        help='Fonte de vídeo: "0" para webcam, ou caminho para um arquivo de vídeo')
    parser.add_argument('--modelo', type=str, default='yolov8n.pt',
                        help='Modelo YOLO a ser usado (ex: yolov8n.pt, yolov8m.pt)')
    
    args = parser.parse_args()
    
    # Inicializar o detector
    try:
        detector = DetectorDeObjetos(modelo_path=args.modelo)
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        sys.exit(1)
        
    # Configurar fonte de vídeo
    fonte = args.fonte
    if fonte.isdigit():
        fonte = int(fonte)
        
    cap = cv2.VideoCapture(fonte)
    
    if not cap.isOpened():
        print(f"Erro: Não foi possível abrir a fonte de vídeo '{fonte}'.")
        sys.exit(1)
        
    print("Iniciando detecção... Pressione 'q' para sair.")
    
    while True:
        ret, quadro = cap.read()
        
        if not ret:
            print("Fim do vídeo ou erro ao ler o quadro.")
            break
            
        # Realizar detecção
        resultados = detector.detectar(quadro)
        
        # Desenhar resultados
        quadro_anotado = detector.desenhar_caixas(quadro, resultados)
        
        # Mostrar FPS
        cv2.imshow('Visao Computacional Avancada - YOLOv8', quadro_anotado)
        
        # Sair com 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
