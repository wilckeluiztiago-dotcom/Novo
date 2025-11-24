# Visão Computacional Avançada com YOLOv8

Este projeto implementa um sistema de detecção de objetos em tempo real utilizando o modelo YOLOv8 (You Only Look Once), estado da arte em visão computacional. O sistema é capaz de identificar e classificar 80 tipos diferentes de objetos, com resultados traduzidos para o Português.

## Funcionalidades

- **Detecção em Tempo Real**: Processamento rápido de vídeo da webcam ou arquivos.
- **80 Classes de Objetos**: Identifica pessoas, carros, animais (gato, cachorro, cavalo), comida (banana, maçã, pizza), objetos domésticos e muito mais.
- **Interface em Português**: Todos os rótulos de identificação são exibidos em Português.
- **Visualização Clara**: Caixas delimitadoras coloridas e texto legível.

## Pré-requisitos

- Python 3.8 ou superior
- Webcam (para detecção em tempo real)

## Instalação

1.  Navegue até o diretório do projeto:
    ```bash
    cd "Área de trabalho/Projetos/VisaoComputacionalAvancada"
    ```

2.  Instale as dependências necessárias:
    ```bash
    pip install -r requirements.txt
    ```
    *Isso instalará o OpenCV, Ultralytics (YOLO) e Numpy.*

## Como Usar

### Usando a Webcam
Para iniciar a detecção usando sua webcam padrão, basta executar:

```bash
python principal.py
```

### Usando um Arquivo de Vídeo
Para processar um arquivo de vídeo existente:

```bash
python principal.py --fonte caminho/para/seu/video.mp4
```

### Escolhendo o Modelo
O padrão é o `yolov8n.pt` (nano), que é muito rápido. Para maior precisão (mas um pouco mais lento), você pode usar o modelo médio:

```bash
python principal.py --modelo yolov8m.pt
```

## Controles
- Pressione **'q'** para encerrar o programa.

## Estrutura do Código
- `principal.py`: Script principal que gerencia a captura de vídeo e loop de execução.
- `detector.py`: Contém a classe `DetectorDeObjetos` que encapsula a lógica do YOLO e as traduções.
- `requirements.txt`: Lista de bibliotecas Python necessárias.
