"""Baixa o detector facial leve usado pelo filtro local."""
from pathlib import Path
from urllib.request import urlretrieve
import cv2
from envelhecimento import MODELO_LOCAL, carregar_modelo

URL = 'https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'

if __name__ == '__main__':
    MODELO_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    if not MODELO_LOCAL.exists():
        temporario = MODELO_LOCAL.with_suffix('.download.onnx')
        try:
            print('Baixando o detector facial YuNet...')
            urlretrieve(URL, temporario)
            cv2.FaceDetectorYN.create(str(temporario), '', (320, 320))
            temporario.replace(MODELO_LOCAL)
        finally:
            temporario.unlink(missing_ok=True)
    carregar_modelo()
    from huggingface_hub import snapshot_download
    from envelhecimento_generativo import MODELO_ID, MODELO_LOCAL as PASTA_TEXTURA
    from tratamento_realista import PASTA as PASTA_SEGMENTACAO
    print('Baixando os modelos de textura e segmentação. Essa etapa pode demorar.')
    snapshot_download('jonathandinu/face-parsing', local_dir=str(PASTA_SEGMENTACAO),
                      allow_patterns=['config.json', 'preprocessor_config.json', 'model.safetensors', 'README.md'])
    snapshot_download(MODELO_ID, local_dir=str(PASTA_TEXTURA),
                      allow_patterns=['model_index.json', '*/config.json', '*/scheduler_config.json',
                                      'tokenizer/*', 'feature_extractor/*', '*/*.fp16.safetensors'])
    print('Modelos prontos. Abra o aplicativo para preparar a execução local.')
