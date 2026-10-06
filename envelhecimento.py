"""Detecta o rosto para tratamento de textura e grisalho na foto original."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

MODELO_LOCAL = Path(__file__).resolve().parent / 'modelos' / 'face_detection_yunet.onnx'
# Compatibilidade com clientes antigos; ambos usam o mesmo filtro conservador.
PERFIS = {'rapido', 'detalhado'}
_detector = None


class FotoInadequada(ValueError):
    pass


def carregar_modelo():
    global _detector
    if _detector is None:
        if not MODELO_LOCAL.exists():
            raise RuntimeError('Baixe o detector executando python preparar_modelo.py antes da apresentação.')
        _detector = cv2.FaceDetectorYN.create(str(MODELO_LOCAL), '', (320, 320), 0.85, 0.3, 5000)
    return _detector


def preparar_apresentacao():
    carregar_modelo()
    from tratamento_realista import carregar_segmentador
    from envelhecimento_generativo import preparar_apresentacao as preparar_textura
    print('Carregando segmentação de cabelo e pele...', flush=True)
    carregar_segmentador()
    print('Preparando o modelo de textura...', flush=True)
    preparar_textura()


def localizar_rosto(foto):
    pixels = np.asarray(foto.convert('RGB'))
    altura, largura = pixels.shape[:2]
    escala = min(1.0, 640 / max(altura, largura))
    miniatura = cv2.resize(pixels, (round(largura * escala), round(altura * escala)))
    detector = carregar_modelo()
    detector.setInputSize((miniatura.shape[1], miniatura.shape[0]))
    _, rostos = detector.detect(cv2.cvtColor(miniatura, cv2.COLOR_RGB2BGR))
    if rostos is None or len(rostos) == 0:
        raise FotoInadequada('Não encontrei um rosto nítido. Use uma foto de frente, bem iluminada, sem cobrir o rosto.')
    if len(rostos) != 1:
        raise FotoInadequada('Use uma foto com apenas uma pessoa, para aplicar o efeito no rosto correto.')
    rosto = rostos[0].copy()
    rosto[:14] /= escala
    olhos = sorted(rosto[4:8].reshape(2, 2), key=lambda ponto: ponto[0])
    distancia = np.linalg.norm(olhos[1] - olhos[0])
    if distancia < 35:
        raise FotoInadequada('O rosto está pequeno na foto. Recorte a imagem ou escolha uma foto mais próxima.')
    nariz = rosto[8:10]
    eixo = (olhos[1] - olhos[0]) / distancia
    deslocamento = abs(np.dot(nariz - (olhos[0] + olhos[1]) / 2, eixo)) / distancia
    if deslocamento > 0.35:
        raise FotoInadequada('Use uma foto mais de frente para posicionar as marcas com cuidado.')
    return rosto


def gerar_imagem(foto: Image.Image, idade: int = 50, perfil: str = 'rapido') -> Image.Image:
    if perfil not in PERFIS or not 40 <= idade <= 90:
        raise FotoInadequada('Escolha uma idade sugerida entre 40 e 90 anos.')
    foto = foto.convert('RGB')
    rosto = localizar_rosto(foto)
    from tratamento_realista import gerar_tratamento
    return gerar_tratamento(foto, rosto, idade)
