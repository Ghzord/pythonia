"""Envelhecimento sutil por edição da foto original, sem regenerar o rosto."""
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


def aplicar_tracos(foto, rosto, idade):
    """Sobrepõe luz e sombra; não move, substitui ou reconstrói nenhum pixel."""
    original = np.asarray(foto.convert('RGB'))
    # Escala artística de envelhecimento: não estima a idade real da pessoa.
    quantidade = float(np.interp(idade, [40, 50, 60, 70, 80, 90], [0.38, 0.75, 1.0, 1.3, 1.6, 1.9]))
    maturidade = float(np.clip((idade - 50) / 40, 0, 1))
    altura, largura = original.shape[:2]
    olhos = sorted(rosto[4:8].reshape(2, 2), key=lambda p: p[0])
    centro = (olhos[0] + olhos[1]) / 2
    distancia = float(np.linalg.norm(olhos[1] - olhos[0]))
    eixo_x = (olhos[1] - olhos[0]) / distancia
    eixo_y = np.array([-eixo_x[1], eixo_x[0]])

    def ponto(x, y):
        return centro + distancia * (x * eixo_x + y * eixo_y)

    def relativo(p):
        delta = p - centro
        return np.array([np.dot(delta, eixo_x), np.dot(delta, eixo_y)]) / distancia

    def elipse(tela, x, y, rx, ry, valor):
        angulos = np.linspace(0, 2 * np.pi, 80)
        pontos = np.array([ponto(x + rx * np.cos(a), y + ry * np.sin(a)) for a in angulos])
        cv2.fillConvexPoly(tela, np.rint(pontos).astype(np.int32), valor, lineType=cv2.LINE_AA)

    nariz = relativo(rosto[8:10])
    boca = relativo((rosto[10:12] + rosto[12:14]) / 2)
    largura_boca = float(np.linalg.norm(rosto[12:14] - rosto[10:12]) / distancia)
    mascara = np.zeros((altura, largura), np.uint8)
    elipse(mascara, 0, 0.35, 0.94, 1.35, 255)
    protecao = np.zeros_like(mascara)
    for x in (-0.5, 0.5):
        elipse(protecao, x, 0, 0.28, 0.14, 255)
        elipse(protecao, x, -0.26, 0.33, 0.10, 255)
    elipse(protecao, boca[0], boca[1], largura_boca * 0.65, 0.25, 255)
    elipse(protecao, nariz[0], nariz[1], 0.20, 0.16, 255)
    suavidade = max(0.6, distancia * 0.015)
    mascara = cv2.GaussianBlur(mascara.astype(np.float32) / 255, (0, 0), suavidade)
    # A borda externa e as feições protegidas permanecem idênticas ao original.
    suporte = np.zeros((altura, largura), np.uint8)
    elipse(suporte, 0, 0.35, 0.96, 1.37, 255)
    mascara[suporte == 0] = 0
    mascara *= 1 - cv2.GaussianBlur(protecao.astype(np.float32) / 255, (0, 0), suavidade)
    mascara[protecao > 0] = 0

    rugas = np.zeros((altura, largura), np.uint8)
    largura_linha = max(1, round(distancia * 0.007))

    def curva(a, b, c, opacidade=200):
        t = np.linspace(0, 1, 36)[:, None]
        pts = (1-t)**2 * np.array(a) + 2*(1-t)*t*np.array(b) + t**2*np.array(c)
        pts = np.rint([ponto(*p) for p in pts]).astype(np.int32)
        cv2.polylines(rugas, [pts], False, opacidade, largura_linha, cv2.LINE_AA)

    # Testa: linhas curtas, levemente curvas e de intensidades diferentes.
    for y, extensao, opacidade in [(-0.53, 0.54, 175), (-0.72, 0.60, 150), (-0.88, 0.43, 120)]:
        curva((-extensao, y), (0.04, y-0.06), (extensao, y+0.01), opacidade)
    for lado in (-1, 1):
        # Cantos dos olhos e marcas finas abaixo das pálpebras.
        for fim_y, alpha in [(-0.08, 140), (0.08, 200), (0.23, 150)]:
            curva((lado*0.77, 0.06), (lado*0.86, fim_y*0.65), (lado*0.92, fim_y), alpha)
        curva((lado*0.28, 0.19), (lado*0.52, 0.34), (lado*0.72, 0.23), 160)
        curva((lado*0.33, 0.30), (lado*0.53, 0.40), (lado*0.68, 0.33), 90)
        # Sulcos suaves nas laterais da boca, sem desenhar sobre os lábios.
        curva((nariz[0]+lado*0.28, nariz[1]+0.10),
              (boca[0]+lado*(largura_boca*0.64+0.06), boca[1]-0.28),
              (boca[0]+lado*(largura_boca*0.65+0.07), boca[1]+0.17), 140)
        # Marcas adicionais aparecem gradualmente nas idades mais altas.
        if maturidade > 0:
            curva((lado*0.32, 0.40), (lado*0.51, 0.51), (lado*0.71, 0.43), round(150*maturidade))
            for deslocamento in (0.0, 0.10):
                curva((lado*(0.65+deslocamento), 0.53),
                      (lado*(0.72+deslocamento), 0.73),
                      (lado*(0.67+deslocamento), 0.91), round(100*maturidade))

    sombra = cv2.GaussianBlur(rugas.astype(np.float32) / 255, (0, 0), max(0.5, distancia*0.006))
    # Pequeno realce ao lado do sulco dá volume, sem contornos escuros fortes.
    luz = cv2.warpAffine(sombra, np.float32([[1, 0, eixo_y[0]*1.2], [0, 1, eixo_y[1]*1.2]]), (largura, altura))
    delta = (luz * 10 - sombra * 60) * quantidade
    volume = np.zeros((altura, largura), np.uint8)
    for lado in (-1, 1):
        elipse(volume, lado*0.50, 0.27, 0.24, 0.09, 255)
    volume = cv2.GaussianBlur(volume.astype(np.float32)/255, (0, 0), max(1.0, distancia*0.04))
    delta -= volume * (3.5 + 2*maturidade) * quantidade
    base = original.astype(np.float32)
    detalhe = base - cv2.GaussianBlur(base, (0, 0), max(0.7, distancia*0.009))
    alterada = base + (delta[..., None] + detalhe * (0.22 * quantidade)) * mascara[..., None]
    resultado = np.clip(np.rint(alterada), 0, 255).astype(np.uint8)
    return Image.fromarray(resultado)


def gerar_imagem(foto: Image.Image, idade: int = 50, perfil: str = 'rapido') -> Image.Image:
    if perfil not in PERFIS or not 40 <= idade <= 90:
        raise FotoInadequada('Escolha uma idade sugerida entre 40 e 90 anos.')
    foto = foto.convert('RGB')
    rosto = localizar_rosto(foto)
    return aplicar_tracos(foto, rosto, idade)
