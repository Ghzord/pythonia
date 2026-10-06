"""Composição conservadora: textura de pele e grisalho sobre a foto original."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

PASTA = Path(__file__).resolve().parent / 'modelos' / 'face-parsing'
_segmentador = None
_processador = None


def carregar_segmentador():
    global _segmentador, _processador
    if _segmentador is None:
        import torch
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
        if not (PASTA / 'model.safetensors').exists():
            raise RuntimeError('Falta o modelo de cabelo e pele. Execute python preparar_modelo.py com internet.')
        torch.set_num_threads(min(4, torch.get_num_threads()))
        _processador = SegformerImageProcessor.from_pretrained(str(PASTA), local_files_only=True)
        _segmentador = SegformerForSemanticSegmentation.from_pretrained(
            str(PASTA), local_files_only=True, use_safetensors=True).eval()
    return _processador, _segmentador


def segmentar(foto):
    import torch
    processador, modelo = carregar_segmentador()
    with torch.inference_mode():
        entrada = processador(images=foto, return_tensors='pt')
        logits = modelo(**entrada).logits
        probabilidades = logits.softmax(dim=1)[0].numpy()
    # Redimensiona os mapas individuais, evitando um tensor grande na GPU.
    mapas = np.stack([cv2.resize(p, foto.size, interpolation=cv2.INTER_LINEAR) for p in probabilidades])
    classes = mapas.argmax(axis=0)
    pele = np.where(classes == 1, mapas[1], 0)
    cabelo = np.where(classes == 13, mapas[13], 0)
    return pele.astype(np.float32), cabelo.astype(np.float32)


def mascara_suave(probabilidade, raio):
    # Borda para dentro: não espalha o efeito sobre pele ou fundo vizinhos.
    interior = np.clip((probabilidade - 0.55) / 0.35, 0, 1)
    suavizada = cv2.GaussianBlur(interior, (0, 0), raio)
    return suavizada * (probabilidade > 0.55)


def compor(foto, textura, caixa, pele, cabelo, rosto, idade):
    original = np.asarray(foto.convert('RGB'))
    base = original.astype(np.float32)
    x0, y0, x1, y1 = caixa
    original_roi = base[y0:y1, x0:x1]
    envelhecida = np.asarray(textura.resize((x1-x0, y1-y0), Image.Resampling.LANCZOS)).astype(np.float32)
    escala = max(1.0, float(rosto[2]) / 150)
    cinza = cv2.cvtColor(original_roi, cv2.COLOR_RGB2GRAY)
    cinza_aged = cv2.cvtColor(envelhecida, cv2.COLOR_RGB2GRAY)
    sigma = max(1.5, float(rosto[2]) * 0.018)
    detalhe_original = cinza - cv2.GaussianBlur(cinza, (0, 0), sigma)
    detalhe_aged = cinza_aged - cv2.GaussianBlur(cinza_aged, (0, 0), sigma)
    # Transfere apenas variações de textura, sem substituir a geometria do rosto.
    forca = float(np.interp(idade, [40, 50, 70, 90], [0.25, 0.55, 0.85, 1.0]))
    textura_delta = np.clip(detalhe_aged - detalhe_original, -22, 15) * forca
    mascara = mascara_suave(pele, escala)
    protecao = np.zeros(pele.shape, np.uint8)
    olhos = rosto[4:8].reshape(2, 2)
    distancia = float(np.linalg.norm(olhos[1]-olhos[0]))
    for olho in olhos:
        cv2.ellipse(protecao, tuple(np.rint(olho).astype(int)),
                    (max(1, round(distancia*.32)), max(1, round(distancia*.22))), 0, 0, 360, 255, -1)
    boca = (rosto[10:12] + rosto[12:14]) / 2
    cv2.ellipse(protecao, tuple(np.rint(boca).astype(int)),
                (max(1, round(distancia*.63)), max(1, round(distancia*.27))), 0, 0, 360, 255, -1)
    cv2.ellipse(protecao, tuple(np.rint(rosto[8:10]).astype(int)),
                (max(1, round(distancia*.25)), max(1, round(distancia*.23))), 0, 0, 360, 255, -1)
    mascara *= 1-cv2.GaussianBlur(protecao.astype(np.float32)/255, (0, 0), escala)
    mascara[protecao > 0] = 0
    base[y0:y1, x0:x1] += textura_delta[..., None] * mascara[y0:y1, x0:x1, None]

    # Grisalho mantém a iluminação e as mechas existentes; não desenha cabelos.
    luminancia = cv2.cvtColor(original.astype(np.float32)/255, cv2.COLOR_RGB2GRAY)
    prata = np.power(np.clip(luminancia, 0, 1), 0.72)*255
    prata = prata[..., None] * np.array([1.0, 0.995, 0.98], np.float32)
    proporcao = float(np.interp(idade, [40, 50, 60, 70, 80, 90], [0.08, 0.28, 0.50, 0.72, 0.86, 0.95]))
    alpha = mascara_suave(cabelo, escala)[..., None] * proporcao
    base = base*(1-alpha) + prata*alpha
    return Image.fromarray(np.clip(np.rint(base), 0, 255).astype(np.uint8))


def gerar_tratamento(foto, rosto, idade):
    from envelhecimento_generativo import _processar
    pele, cabelo = segmentar(foto)
    x, y, largura, altura = rosto[:4]
    lado = max(largura*1.65, altura*1.30)
    centro_x, centro_y = x+largura/2, y+altura*.44
    caixa = (max(0, round(centro_x-lado/2)), max(0, round(centro_y-lado/2)),
             min(foto.width, round(centro_x+lado/2)), min(foto.height, round(centro_y+lado/2)))
    textura = _processar(foto.crop(caixa), idade, 384, 18)
    return compor(foto, textura, caixa, pele, cabelo, rosto, idade)
