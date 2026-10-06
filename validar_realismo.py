"""Gera uma comparação local do tratamento, sem enviar fotos a serviços externos."""
import argparse
import json
import time
from pathlib import Path
from PIL import Image, ImageDraw
from envelhecimento import localizar_rosto
from tratamento_realista import gerar_tratamento

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('foto', type=Path)
    args = parser.parse_args()
    foto = Image.open(args.foto).convert('RGB')
    foto.thumbnail((1024, 1024))
    rosto = localizar_rosto(foto)
    pasta = Path('resultados_benchmark/realismo')
    pasta.mkdir(exist_ok=True, parents=True)
    quadros = [foto]
    for idade in (50, 70):
        inicio = time.perf_counter()
        saida = gerar_tratamento(foto, rosto, idade)
        saida.save(pasta / f'{args.foto.stem}_{idade}.png')
        quadros.append(saida)
        print(json.dumps(dict(idade=idade, segundos=round(time.perf_counter()-inicio, 2))), flush=True)
    largura = 400
    altura = round(foto.height * largura / foto.width)
    comparacao = Image.new('RGB', (largura*3, altura+28), 'white')
    for i, (quadro, titulo) in enumerate(zip(quadros, ['Original', 'Referencia: 50 anos', 'Referencia: 70 anos'])):
        comparacao.paste(quadro.resize((largura, altura)), (i*largura, 28))
        ImageDraw.Draw(comparacao).text((i*largura+8, 8), titulo, fill='black')
    comparacao.save(pasta / f'{args.foto.stem}_comparacao.png')
