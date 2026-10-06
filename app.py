"""Recebe a foto do HTML e devolve a simulação gerada localmente."""
import warnings
import hashlib
from collections import OrderedDict
from pathlib import Path
from threading import Lock
from uuid import uuid4
from flask import Flask, jsonify, render_template, request, url_for
from PIL import Image, ImageOps, UnidentifiedImageError
from envelhecimento import gerar_imagem, PERFIS, FotoInadequada

BASE = Path(__file__).resolve().parent
app = Flask(__name__)
app.config.update(MAX_CONTENT_LENGTH=10 * 1024 * 1024, UPLOAD_FOLDER=BASE / 'static' / 'uploads')
Image.MAX_IMAGE_PIXELS = 20_000_000
processamento = Lock()
cache_resultados = OrderedDict()
ENDERECO_LOCAL = 'http://127.0.0.1:5000'


def site_ja_aberto():
    """Reconhece o projeto sem encerrar outro serviço que use a mesma porta."""
    from urllib.request import urlopen
    from urllib.error import URLError
    try:
        with urlopen(ENDERECO_LOCAL, timeout=2) as resposta:
            return '<title>Projeto Esperança | Envelhecimento facial</title>'.encode('utf-8') in resposta.read(65536)
    except (URLError, OSError):
        return False

@app.get('/')
def index():
    return render_template('index.html')

@app.errorhandler(413)
def arquivo_grande(error):
    return jsonify(erro='A foto deve ter no máximo 10 MB.'), 413

@app.post('/api/envelhecer')
def envelhecer():
    arquivo = request.files.get('file')
    if not arquivo or not arquivo.filename:
        return jsonify(erro='Selecione uma foto.'), 400
    perfil = request.form.get('perfil', 'rapido')
    if perfil not in PERFIS:
        return jsonify(erro='Escolha um modo de geração válido.'), 400
    try:
        idade = int(request.form.get('idade', '70'))
        if not 40 <= idade <= 90:
            raise ValueError
    except ValueError:
        return jsonify(erro='Escolha uma idade entre 40 e 90 anos.'), 400
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(arquivo.stream) as imagem:
                if imagem.format not in {'JPEG', 'PNG', 'WEBP'}:
                    return jsonify(erro='Envie uma foto JPG, PNG ou WebP.'), 400
                foto = ImageOps.exif_transpose(imagem).convert('RGB')
                foto.thumbnail((1024, 1024))
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        return jsonify(erro='Não consegui abrir essa foto. Use outro JPG, PNG ou WebP.'), 400
    if not processamento.acquire(blocking=False):
        return jsonify(erro='Já há uma foto sendo processada. Tente novamente em instantes.'), 409
    criados = []
    try:
        pasta = Path(app.config['UPLOAD_FOLDER'])
        chave = (hashlib.sha256(foto.tobytes()).hexdigest(), foto.size, idade, perfil)
        anterior = cache_resultados.get(chave)
        if anterior and all((pasta / nome).is_file() for nome in anterior):
            cache_resultados.move_to_end(chave)
            return jsonify(original=url_for('static', filename=f'uploads/{anterior[0]}'),
                           aged=url_for('static', filename=f'uploads/{anterior[1]}'), reutilizado=True)
        resultado = gerar_imagem(foto, idade, perfil)
        pasta.mkdir(parents=True, exist_ok=True)
        identificador = uuid4().hex
        original = pasta / f'{identificador}_original.png'
        saida = pasta / f'{identificador}_envelhecida.png'
        criados = [original, saida]
        foto.save(original)
        resultado.save(saida)
        cache_resultados[chave] = (original.name, saida.name)
        cache_resultados.move_to_end(chave)
        while len(cache_resultados) > 32:
            cache_resultados.popitem(last=False)
        return jsonify(original=url_for('static', filename=f'uploads/{original.name}'),
                       aged=url_for('static', filename=f'uploads/{saida.name}'), reutilizado=False)
    except FotoInadequada as error:
        return jsonify(erro=str(error)), 400
    except Exception as error:
        for caminho in criados:
            caminho.unlink(missing_ok=True)
        app.logger.exception('Falha ao gerar a imagem')
        mensagem = str(error) if isinstance(error, RuntimeError) else 'Não foi possível gerar a foto. Confira o terminal do Python.'
        return jsonify(erro=mensagem), 503
    finally:
        processamento.release()

if __name__ == '__main__':
    import os
    import socket
    import time
    from envelhecimento import preparar_apresentacao
    from waitress import create_server
    # Reserva a porta antes de carregar outra cópia do modelo na GPU.
    conexao = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
            conexao.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        conexao.bind((os.environ.get('HOST', '127.0.0.1'), 5000))
        servidor = create_server(app, sockets=[conexao], threads=4)
    except OSError as error:
        conexao.close()
        if site_ja_aberto():
            import webbrowser
            print(f'O Projeto Esperança já está rodando. Abra {ENDERECO_LOCAL}', flush=True)
            webbrowser.open(ENDERECO_LOCAL)
            raise SystemExit(0)
        raise SystemExit('A porta 5000 está ocupada. Se acabou de iniciar a IA, aguarde a mensagem "IA pronta" na primeira janela. Caso contrário, confira qual programa usa essa porta.') from error
    try:
        print('Preparando a IA para a feira. Aguarde antes de abrir o site...', flush=True)
        inicio = time.perf_counter()
        preparar_apresentacao()
        print(f'IA pronta em {time.perf_counter() - inicio:.0f}s. Abra http://127.0.0.1:5000', flush=True)
        import webbrowser
        webbrowser.open(ENDERECO_LOCAL)
        servidor.run()
    finally:
        servidor.close()
