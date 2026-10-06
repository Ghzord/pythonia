"""Testa o contrato HTML/Python sem baixar ou executar o modelo pesado."""
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
import app as modulo


def upload(nome='foto.png', idade='70'):
    dados = io.BytesIO()
    Image.new('RGB', (64, 80), 'white').save(dados, format='PNG')
    dados.seek(0)
    return {'file': (dados, nome), 'idade': idade}


class Integracao(unittest.TestCase):
    def setUp(self):
        modulo.cache_resultados.clear()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = patch.dict(modulo.app.config, TESTING=True, UPLOAD_FOLDER=Path(self.temp.name))
        self.config.start()
        self.addCleanup(self.config.stop)
        self.client = modulo.app.test_client()

    def test_pagina(self):
        resposta = self.client.get('/')
        self.assertEqual(resposta.status_code, 200)
        self.assertIn(b'/api/envelhecer', resposta.data)

    def test_upload_retorna_imagens_e_ignora_nome_do_usuario(self):
        with patch.object(modulo, 'gerar_imagem', return_value=Image.new('RGB', (64, 80))) as gerar:
            resposta = self.client.post('/api/envelhecer', data=upload('../../fora.png'))
        self.assertEqual(resposta.status_code, 200)
        self.assertEqual(gerar.call_args.args[1], 70)
        for campo in ('original', 'aged'):
            nome = resposta.json[campo].rsplit('/', 1)[1]
            self.assertNotIn('fora', nome)
            with Image.open(Path(self.temp.name) / nome) as imagem:
                self.assertEqual(imagem.format, 'PNG')

    def test_entradas_invalidas_nao_executam_modelo(self):
        with patch.object(modulo, 'gerar_imagem') as gerar:
            self.assertEqual(self.client.post('/api/envelhecer').status_code, 400)
            self.assertEqual(self.client.post('/api/envelhecer', data=upload(idade='abc')).status_code, 400)
            self.assertEqual(self.client.post('/api/envelhecer', data=upload(idade='150')).status_code, 400)
            self.assertEqual(self.client.post('/api/envelhecer', data={'file': (io.BytesIO(b'invalido'), 'foto.png')}).status_code, 400)
            gerar.assert_not_called()

    def test_limite_upload(self):
        with patch.dict(modulo.app.config, MAX_CONTENT_LENGTH=100):
            resposta = self.client.post('/api/envelhecer', data=upload())
        self.assertEqual(resposta.status_code, 413)
        self.assertIn('erro', resposta.json)

    def test_cache_reutiliza_e_distingue_idade_e_perfil(self):
        with patch.object(modulo, 'gerar_imagem', return_value=Image.new('RGB', (64, 80))) as gerar:
            primeira = self.client.post('/api/envelhecer', data=upload())
            segunda = self.client.post('/api/envelhecer', data=upload('outro-nome.png'))
            self.assertEqual(gerar.call_count, 1)
            self.assertFalse(primeira.json['reutilizado'])
            self.assertTrue(segunda.json['reutilizado'])
            self.assertEqual(primeira.json['aged'], segunda.json['aged'])
            self.client.post('/api/envelhecer', data=upload(idade='80'))
            dados = upload()
            dados['perfil'] = 'detalhado'
            self.client.post('/api/envelhecer', data=dados)
            self.assertEqual(gerar.call_count, 3)

    def test_cache_regenera_quando_arquivo_foi_apagado(self):
        with patch.object(modulo, 'gerar_imagem', return_value=Image.new('RGB', (64, 80))) as gerar:
            primeira = self.client.post('/api/envelhecer', data=upload())
            (Path(self.temp.name) / primeira.json['aged'].rsplit('/', 1)[1]).unlink()
            segunda = self.client.post('/api/envelhecer', data=upload())
            self.assertFalse(segunda.json['reutilizado'])
            self.assertEqual(gerar.call_count, 2)

    def test_perfil_invalido(self):
        dados = upload()
        dados['perfil'] = 'invalido'
        with patch.object(modulo, 'gerar_imagem') as gerar:
            self.assertEqual(self.client.post('/api/envelhecer', data=dados).status_code, 400)
            gerar.assert_not_called()

    def test_rosto_inadequado_exibe_mensagem(self):
        with patch.object(modulo, 'gerar_imagem', side_effect=modulo.FotoInadequada('Use uma foto de frente.')):
            resposta = self.client.post('/api/envelhecer', data=upload())
        self.assertEqual(resposta.status_code, 400)
        self.assertEqual(resposta.json['erro'], 'Use uma foto de frente.')
        self.assertFalse(modulo.processamento.locked())

    def test_ocupado(self):
        modulo.processamento.acquire()
        try:
            self.assertEqual(self.client.post('/api/envelhecer', data=upload()).status_code, 409)
        finally:
            modulo.processamento.release()

    def test_falha_libera_proxima_geracao(self):
        with patch.object(modulo, 'gerar_imagem', side_effect=RuntimeError('Modelo indisponível')):
            resposta = self.client.post('/api/envelhecer', data=upload())
        self.assertEqual(resposta.status_code, 503)
        self.assertFalse(modulo.processamento.locked())
        self.assertEqual(list(Path(self.temp.name).iterdir()), [])
        self.assertEqual(resposta.json['erro'], 'Modelo indisponível')

if __name__ == '__main__':
    unittest.main()
