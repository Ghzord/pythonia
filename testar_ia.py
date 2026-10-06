"""Gera uma foto pelo mesmo endpoint do site e mede o tempo no computador."""
import argparse
import time
from pathlib import Path
from app import app

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Teste completo da IA local, sem navegador.')
    parser.add_argument('foto', type=Path)
    parser.add_argument('--idade', type=int, default=70)
    parser.add_argument('--perfil', choices=['rapido', 'detalhado'], default='rapido')
    args = parser.parse_args()
    inicio = time.perf_counter()
    with args.foto.open('rb') as arquivo, app.test_client() as client:
        resposta = client.post('/api/envelhecer', data={
            'file': (arquivo, args.foto.name), 'idade': str(args.idade), 'perfil': args.perfil})
    if resposta.status_code != 200:
        raise SystemExit(resposta.get_json()['erro'])
    print(f'Tempo total: {time.perf_counter() - inicio:.1f} segundos')
    print('Resultado:', Path(__file__).resolve().parent / resposta.get_json()['aged'].lstrip('/'))
