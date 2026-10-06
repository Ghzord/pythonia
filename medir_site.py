"""Mede uma geração nova e uma repetição com o servidor já preparado."""
import argparse
import json
import time
from pathlib import Path
import requests

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('foto', type=Path)
    parser.add_argument('--perfil', choices=['rapido', 'detalhado'], default='rapido')
    parser.add_argument('--idade', type=int, default=70)
    args = parser.parse_args()
    resultados = []
    for tentativa in range(2):
        inicio = time.perf_counter()
        with args.foto.open('rb') as foto:
            resposta = requests.post('http://127.0.0.1:5000/api/envelhecer',
                files={'file': (args.foto.name, foto)},
                data={'idade': args.idade, 'perfil': args.perfil}, timeout=(10, 600))
        dados = resposta.json()
        if not resposta.ok:
            raise SystemExit(dados.get('erro', resposta.text))
        medicao = dict(tentativa=tentativa + 1, perfil=args.perfil, idade=args.idade,
                       segundos=round(time.perf_counter() - inicio, 2), **dados)
        resultados.append(medicao)
        print(json.dumps(medicao), flush=True)
    pasta = Path('resultados_benchmark')
    pasta.mkdir(exist_ok=True)
    (pasta / f'site_{args.perfil}_{args.idade}.json').write_text(
        json.dumps(resultados, indent=2), encoding='utf-8')
