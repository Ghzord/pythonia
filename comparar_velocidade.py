"""Compara configurações na GPU local; não altera a configuração do site."""
import argparse
import json
import time
from pathlib import Path

from PIL import Image, ImageOps
from envelhecimento_generativo import carregar_modelo

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('foto', type=Path)
    parser.add_argument('--atencao', choices=['nativa', 'fatiada'], default='fatiada')
    args = parser.parse_args()
    inicio = time.perf_counter()
    pipe = carregar_modelo()
    import torch
    from diffusers import EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
    if args.atencao == 'nativa':
        pipe.disable_attention_slicing()
    else:
        pipe.enable_attention_slicing('max')
    pasta = Path('resultados_benchmark') / args.atencao
    pasta.mkdir(exist_ok=True, parents=True)
    print(f'Carregamento: {time.perf_counter() - inicio:.1f}s', flush=True)
    resultados = []
    with Image.open(args.foto) as foto:
        foto = ImageOps.exif_transpose(foto).convert('RGB')
    for nome, tamanho, passos, scheduler in [
        ('rapido_euler', 256, 8, EulerAncestralDiscreteScheduler),
        ('feira_euler', 320, 10, EulerAncestralDiscreteScheduler),
        ('feira_dpm', 320, 10, DPMSolverMultistepScheduler),
    ]:
        pipe.scheduler = scheduler.from_config(pipe.scheduler.config)
        entrada = ImageOps.contain(foto, (tamanho, tamanho))
        entrada = entrada.resize(tuple(max(8, n // 8 * 8) for n in entrada.size))
        inicio = time.perf_counter()
        with torch.inference_mode():
            saida = pipe(
                prompt='Make this person look 70 years old, with natural facial wrinkles and gray hair. Keep the same person, pose, clothing and background.',
                negative_prompt='cartoon, painting, distorted face, blurry, deformed eyes',
                image=entrada, num_inference_steps=passos, guidance_scale=7.5,
                image_guidance_scale=1.5, generator=torch.Generator('cpu').manual_seed(42))
        duracao = time.perf_counter() - inicio
        saida.images[0].save(pasta / f'{nome}.png')
        resultado = dict(nome=nome, tamanho=tamanho, passos=passos, segundos=round(duracao, 2),
                         bloqueado=bool(saida.nsfw_content_detected and any(saida.nsfw_content_detected)))
        resultados.append(resultado)
        print(json.dumps(resultado), flush=True)
        (pasta / 'tempos.json').write_text(json.dumps(resultados, indent=2), encoding='utf-8')
