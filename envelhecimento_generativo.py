"""Edição com InstructPix2Pix, usando somente os pesos baixados no projeto."""
from pathlib import Path
import os
from PIL import Image, ImageOps

MODELO_ID = 'timbrooks/instruct-pix2pix'
MODELO_LOCAL = Path(__file__).resolve().parent / 'modelos' / 'instruct-pix2pix'
_pipeline = None
# Resolução máxima e quantidade de passos de cada perfil.
PERFIS = {
    'rapido': (256, 8),
    'detalhado': (320, 10),
}

def carregar_modelo():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    if not (MODELO_LOCAL / 'model_index.json').exists():
        raise RuntimeError('O modelo ainda não foi baixado. Execute python preparar_modelo.py com internet antes da apresentação.')
    try:
        import torch
        from diffusers import EulerAncestralDiscreteScheduler
        from pipeline_local import PipelineLocal, VAEEstavel
    except ImportError as error:
        raise RuntimeError('Instale as dependências com python -m pip install -r requirements.txt.') from error
    try:
        cuda = torch.cuda.is_available()
        dtype = torch.float16 if cuda and os.environ.get('IA_PRECISAO') != 'total' else torch.float32
        # Nesta GTX 16, convoluções cuDNN em fp16 retornaram NaN.
        # As convoluções nativas preservam fp16 sem esse erro no teste local.
        if cuda and 'GTX 16' in torch.cuda.get_device_name(0).upper() and dtype == torch.float16:
            torch.backends.cudnn.enabled = False
        vae = VAEEstavel.from_pretrained(
            str(MODELO_LOCAL), subfolder='vae', torch_dtype=torch.float32,
            variant='fp16', use_safetensors=True, local_files_only=True)
        pipe = PipelineLocal.from_pretrained(
            str(MODELO_LOCAL), torch_dtype=dtype,
            vae=vae, variant='fp16', use_safetensors=True, local_files_only=True)
        if pipe.safety_checker is not None:
            pipe.safety_checker.to(dtype=torch.float32)
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.enable_attention_slicing()
        pipe.vae.enable_slicing()
        if cuda:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to('cpu')
        _pipeline = pipe
        return pipe
    except Exception as error:
        raise RuntimeError('Não consegui carregar a IA. Confira as dependências e execute preparar_modelo.py novamente. Detalhes no terminal.') from error

def gerar_imagem(foto: Image.Image, idade: int, perfil: str = 'rapido') -> Image.Image:
    if perfil not in PERFIS:
        raise ValueError('Modo de geração inválido.')
    tamanho, passos = PERFIS[perfil]
    return _processar(foto, idade, tamanho, passos)


def preparar_apresentacao():
    """Carrega os pesos e inicializa a GPU antes de abrir o atendimento."""
    _processar(Image.new('RGB', (256, 256), (128, 128, 128)), 70, 256, 1)


def _processar(foto, idade, tamanho, passos):
    pipe = carregar_modelo()
    import torch

    def validar_latentes(_pipe, _passo, _tempo, dados):
        if not torch.isfinite(dados['latents']).all():
            raise RuntimeError('A GPU produziu valores inválidos. Use IA_PRECISAO=total e reinicie o aplicativo.')
        return dados

    entrada = ImageOps.contain(foto, (tamanho, tamanho))
    largura, altura = entrada.size
    entrada = entrada.resize((max(8, largura // 8 * 8), max(8, altura // 8 * 8)), Image.Resampling.LANCZOS)
    try:
        with torch.inference_mode():
            resposta = pipe(
                prompt=f'Add realistic fine skin wrinkles to this same face at age {idade}. Preserve the same face, facial proportions, expression, eyes, mouth and lighting.',
                negative_prompt='cartoon, painting, distorted face, blurry, deformed eyes',
                image=entrada, num_inference_steps=passos, guidance_scale=5.0,
                image_guidance_scale=2.2, generator=torch.Generator('cpu').manual_seed(42),
                callback_on_step_end=validar_latentes)
        if resposta.nsfw_content_detected and any(resposta.nsfw_content_detected):
            raise RuntimeError('O modelo não liberou esse resultado. Tente outra foto.')
        return resposta.images[0]
    except torch.OutOfMemoryError as error:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise RuntimeError('Faltou memória para gerar a foto. Feche jogos e outros programas e tente novamente.') from error
