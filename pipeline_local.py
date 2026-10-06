"""Mantém a conversão de imagens em float32 e permite UNet em float16."""
import torch
from diffusers import AutoencoderKL, StableDiffusionInstructPix2PixPipeline


class VAEEstavel(AutoencoderKL):
    def decode(self, z, *args, **kwargs):
        resultado = super().decode(z.to(dtype=self.dtype), *args, **kwargs)
        imagem = resultado[0] if isinstance(resultado, tuple) else resultado.sample
        if not torch.isfinite(imagem).all():
            raise RuntimeError('A conversão do resultado produziu valores inválidos.')
        return resultado


class PipelineLocal(StableDiffusionInstructPix2PixPipeline):
    def prepare_image_latents(self, image, batch_size, num_images_per_prompt, dtype,
                             device, do_classifier_free_guidance, generator=None):
        latentes = super().prepare_image_latents(
            image, batch_size, num_images_per_prompt, self.vae.dtype,
            device, do_classifier_free_guidance, generator)
        if not torch.isfinite(latentes).all():
            raise RuntimeError('A conversão da foto produziu valores inválidos.')
        return latentes.to(dtype=dtype)

    def run_safety_checker(self, image, device, dtype):
        checker_dtype = self.safety_checker.dtype if self.safety_checker is not None else dtype
        return super().run_safety_checker(image, device, checker_dtype)
