# Projeto Esperança

O site mantém a foto como base e aplica dois tratamentos: grisalho no cabelo segmentado e variações de textura na pele. Um recorte do rosto passa pelo InstructPix2Pix para gerar uma referência de textura. A composição transfere apenas detalhes de luminosidade dessa referência, sem colar outro rosto na foto. Olhos, boca e nariz têm regiões protegidas.

## Usar na feira

Execute `iniciar.bat`, espere **IA pronta** e abra http://127.0.0.1:5000. Mantenha o terminal aberto para reutilizar os modelos carregados. Se o aplicativo já estiver rodando, o iniciador abre o site existente. Ao atualizar o código, feche o servidor anterior e inicie novamente.

Use uma foto frontal, nítida, com uma pessoa e rosto descoberto. A idade sugerida controla a intensidade da textura e a proporção de grisalho. Aos 50 o grisalho é parcial; em idades maiores ele aumenta. Isso é uma escolha artística: o sistema não conhece a idade atual e não prevê quando uma pessoa terá cabelos brancos. Também não rejuvenesce uma pessoa mais velha.

As linhas desenhadas do filtro anterior não são usadas. A textura é limitada para preservar a aparência da pessoa; por isso o resultado pode ser discreto. Cabelos finos, fundos parecidos com cabelo, acessórios e iluminação difícil podem causar erros de segmentação. O efeito não é uma previsão biológica nem tem qualidade fotográfica garantida.

Esta versão usa modelos maiores e demora mais que o filtro de linhas. Uma GPU NVIDIA ajuda; na CPU a geração pode ser muito lenta. O primeiro carregamento também demora. Fotos repetidas com as mesmas opções reutilizam o resultado salvo enquanto o servidor estiver aberto. O site informa quando isso acontece.

Na comparação local de 50 e 70 anos em 05/10/2026, a primeira geração levou 145 s incluindo carregamento, e a segunda levou 68 s com os modelos em memória. O servidor iniciou em 39 s no teste seguinte. São medições da GTX 1660 SUPER com a foto de exemplo, não um tempo garantido para toda foto. A composição passou por inspeção visual e os 12 testes automatizados passaram.

## Instalar em outro computador

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe preparar_modelo.py
.\.venv\Scripts\python.exe app.py
```

Para usar CUDA, instale o PyTorch compatível com a GPU seguindo https://pytorch.org/get-started/locally/. Neste computador foi utilizada a versão CUDA 12.6. O comando abaixo verifica se a GPU foi reconhecida:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

O script de preparação baixa YuNet, a segmentação `jonathandinu/face-parsing` e os componentes do InstructPix2Pix. Depois de preparados, são carregados exclusivamente dos arquivos locais em `modelos`. Fotos não são enviadas a serviços externos; as entradas e saídas ficam em `static/uploads`.

Na GTX 1660, as convoluções cuDNN em float16 produziram valores inválidos nos testes. O código usa convoluções nativas, com VAE e verificador de conteúdo em float32. A verificação de conteúdo continua ativa. `IA_PRECISAO=total` permite usar precisão completa se outro equipamento apresentar erros numéricos, com maior consumo de memória.

## Verificação e integração

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_app test_realismo
.\.venv\Scripts\python.exe validar_realismo.py "caminho\foto.jpg"
```

O segundo comando salva uma comparação de 50 e 70 anos em `resultados_benchmark/realismo`. `medir_site.py` mede a rota HTTP com o site já aberto. Os testes verificam a preservação de regiões protegidas, a progressão do grisalho, uploads inválidos e o cache.

O formulário envia `file` e `idade` para `POST /api/envelhecer`. A resposta contém `original`, `aged` e `reutilizado`. Fotos inválidas retornam HTTP 400 com o campo `erro`. São aceitos JPEG, PNG e WebP de até 10 MB, redimensionados proporcionalmente para até 1024 pixels.

## Modelos utilizados

- [YuNet / OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet): detecção facial; licença em `modelos/YUNET_LICENSE.txt`.
- [Face Parsing, de Jonathan Dinu](https://huggingface.co/jonathandinu/face-parsing): separação de cabelo, pele e demais regiões. A ficha do modelo indica uso não comercial para pesquisa e educação; cópia em `modelos/face-parsing/README.md`.
- [InstructPix2Pix](https://huggingface.co/timbrooks/instruct-pix2pix): referência de textura gerada por instruções. O site compõe os detalhes sobre a foto original.
