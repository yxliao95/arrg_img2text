# Document

## Install

We are using CUDA/12.4

- `conda create -n arrg_img2text python=3.11 -y`
- `conda activate arrg_img2text`
- `conda install -y pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia`
- `pip install transformers` (==4.47.1)
- `pip install datasets` (==3.2.0)
- `pip install imagehash` (==4.3.1)
    - `pip install transformers==4.47.1 datasets==3.2.0 imagehash==4.3.1`
- `pip install tensorboard` (==2.18.0)
- `pip install accelerate` (==1.2.1)
- `pip install mlflow` (==2.19.0)

If CPU only

- `pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cpu`
- `pip install 'transformers[torch]'` (==4.47.1)

If Arcca
- `conda install -y pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia`
- `pip install "numpy<2"`

Download LLM:

- `pip install --upgrade huggingface_hub`
- `huggingface-cli login --token your_access_token_here`
    - Get the access token from: https://huggingface.co/settings/tokens
- `huggingface-cli download meta-llama/Llama-3.2-1B --local-dir /your/specific/path`