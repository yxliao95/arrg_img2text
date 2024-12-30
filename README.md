# Document

## Install

We are using CUDA/12.4

- `conda create -n arrg_reinforce python=3.11 -y`
    - conda create --name arrg_img2text --clone arrg_reinforce
    - conda remove --name arrg_reinforce --all
- `conda activate arrg_reinforce`
- `conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia -y`
- `pip install transformers` (==4.47.1)
- `pip install datasets` (==3.2.0)
- `pip install imagehash` (==4.3.1)
<!-- - `pip install tensorboard` (==2.18.0) -->

If CPU only

- `pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cpu`
- `pip install 'transformers[torch]'` (==4.47.1)

Download LLM:

- `pip install --upgrade huggingface_hub`
- `huggingface-cli login --token your_access_token_here`
    - Get the access token from: https://huggingface.co/settings/tokens
- `huggingface-cli download meta-llama/Llama-3.2-1B --local-dir /your/specific/path`