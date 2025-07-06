# Document

## Install

We are using CUDA/12.4

- `conda create -n arrg_img2text python=3.11 -y`
- `conda activate arrg_img2text`
- `conda install -y pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia`
- `pip install transformers` (==4.49.0)
- `pip install datasets` (==3.2.0)
- `pip install imagehash` (==4.3.1)
- `pip install tensorboard` (==2.18.0)
- `pip install accelerate` (==1.5.2)
- `pip install mlflow` (==2.19.0)
- `pip install sentencepiece` (==0.2.0)
- `pip install peft` (==0.15.1)

vilmedic-scorers requirements (follows the error section after installing these libraries)

- `pip install bert-score` (==0.3.13)
- `pip install rouge-score` (==0.1.2)
- `pip install f1chexbert` (==0.0.2)
- `pip install radgraph` (==0.1.14)
- `pip install stanza` (==1.10.1)
- `pip install gdown` (==5.2.0)
- `pip install base58` (==2.1.1)

If CPU only

- `pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cpu`
- `pip install 'transformers[torch]'` (==4.47.1)

If Arcca (cuda 11.7)

- `conda install -y pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia`
- `pip install "numpy<2"`

Download LLM:

- `pip install --upgrade huggingface_hub`
- `huggingface-cli login --token your_access_token_here`
    - Get the access token from: https://huggingface.co/settings/tokens
- `huggingface-cli download meta-llama/Llama-3.2-1B --local-dir /your/specific/path/Llama-3.2-1B`

## Launch MLflow
nohup mlflow server --host localhost --port 6026 --backend-store-uri file:/home/yuxiang/liao/workspace/arrg_img2text/outputs/mlruns > /dev/null 2>&1 &

## Error

### Using radgraph library
When using radgraph, you may have this error: `FileNotFoundError: [Errno 2] No such file or directory: '.../.cache/radgraph/0.1.2/radgraph-xl.tar.gz'`. Our dirty trick is changing the `self.model_path` in the script.
 - go to `...envs/arrg_img2text/lib/python3.11/site-packages/radgraph/radgraph.py`
 - replace the old code block with the new one:

```python
##########################
# old; line 21, 73-81
##########################
from radgraph.utils import download_model

try:
    if not os.path.exists(self.model_path):
        download_model(
            repo_id="StanfordAIMI/RRG_scorers",
            cache_dir=CACHE_DIR,
            filename=MODEL_MAPPING[model_type],
        )
except Exception as e:
    print("Model download error", e)

##########################
# new
##########################
from huggingface_hub import hf_hub_download

self.model_path = hf_hub_download(repo_id="StanfordAIMI/RRG_scorers", filename=MODEL_MAPPING[model_type], cache_dir=CACHE_DIR)
```

You may see error `AttributeError: add_special_tokens conflicts with the method add_special_tokens in BertTokenizer`. This is caused by the `allennlp` library inside the `radgrpah` library. Where `allennlp` is incompatiable to the latest `transformers` library. Our dirty trick:
 - go to file `.../envs/arrg_img2text/lib/python3.11/site-packages/radgraph/allennlp/data/tokenizers/pretrained_transformer_tokenizer.py`
 - remove the param `add_special_tokens=False` from the method call (line 80)
 - remove the param `add_special_tokens=True` from the method call (line 122)


### Using f1chexbert library

You may have this error: `FileNotFoundError: [Errno 2] No such file or directory: '.../.cache/chexbert/chexbert.pth`. Our dirty trick is changing the `self.model_path` in the script.
 - go to `...envs/arrg_img2text/lib/python3.11/site-packages/f1chexbert/f1chexbert.py`
 - replace the old code block with the new one:

```python
##########################
# old; line 144-146
##########################
checkpoint = os.path.join(CACHE_DIR, "chexbert.pth")
if not os.path.exists(checkpoint):
    download_model(repo_id='StanfordAIMI/RRG_scorers', cache_dir=CACHE_DIR, filename="chexbert.pth")

##########################
# new
##########################
checkpoint = hf_hub_download(repo_id="StanfordAIMI/RRG_scorers", cache_dir=CACHE_DIR, filename="chexbert.pth")
```

## Test vilmedic-scorers

Run this script: `path_to_this_repo/arrg_img2text/scorers/scores.py`. Dont forget to change `sys.path.append("path_to_this_repo/arrg_img2text")` on top of the script.

