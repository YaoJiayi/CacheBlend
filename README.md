# CacheBlend (Under Construction): 

This is the code repo for [CacheBlend: Fast Large Language Model Serving with Cached Knowledge Fusion](https://arxiv.org/pdf/2405.16444). The current implementation is based on [vLLM](https://github.com/vllm-project/vllm/tree/main).

## Installation
`Python>=3.9` and `CUDA >= 12.1` are required. An Nvidia GPU with `>=40 GB` memory is recommended.
To install CacheBlend depenencies:
```
git clone git@github.com:YaoJiayi/CacheBlend.git
cd CacheBlend/vllm_blend
pip install -e .
cd ..
pip install -r requirements.txt
```


## Example run
### Run LLM inference with CacheBlend
```
python example/blend.py
```

## Run Musique dataset
### Compare LLM inference with CacheBlend and normal prefill
```
python example/blend_musique.py
```
## References
