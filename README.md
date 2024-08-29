# CacheBlend (Under Construction): 

This is the code repo for [CacheBlend: Fast Large Language Model Serving with Cached Knowledge Fusion](https://arxiv.org/pdf/2405.16444). The current implementation is based on [vLLM](https://github.com/vllm-project/vllm/tree/main).

## Installation
To install CacheBlend depenencies
```
git clone git@github.com:YaoJiayi/CacheBlend.git
cd CacheBlend/vllm_blend
pip install -e .
cd ..
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
