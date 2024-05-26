# CacheBlend: 

This is an anonymous code repo (under construction) for [CacheBlend: Fast Large Language Model Serving with Cached Knowledge Fusion]() (in submission, EuroSys'25). The current implementation is based on [vLLM](https://github.com/vllm-project/vllm/tree/main).

## Installation
To install CacheBlend depenencies
```
git clone git@github.com:YaoJiayi/CacheBlend.git
cd CacheBlend
pip install -e .
```


## Example run
### Run LLM inference with CacheBlend
Step 1: To independently generate the KV caches for multiple text segments
```
python examples/preprocess.py --text_path inputs/1.json --cache_path <PATH OF THE KV CACHE>
```


Step 2: To run LLM inference w/ CacheFuse
```
python examples/fuse_gen.py --cache_path <PATH OF THE KV CACHE>
```

### Run normal LLM inference
To run LLM inference w/o CacheBlend
```
python examples/normal_gen.py
```
## References
