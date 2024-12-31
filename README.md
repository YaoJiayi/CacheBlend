# CacheBlend (Under Construction): 

This is the code repo for [CacheBlend: Fast Large Language Model Serving with Cached Knowledge Fusion](https://arxiv.org/pdf/2405.16444). The current implementation is based on [vLLM](https://github.com/vllm-project/vllm/tree/main).


### The newest updates will always be at [LMCache](https://github.com/LMCache/LMCache). Stay tuned !!!

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
To run datasets other than musique, please replace `musique` with `samsum` or `wikimqa` in the above command.
## References
```
@misc{yao2024cacheblendfastlargelanguage,
      title={CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion}, 
      author={Jiayi Yao and Hanchen Li and Yuhan Liu and Siddhant Ray and Yihua Cheng and Qizheng Zhang and Kuntai Du and Shan Lu and Junchen Jiang},
      year={2024},
      eprint={2405.16444},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2405.16444}, 
}
```
