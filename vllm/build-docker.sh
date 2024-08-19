lmcache_commit_id=$(git ls-remote https://github.com/LMCache/demo refs/heads/blend-demo | cut -f 1)
DOCKER_BUILDKIT=1 docker build --build-arg LMCACHE_COMMIT_ID=$lmcache_commit_id . --target vllm-lmcache --tag vllm-lmcache:blend --build-arg max_jobs=32 --build-arg nvcc_threads=32 --platform linux/amd64 #--no-cache
#VLLM_NCCL_SO_PATH=/home/yihua/libnccl.so.2.18.1 DOCKER_BUILDKIT=1 docker build . --target vllm-lmcache --tag vllm-lmcache:test --build-arg max_jobs=32 --build-arg nvcc_threads=32 --platform linux/amd64
