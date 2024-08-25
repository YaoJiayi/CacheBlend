from vllm import LLM, SamplingParams
import torch
import json
import numpy as np
from transformers import AutoTokenizer
from utils import load_dataset, normalize_question, build_qa_prompt, compute_f1
from pathlib import Path

eval_dataset = load_dataset("inputs/musique_s.json")

llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2", gpu_memory_utilization=0.5,
          #tokenizer=tokenizer,
          )
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
llm.set_tokenizer(tokenizer)

prefix_prompt = "You will be asked a question after reading several passages. Please directly answer the question based on the given passages. Do NOT repeat the question. The answer should be within 5 words..\nPassages:\n"
query_prompt = "\n\nAnswer the question directly based on the given passages. Do NOT repeat the question. The answer should be within 5 words. \nQuestion:"

ttft_blend = []
ttft_full = []
f1_blend = []
f1_full = []

#TODO (Jiayi): fix filler tokens at the begining or pass in tokenizer
for ex in eval_dataset:
    answers = ex["answers"]
    doc_prompts, q_prompt = build_qa_prompt(ex, query_prompt)
    doc_chunk_ids = [tokenizer.encode(doc)[1:] for doc in doc_prompts]
    q_ids = tokenizer.encode(q_prompt)[1:]

    #import pdb
    #pdb.set_trace()
    
    #while len(list(chain.from_iterable(doc_chunk_ids))) > max_ctx_len:
    #    del_idx = len(doc_chunk_ids)-1
    #    del doc_chunk_ids[del_idx]
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0, max_tokens=1)

    # Create an tokenizer and LLM.
    cache_fuse_metadata = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata
    cache_fuse_metadata['collect'] = False
    cache_fuse_metadata['check'] = False

    #s_start_full = [733, 4138, 28793] + tokenizer.encode(prefix_prompt)[1:]
    s_start_full = [733, 16289, 28793] + tokenizer.encode(prefix_prompt)[1:]
    s_start_len = len(s_start_full) + 1

    #s_start = [518, 25580, 29962]
    s_start = []
    s_start_1_len = len(s_start) + 1

    #s_end = [518, 29914, 25580, 29962]
    s_end = [733, 28748, 16289, 28793]
    s_end_len = len(s_end)
    old_kvs = []

    doc_chunk_ids = [s_start+chunk_ids for chunk_ids in doc_chunk_ids]
    doc_chunk_ids = [s_start_full] + doc_chunk_ids
    doc_chunk_ids = doc_chunk_ids + [s_start+q_ids+s_end]

    last_len = len([q_ids+s_end])

    cache_fuse_metadata['collect'] = True
    cache_fuse_metadata["check"] = False
    num_layer = 32
    chunk_past_key_values = []
    
    # Concatenate old KVs
    for i in range(len(doc_chunk_ids)):
        prompts = [tokenizer.decode(doc_chunk_ids[i])]
        llm.generate(prompts, sampling_params)
        
        llm_layers = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.layers
        for j in range(num_layer):
            past_key_values = llm_layers[j].self_attn.hack_kv
            if i == 0:
                temp_k = past_key_values[0][:s_start_len].clone() # do not chage with s_start_1
                temp_v = past_key_values[1][:s_start_len].clone()
            else:
                temp_k = past_key_values[0][s_start_1_len:len(doc_chunk_ids[i])+1].clone()
                temp_v = past_key_values[1][s_start_1_len:len(doc_chunk_ids[i])+1].clone()    

            if i == 0:
                chunk_past_key_values.append([temp_k, temp_v])
            else:
                #pdb.set_trace()
                chunk_past_key_values[j][0] = torch.cat((chunk_past_key_values[j][0],temp_k), dim=0)
                chunk_past_key_values[j][1] = torch.cat((chunk_past_key_values[j][1],temp_v), dim=0)
        #print(temp_k.shape[0])
        llm.llm_engine.model_executor.driver_worker.model_runner.model.model.old_kvs = chunk_past_key_values
        
    input_ids = []

    for i in range(len(doc_chunk_ids)):
        if i == 0:
            temp_ids = doc_chunk_ids[i]
        else:
            temp_ids = doc_chunk_ids[i][s_start_1_len-1:]
        input_ids += temp_ids
        
    input_prompt = tokenizer.decode(input_ids)
    
    sampling_params = SamplingParams(temperature=0, max_tokens=32)
    cache_fuse_metadata["check"] = True
    cache_fuse_metadata['collect'] = False
    cache_fuse_metadata['suffix_len'] = last_len
    
    output = llm.generate([input_prompt], sampling_params)
    res = output[0].outputs[0].text
    print(f"Cached generation: {res}")
    ttft = output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time
    print(f"TTFT with cache: {ttft}")
    ttft_blend.append(ttft)
    f1 = max([compute_f1(res, answer, tokenizer) for answer in answers])
    f1_blend.append(f1)

    
    sampling_params = SamplingParams(temperature=0, max_tokens=32)
    cache_fuse_metadata["check"] = False
    cache_fuse_metadata['collect'] = False
    output = llm.generate([input_prompt], sampling_params)
    res = output[0].outputs[0].text
    print(f"Normal generation: {res}")
    ttft = output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time
    print(f"TTFT with full prefill: {ttft}")
    ttft_full.append(ttft)
    f1 = max([compute_f1(res, answer, tokenizer) for answer in answers])
    f1_full.append(f1)
    print("------------")

print("---------------Result Summary---------------------")
print(f"TTFT with cache: {np.mean(ttft_blend)}")
print(f"TTFT with full prefill: {np.mean(ttft_full)}")
print(f"F1 with cache: {np.mean(f1_blend)}")
print(f"F1 with full prefill: {np.mean(f1_full)}")
