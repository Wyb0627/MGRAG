# MGRAG
The official code repository of the paper "MGRAG: A Graph-based Multimodal Retrieval-augmented Generation System with  Large Language Models"


## Requirment
Before running the MGRAG, you need to set up [vLLM](https://github.com/vllm-project/vllm) services with [LMCache](https://github.com/LMCache/LMCache).

The MGRAG runs with Python 3.10.16, and you can install the required package via. requirements.txt

## Run Experiment
Download the dataset [MMQA](https://github.com/allenai/multimodalqa) and [MRAMG](https://github.com/MRAMG-Bench/MRAMG), then place them w.r.t. the directory setting in the dataset.py

The image caption we used for each dataset is placed in the /caption folder.

You can run MGRAG on the MMQA dataset via: 
```
  python run.py --rag anyrag --stop_detect graph --LLM qwen --k 5 \
  --gpu YOUR_GPU_ID --use_vllm --enable_parallel --path_recent_nodes 3 \
  --add_passage_node --use_lmcache --graph_score all --dataset mmqa --enable_blending \
  --use_dpr --enable_graphrag
```
