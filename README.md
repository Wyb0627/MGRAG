# MGRAG
The official code repository of the paper "MGRAG: A Graph-based Multimodal Retrieval-augmented Generation System with  Large Language Models"



For example, you can run MGRAG on the MMQA dataset via: 
```
  python run.py --rag anyrag --stop_detect graph --LLM qwen --k 5 \
  --gpu YOUR_GPU_ID --use_vllm --enable_parallel --path_recent_nodes 3 \
  --add_passage_node --use_lmcache --graph_score all --dataset mmqa --enable_blending \
  --use_dpr --enable_graphrag
```
