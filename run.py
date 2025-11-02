import os

os.environ['LMCACHE_ENABLE_BLENDING'] = 'True'

from argparse import ArgumentParser

import torch.cuda
import asyncio
import sys

parser = ArgumentParser()
parser.add_argument("--gpu", type=str, default='0')
parser.add_argument("--LLM", type=str, default='qwen')
parser.add_argument("--k", type=int, default=5)
parser.add_argument("--dataset", type=str, default='mmqa',
                    choices=['arxiv', 'manual', 'web', 'wiki', 'wit', 'recipe', 'mmqa', 'webqa'])
parser.add_argument("--rag", type=str, default='none')
parser.add_argument("--image_caption", action="store_true")
parser.add_argument("--insert", action="store_true")
parser.add_argument("--LLM_caption", action="store_true")
parser.add_argument("--no_original_support", action="store_true")
parser.add_argument("--stop_detect", type=str, default='none')
parser.add_argument("--threshold", type=float, default=0.7)
parser.add_argument("--graph_matching_mode", type=str, default="path", choices=["pairwise", "path"],
                    help="graph matching方式: pairwise或path")
parser.add_argument("--path_agg_method", type=str, default="mean", choices=["mean", "concat", "weighted"],
                    help="path embedding聚合方式")
parser.add_argument("--path_recent_nodes", type=int, default=2,
                    help="path matching中聚合最近n个节点的数量，默认为2, -1则为全路径match")
parser.add_argument("--enable_parallel", action="store_true",
                    help="启用并行化的graph matching")
parser.add_argument("--parallel_type", type=str, default="thread", choices=["thread", "process"],
                    help="并行化类型：thread(线程)或process(进程)")
parser.add_argument("--max_workers", type=int, default=None,
                    help="并行化时的最大工作线程/进程数，默认为CPU核心数")
parser.add_argument("--parallel_threshold", type=int, default=20,
                    help="启用并行化的最小节点数阈值，默认为20")
# parser.add_argument("--load_index", action="store_true")
parser.add_argument("--from_scratch", action="store_true")
parser.add_argument("--use_vllm", action="store_true")
parser.add_argument("--blend_mode", type=str, default="graphrag")
parser.add_argument("--use_dpr", action="store_true")
parser.add_argument("--input_graph_summary", action="store_true")
parser.add_argument("--add_passage_node", action="store_true")
parser.add_argument("--graph_score", type=str, default="all", choices=["degree", "pagerank", "hybrid", "all"])

# 添加LMCache相关参数
parser.add_argument("--use_lmcache", action="store_true",
                    help="启用LMCache缓存")
parser.add_argument("--max_cache_size", type=float, default=10.0,
                    help="最大缓存大小(GB)")
parser.add_argument("--pagerank_weight", type=float, default=0.7,
                    help="PageRank权重")
parser.add_argument("--recency_weight", type=float, default=0.3,
                    help="最近访问时间权重")
parser.add_argument("--enable_graphrag", action="store_true", default=True,
                    help="启用GraphRAG eviction策略")
parser.add_argument("--enable_blending", action="store_true",
                    help="启用CacheBlend")
parser.add_argument("--blend_recompute_ratio", type=float, default=0.15,
                    help="Blend重新计算比例")
parser.add_argument("--blend_min_tokens", type=int, default=256,
                    help="启用Blend的最小token数")
parser.add_argument("--chunk_size", type=int, default=256,
                    help="缓存块大小")

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

from dataset import *
from torch.utils.data import DataLoader
from utils import *
from transformers import logging as trans_logging
import logging
import json
import time
from MMRAG import MMRAG
from metric import *
from transformers import pipeline

# Set the verbosity level to error
trans_logging.set_verbosity_error()
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("llama_index").setLevel(logging.WARNING)
logging.getLogger("absl").setLevel(logging.WARNING)

setup_seed(0)
if args.dataset == 'mmqa':
    dev_data = MMQADataset('dev')
elif args.dataset == 'webqa':
    dev_data = WebQADataset('val')
elif args.dataset in ['arxiv', 'manual', 'web', 'wiki', 'wit', 'recipe']:
    dev_data = MRAMGDataset(split=args.dataset)
else:
    raise ValueError('Dataset not supported')
if len(dev_data) == 0:
    raise ValueError('No data loaded')
dev_data_loader = DataLoader(dev_data, batch_size=1, shuffle=False)
print(f'Args: {args}')
model = MMRAG(args)
index_start = time.time()
if os.path.exists(f'./working_dir_{args.dataset}/index.faiss'):
    print('Loading existing index...')
    model.rag_backbone.load_index()
else:
    support_data = load_support_text(args.dataset)

    if not os.path.exists(f'./caption/{args.dataset}_caption.json'):
        image_to_text_model = pipeline("image-to-text",
                                       model="nlpconnect/vit-gpt2-image-captioning",
                                       device=torch.device('cuda'))
        if args.dataset == 'mmqa':
            support_data_img = load_support_img_caption_mmqa(image_to_text_model, batch_size=512)
        elif args.dataset == 'webqa':
            support_data_img = load_support_img_caption_webqa(image_to_text_model, batch_size=512)
        elif args.dataset in ['arxiv', 'manual', 'web', 'wiki', 'wit', 'recipe']:
            support_data_img = load_support_img_caption_mramg(image_to_text_model, split=args.dataset)
        with open(f'./caption/{args.dataset}_caption.json', 'w') as f:
            json.dump(support_data_img, f, indent=2)
        # Convert times 3h 11 mins
    else:
        with open(f'./caption/{args.dataset}_caption.json', 'r') as f:
            support_data_img = json.load(f)
    index_start = time.time()
    model.rag_backbone.index(support_data, list(support_data_img.values()))
index_end = time.time()
print('Finish Index!')


async def main():
    graph_count = 0
    oom_count = 0
    start_time = time.time()
    print('Begin Testing!')
    ans_list = []
    gt_list = []
    g_string_list = []
    graph_node_list = []
    graph_edge_list = []
    graph_node_list_q = []
    graph_edge_list_q = []
    q_type_list, query_list, response_list, support_text_list, query_image_list, dpr_result_list = [], [], [], [], [], []
    for idx, data in enumerate(tqdm.tqdm(dev_data_loader, desc='Testing')):
        query_image = []
        support_text_ori = []
        query_image_id = set()
        gt_list_per_query = []
        if not args.no_original_support and args.dataset == 'mmqa':
            for support_data in data['context']:
                if support_data['modality'][0] == 'image':
                    query_image.append(support_data)
                    query_image_id.add(support_data['id'][0])
                elif support_data['modality'][0] == 'text':
                    if type(support_data['text']) is list:
                        support_data['text'] = '\n'.join(support_data['text'])
                    support_text_ori.append(support_data['text'])
        else:
            support_text_rag = ''
        ttft_start = time.time()
        if args.image_caption:
            for support_data in query_image:
                model.add_caption(support_data, data['query'][0], args.LLM_caption)
        start_token_ans = '<ans>'
        end_token_ans = '</ans>'
        model.clean_graph()
        dpr_result = []
        try:
            retrieved_list, graph_str, dpr_result, query_graph = model.any_rag_query(data['query'][0], args.k)
            if args.stop_detect == 'entity':
                if retrieved_list:
                    entities_string = '\n\n'.join(retrieved_list)
                else:
                    entities_string = ''
            if args.stop_detect in ['graph', 'exact'] and args.input_graph_summary:
                if graph_str:
                    graph_str = model.generate_graph_summary_from_string(data['query'][0], graph_str)
            response = model.chat(query_text=data['query'][0],
                                  query_image=query_image,
                                  support_text=support_text_ori,
                                  graph_string=graph_str if args.stop_detect in ['graph', 'exact'] else entities_string,
                                  dpr_result=dpr_result if args.use_dpr else [],
                                  ttft=ttft_start)
            graph_node_list.append(model.Graph.number_of_nodes())
            graph_edge_list.append(model.Graph.number_of_edges())
            graph_node_list_q.append(query_graph.number_of_nodes())
            graph_edge_list_q.append(query_graph.number_of_edges())
        except torch.OutOfMemoryError:
            response = 'Cuda OOM'
            support_text_rag = ''
            oom_count += 1
            print(f'Out of memory at idx {idx}')
        if start_token_ans in response.lower() and end_token_ans in response.lower():
            answer = cut_str(response.lower(), start_token_ans, end_token_ans)[0]
        else:
            answer = response.lower()
        for candidate_answer in data['answers']:
            if isinstance(candidate_answer['answer'], list):
                gt_answer = candidate_answer['answer'][0]
            else:
                gt_answer = candidate_answer['answer']
            gt_list_per_query.append(gt_answer)
        ans_list.append(answer)
        gt_list.append(gt_list_per_query)
        q_type_list.append(data['Qcate'][0])
        query_list.append(data['query'][0])
        response_list.append(response)
        support_text_list.append(support_text_ori)
        query_image_list.append(query_image)
        g_string_list.append(graph_str)
        dpr_result_list.append(dpr_result)
    inference_end_time = time.time()
    result_dict = {
        'ans_list': ans_list,
        'gt_list': gt_list,
        'q_type_list': q_type_list,
        'query_list': query_list,
        'response_list': response_list,
        'support_text_list': support_text_list,
        'query_image_list': query_image_list,
        'dpr_result_list': dpr_result_list,
        'graph_string_list': g_string_list
    }
    with open(f'./results/result_{args.rag}_{args.dataset}_{args.path_recent_nodes}_{args.threshold}.json', 'w') as f:
        json.dump(convert_tensors_to_lists(result_dict), f, indent=2)
    total_metric_dict, results = await calculate_metrics(answer_list=ans_list, gt_list=gt_list, q_type_list=q_type_list,
                                                         query_list=query_list,
                                                         response_list=response_list,
                                                         support_text_list=support_text_list,
                                                         query_image_list=query_image_list,
                                                         dpr_result_list=dpr_result_list,
                                                         g_string_list=g_string_list,
                                                         args=args)

    print(f'Successfully graphed percentage: {graph_count / len(dev_data_loader)}')
    print(f'OOM count: {oom_count}')
    print(f'Average number of nodes in the constructed data graph: {np.mean(graph_node_list)}')
    print(f'Average number of edges in the constructed data graph: {np.mean(graph_edge_list)}')
    print(f'Average number of nodes in the constructed query graph: {np.mean(graph_node_list_q)}')
    print(f'Average number of edges in the constructed query graph: {np.mean(graph_edge_list_q)}')
    print(f'Coarse grain indexing time cost: {(index_end - index_start) / 60} mins')
    print(f'Fine grain indexing time cost: {model.fine_grain_idx_time / 60} mins')
    if args.stop_detect == 'exact':
        result_file_name = f'./results/result_{args.rag}_{args.dataset}_exact.json'
    else:
        result_file_name = f'./results/result_{args.rag}_{args.dataset}_{args.path_recent_nodes}_{args.threshold}.json'
    with open(result_file_name, 'w') as f:
        json.dump(convert_tensors_to_lists(results), f, indent=2)
    model.output_fail_prompts()
    for key, value in total_metric_dict.items():
        print(f'{key}: {value / len(dev_data_loader)}')
    print(f'AVG token usage: {model.token_usage / len(dev_data)}')
    print(f'AVG converge iteration: {sum(model.iteration_end_list) / len(dev_data)}')
    print(f'Total time cost: {(inference_end_time - start_time) / 60} mins')
    print(f'Results saved to {result_file_name}')
    sys.stdout.flush()
    try:
        if hasattr(model, 'lmcache_integrator') and model.lmcache_integrator is not None:
            print("Cleaning up LMCacheVLLMIntegrator...")
            sys.stdout.flush()
            model.lmcache_integrator.cleanup()
            print("LMCacheVLLMIntegrator cleaned up successfully")
            sys.stdout.flush()
    except Exception as e:
        print(f"Warning: Failed to cleanup LMCacheVLLMIntegrator: {e}")
        sys.stdout.flush()
    print('Finished!')
    sys.stdout.flush()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nReceived KeyboardInterrupt, exiting...")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        import os
        os._exit(0)
