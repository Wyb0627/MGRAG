from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers import CLIPProcessor, CLIPModel
from transformers import pipeline
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams
from lmcache.storage_backend.evictor import GraphRAGEvictor
from prompt import *
from openai import AzureOpenAI
import time
from utils import *
from vanillarag_langchain import VanillaRAG
import torch
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import partial
from parallel import *
from networkx.algorithms.isomorphism import DiGraphMatcher

# 添加LMCache相关的导入
try:
    from lmcache.v1.cache_engine import LMCacheEngineBuilder
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.config import LMCacheEngineMetadata
    from lmcache_vllm.lmcache_utils import ENGINE_NAME

    LMCACHE_AVAILABLE = True
except ImportError:
    LMCACHE_AVAILABLE = False
    print("LMCache not available, running without cache support")

# 添加OpenAI库导入用于远程vLLM API
try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI library not available, running without remote vLLM API support")


class MMRAG():
    def __init__(self, args):
        self.args = args
        self.token_usage = 0
        self.image_path_mmqa = '../dataset/multimodalqa/dataset/final_dataset_images/'
        self.image_path_webqa = '../dataset/WebQA/webqa_image/image_file/'
        self.tuple_delimiter = '[SEP]'
        self.record_delimiter = '[REC]'
        self.completion_delimiter = '[END]'
        self.graph_matching_mode = getattr(args, 'graph_matching_mode', 'pairwise')
        self.path_agg_method = getattr(args, 'path_agg_method', 'mean')
        self.threshold = getattr(args, 'threshold', 0.75)
        self.path_agg_weights = getattr(args, 'path_agg_weights', None)
        self.path_recent_nodes = getattr(args, 'path_recent_nodes', 5)
        self.enable_parallel = getattr(args, 'enable_parallel', False)
        self.max_workers = getattr(args, 'max_workers', None)
        self.parallel_threshold = getattr(args, 'parallel_threshold', 20)  # 启用并行化的最小节点数
        self.Graph = nx.DiGraph()
        self.iteration_end_list = []
        self.find_subgraph = 0
        self.fine_grain_idx_time = 0
        self.graphrag_evictor = None
        self.ttft_list = []

        # 初始化LMCache引擎
        self.lmcache_engine = None
        if hasattr(args, 'use_lmcache') and args.use_lmcache and LMCACHE_AVAILABLE:
            try:
                # 使用自动配置系统
                from cache.lmcache_model_config import get_optimal_config_for_model, print_optimal_config

                # 获取模型名称
                model_name = getattr(self, 'model_name', None)
                if model_name is None:
                    # 如果还没有设置model_name，根据args.LLM推断
                    if self.args.LLM == 'qwen':
                        model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
                    elif self.args.LLM == 'mimo':
                        model_name = "XiaomiMiMo/MiMo-VL-7B-RL"
                    elif self.args.LLM == 'llava':
                        model_name = "llava-hf/llama3-llava-next-8b-hf"
                    else:
                        model_name = "qwen"  # 默认
                print(f"Auto-configuring LMCache for model: {model_name}")

                # 获取最优配置
                optimal_config = get_optimal_config_for_model(
                    model_name=model_name,
                    gpu_memory_gb=getattr(args, 'gpu_memory_gb', None),  # 允许用户指定GPU内存
                    auto_detect_gpu=True
                )

                if optimal_config:
                    # 打印最优配置
                    print_optimal_config(optimal_config)

                    # 使用自动配置的超参数
                    auto_hyperparams = optimal_config['hyperparams']
                    gpu_params = optimal_config['gpu_connector_params']
                    model_info = optimal_config['model_info']

                    # 更新args中的相关参数（如果用户没有明确设置）
                    for key, value in auto_hyperparams.items():
                        if not hasattr(args, key) or getattr(args, key) is None:
                            setattr(args, key, value)
                            print(f"Auto-set {key} = {value}")

                    # 设置LMCache环境变量
                    from cache.lmcache_config import setup_lmcache_for_mmrag
                    # lmcache_env_config = setup_lmcache_for_mmrag(args)

                    # 创建LMCache配置
                    lmcache_config = LMCacheEngineConfig.from_defaults(
                        chunk_size=auto_hyperparams['chunk_size'],
                        local_device=auto_hyperparams['local_device'],
                        max_local_cache_size=auto_hyperparams['max_local_cache_size'],
                        enable_blending=True,
                        blend_recompute_ratio=auto_hyperparams['blend_recompute_ratio'],
                        blend_min_tokens=auto_hyperparams['blend_min_tokens'],
                        blend_separator=auto_hyperparams['blend_separator'],
                        blend_add_special_in_precomp=auto_hyperparams['blend_add_special_in_precomp'],
                        enable_p2p=auto_hyperparams['enable_p2p'],
                        remote_url=auto_hyperparams.get('remote_url'),
                        remote_serde=auto_hyperparams['remote_serde'],
                        pipelined_backend=auto_hyperparams['pipelined_backend'],
                        save_decode_cache=auto_hyperparams['save_decode_cache']
                    )

                    # 创建LMCache元数据
                    lmcache_metadata = LMCacheEngineMetadata(
                        model_name=model_info['model_name'],
                        world_size=1,
                        worker_id=0,
                        fmt="vllm",
                        kv_dtype=gpu_params['kv_dtype'],
                        kv_shape=(gpu_params['num_layers'], 2, auto_hyperparams['chunk_size'],
                                  gpu_params['hidden_dim_size'] // 128, 128),  # 自动计算num_kv_head
                        use_mla=False
                    )

                else:
                    # 如果自动配置失败，使用默认配置
                    print("Warning: Auto-configuration failed, using default config")
                    # 创建LMCache配置
                    lmcache_config = LMCacheEngineConfig.from_defaults(
                        chunk_size=getattr(args, 'chunk_size', 256),
                        local_device="cuda",
                        max_local_cache_size=getattr(args, 'max_cache_size', 10.0),
                        enable_blending=True,
                        blend_recompute_ratio=getattr(args, 'blend_recompute_ratio', 0.15),
                        blend_min_tokens=getattr(args, 'blend_min_tokens', 256),
                        blend_separator="[LMCACHE_BLEND_SEP]",
                        blend_add_special_in_precomp=False,
                        enable_p2p=False,
                        remote_url=None,
                        remote_serde="torch",
                        pipelined_backend=False,
                        save_decode_cache=False
                    )

                    # 创建LMCache元数据
                    lmcache_metadata = LMCacheEngineMetadata(
                        model_name="qwen2.5-vl-7b",
                        world_size=1,
                        worker_id=0,
                        fmt="vllm",
                        kv_dtype=torch.bfloat16,
                        kv_shape=(32, 2, 256, 32, 128),
                        use_mla=False
                    )

                # 创建GPU连接器
                try:
                    from lmcache.v1.gpu_connector import VLLMPagedMemGPUConnectorV2

                    if optimal_config:
                        # 使用自动配置的GPU参数
                        gpu_connector = VLLMPagedMemGPUConnectorV2(
                            hidden_dim_size=gpu_params['hidden_dim_size'],
                            num_layers=gpu_params['num_layers'],
                            use_gpu=True,
                            chunk_size=auto_hyperparams['chunk_size'],
                            dtype=gpu_params['kv_dtype'],
                            device="cuda",
                            use_mla=False
                        )
                    else:
                        # 使用默认GPU参数
                        gpu_connector = VLLMPagedMemGPUConnectorV2(
                            hidden_dim_size=4096,
                            num_layers=32,
                            use_gpu=True,
                            chunk_size=256,
                            dtype=torch.bfloat16,
                            device="cuda",
                            use_mla=False
                        )

                    print(f"Created GPU connector successfully")

                except ImportError as e:
                    print(f"Warning: Could not import GPU connector: {e}")
                    print("LMCache will be initialized without GPU connector")
                    gpu_connector = None
                except Exception as e:
                    print(f"Warning: Failed to create GPU connector: {e}")
                    print("LMCache will be initialized without GPU connector")
                    gpu_connector = None

                # 初始化LMCache引擎
                self.lmcache_engine = LMCacheEngineBuilder.get_or_create(
                    ENGINE_NAME,
                    lmcache_config,
                    lmcache_metadata,
                    gpu_connector
                )

                print("LMCache engine initialized successfully")

                # 初始化GraphRAG evictor（如果使用LMCache）
                try:
                    self.graphrag_evictor = GraphRAGEvictor(
                        max_cache_size=getattr(args, 'max_cache_size', 10.0),
                        pagerank_weight=getattr(args, 'pagerank_weight', 0.7),
                        recency_weight=getattr(args, 'recency_weight', 0.3),
                        enable_graphrag=getattr(args, 'enable_graphrag', True)
                    )
                    print("GraphRAG evictor initialized successfully")
                except ImportError as e:
                    print(f"Failed to import GraphRAGEvictor: {e}")
                    print("Continuing without GraphRAG evictor support")

            except Exception as e:
                print(f"Failed to initialize LMCache engine: {e}")
                print("Continuing without LMCache support")
                self.lmcache_engine = None

        self.blend_mode = getattr(args, 'blend_mode', 'graphrag')

        if args.dataset == 'mmqa':
            self.image_path = self.image_path_mmqa
        elif args.dataset == 'webqa':
            self.image_path = self.image_path_webqa
        if self.args.LLM in ['qwen', 'mimo']:
            if self.args.LLM == 'qwen':
                model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
            else:
                model_name = 'XiaomiMiMo/MiMo-VL-7B-RL'
            self.model_name = model_name
            if args.use_vllm:
                vllm_api_base = getattr(args, 'vllm_api_base', 'http://localhost:8001/v1')
                vllm_api_key = getattr(args, 'vllm_api_key', 'EMPTY')
                self.vllm_client = OpenAI(
                    api_key=vllm_api_key,
                    base_url=vllm_api_base,
                )
                # 获取模型列表
                try:
                    models = self.vllm_client.models.list()
                    self.vllm_model = models.data[0].id if models.data else model_name
                    print(f"Connected to remote vLLM API with model: {self.vllm_model}")
                except Exception as e:
                    print(f"Failed to connect to remote vLLM API: {e}")
            else:
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_name,
                    # 'Qwen/Qwen2.5-VL-32B-Instruct-AWQ',
                    # torch_dtype='auto',
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    device_map="auto",
                )
            # default processer
            self.processor = AutoProcessor.from_pretrained(model_name)
            # self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-32B-Instruct-AWQ")

        if self.args.rag == 'anyrag':
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.image_to_text = pipeline("image-to-text",
                                      model="nlpconnect/vit-gpt2-image-captioning",
                                      device=torch.device('cuda'))
        # 添加InstructBLIP模型初始化
        self.fail_prompts = []

        if args.rag == 'anyrag':
            self.rag_backbone = VanillaRAG(args)

    def build_graph_and_pagerank(self, retrieved_facts):
        """
        构建图并计算PageRank分数，同时更新evictor的图信息
        """
        G = nx.DiGraph()
        for s, r, t in retrieved_facts:
            G.add_edge(s, t, relation=r)

        # 计算PageRank分数
        pagerank_dict = nx.pagerank(G)

        # 归一化pagerank分数
        scores = torch.tensor([pagerank_dict.get(s, 0.0) for s, _, _ in retrieved_facts], dtype=torch.float32)
        if scores.max() > 0:
            scores = scores / scores.max()

        # 更新evictor的图信息
        if self.graphrag_evictor is not None:
            self.graphrag_evictor.update_graph_and_pagerank(G, pagerank_dict)

        return G, scores

    def compute_and_store_pagerank(self):
        """
        计算当前图的PageRank分数并存储，供后续使用
        """
        if len(self.Graph.nodes()) == 0:
            self.pagerank_scores = {}
            return self.pagerank_scores

        try:
            # 计算当前图的PageRank分数
            self.pagerank_scores = nx.pagerank(self.Graph)

            # 更新evictor的图信息
            if self.graphrag_evictor is not None:
                self.graphrag_evictor.update_graph_and_pagerank(self.Graph, self.pagerank_scores)

            # print(f"Computed PageRank scores for {len(self.pagerank_scores)} nodes")
        except Exception as e:
            print(f"Failed to compute PageRank scores: {e}")
            self.pagerank_scores = {}

        return self.pagerank_scores

    def register_kv_chunk_entity_mapping(self, cache_key, entity_name):
        """
        注册KV chunk与图实体的映射关系，用于evictor的评分
        """
        if self.graphrag_evictor is not None:
            self.graphrag_evictor.register_entity_mapping(cache_key, entity_name)

    def update_evictor_graph_info(self):
        """
        更新evictor的图信息和PageRank分数
        """
        if self.graphrag_evictor is not None and len(self.Graph.nodes()) > 0:
            try:
                # 计算当前图的PageRank分数
                pagerank_scores = nx.pagerank(self.Graph)
                self.graphrag_evictor.update_graph_and_pagerank(self.Graph, pagerank_scores)
                print(f"Updated evictor with graph containing {len(self.Graph.nodes())} nodes")
            except Exception as e:
                print(f"Failed to update evictor graph info: {e}")

    def get_cache_stats(self):
        """
        获取缓存统计信息
        """
        if self.graphrag_evictor is not None:
            return self.graphrag_evictor.get_cache_stats()
        return {}

    def graph_matching_stop_detection(self, query_graph, anchor_node, threshold=0.75,
                                      recent_nodes=5) -> nx.DiGraph or str:
        if getattr(self, "graph_matching_mode", "pairwise") == "pairwise":
            return self._graph_matching_pairwise(query_graph, anchor_node, threshold)
        else:
            agg_method = getattr(self, "path_agg_method", "mean")
            agg_weights = getattr(self, "path_agg_weights", None)
            # recent_nodes = getattr(self, "path_recent_nodes", 5)
            return self._graph_matching_path(query_graph, anchor_node, threshold, agg_method, agg_weights, recent_nodes)

    def graph_matching_stop_detection_parallel(self, query_graph, anchor_node, threshold=0.75,
                                               max_workers=None, recent_nodes=5) -> nx.DiGraph or str:
        """并行化的graph matching方法"""
        if getattr(self, "graph_matching_mode", "pairwise") == "pairwise":
            return self._graph_matching_pairwise_parallel(query_graph, anchor_node, threshold, max_workers)
        elif getattr(self, "graph_matching_mode", "pairwise") == "exact":
            return self._graph_matching_exact_parallel(query_graph, anchor_node, threshold, max_workers)
        else:
            agg_method = getattr(self, "path_agg_method", "mean")
            agg_weights = getattr(self, "path_agg_weights", None)
            # recent_nodes = getattr(self, "path_recent_nodes", 5)
            return self._graph_matching_path_parallel(query_graph, anchor_node, threshold, agg_method, agg_weights,
                                                      recent_nodes, max_workers)

    def graph_matching_stop_detection_process(self, query_graph, anchor_node, threshold=0.75,
                                              max_workers=None) -> nx.DiGraph or str:
        """进程并行的graph matching方法"""
        if getattr(self, "graph_matching_mode", "pairwise") == "pairwise":
            return self._graph_matching_pairwise_process(query_graph, anchor_node, threshold, max_workers)

        else:
            agg_method = getattr(self, "path_agg_method", "mean")
            agg_weights = getattr(self, "path_agg_weights", None)
            recent_nodes = getattr(self, "path_recent_nodes", 5)
            return self._graph_matching_path_process(query_graph, anchor_node, threshold, agg_method, agg_weights,
                                                     recent_nodes, max_workers)

    def aggregate_embeddings(self, embeddings, method="mean", weights=None):
        if not embeddings:
            return None
        if method == "mean":
            return np.mean(embeddings, axis=0)
        elif method == "concat":
            return np.concatenate(embeddings, axis=0)
        elif method == "weighted":
            if weights is None or len(weights) != len(embeddings):
                raise ValueError("weights must be provided and match the number of embeddings for weighted aggregation")
            weights = np.array(weights) / np.sum(weights)
            return np.sum([w * e for w, e in zip(weights, embeddings)], axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def _graph_matching_pairwise(self, query_graph, anchor_node, threshold=0.75):
        matched_subgraphs = []
        for candidate, data in self.Graph.nodes(data=True):
            sim = cosine_sim(data['embedding'], query_graph.nodes[anchor_node]['embedding'])
            if sim <= threshold:
                continue
            GM = DiGraphMatcher(self.Graph, query_graph, node_match=partial(node_edge_match, threshold=threshold),
                                edge_match=partial(node_edge_match, threshold=threshold))
            for mapping in GM.subgraph_isomorphisms_iter():
                if mapping.get(candidate, None) != anchor_node:
                    continue
                sub_nodes = list(mapping.keys())
                subgraph = self.Graph.subgraph(sub_nodes).copy()
                similarity = calculate_subgraph_similarity(query_graph, subgraph, mapping)
                matched_subgraphs.append((subgraph, similarity))
        if matched_subgraphs:
            matched_subgraphs.sort(key=lambda x: x[1], reverse=True)
            return matched_subgraphs[0][0]
        else:
            return ''

    def _process_candidate_pairwise(self, candidate_data, query_graph, anchor_node, threshold):
        """处理单个候选节点的pairwise matching（用于并行化）"""
        candidate, data = candidate_data

        # 检查anchor node相似度 - 提前过滤
        sim = cosine_sim(data['embedding'], query_graph.nodes[anchor_node]['embedding'])
        if sim <= threshold:
            return None

        # 执行子图同构匹配
        GM = DiGraphMatcher(self.Graph, query_graph, node_match=partial(node_edge_match, threshold=threshold),
                            edge_match=partial(node_edge_match, threshold=threshold))
        matched_subgraphs = []

        for mapping in GM.subgraph_isomorphisms_iter():
            if mapping.get(candidate, None) != anchor_node:
                continue
            sub_nodes = list(mapping.keys())
            subgraph = self.Graph.subgraph(sub_nodes).copy()
            similarity = calculate_subgraph_similarity(query_graph, subgraph, mapping)
            matched_subgraphs.append((subgraph, similarity))

        return matched_subgraphs

    def _graph_matching_pairwise_parallel(self, query_graph, anchor_node, threshold=0.75, max_workers=None):
        """并行化的pairwise graph matching - 优化版本"""
        # 设置当前阈值供匹配函数使用
        self._current_threshold = threshold

        # 动态决定是否使用并行化
        candidates = list(self.Graph.nodes(data=True))

        # 智能决策是否使用并行化
        should_parallelize = (
                self.enable_parallel and
                len(candidates) >= self.parallel_threshold and
                max_workers is not None and
                max_workers > 1
        )

        if not should_parallelize:
            return self._graph_matching_pairwise(query_graph, anchor_node, threshold)

        # 限制最大线程数
        max_workers = min(max_workers, mp.cpu_count(), 8, len(candidates))

        # 使用线程池并行处理候选节点
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 创建偏函数，固定其他参数
            process_func = partial(self._process_candidate_pairwise,
                                   query_graph=query_graph,
                                   anchor_node=anchor_node,
                                   threshold=threshold)

            # 并行处理所有候选节点
            futures = [executor.submit(process_func, candidate_data) for candidate_data in candidates]

            # 收集结果
            all_matched_subgraphs = []
            for future in futures:
                result = future.result()
                if result is not None:
                    all_matched_subgraphs.extend(result)

        # 排序并返回最佳匹配
        if all_matched_subgraphs:
            all_matched_subgraphs.sort(key=lambda x: x[1], reverse=True)
            return all_matched_subgraphs[0][0]
        else:
            return ''

    def _graph_matching_pairwise_process(self, query_graph, anchor_node, threshold=0.75, max_workers=None):
        raise NotImplementedError("Process-based pairwise graph matching is not implemented yet.")

    def _graph_matching_path(self, query_graph, anchor_node, threshold, agg_method, agg_weights, recent_nodes=5):
        matched_subgraphs = []
        for candidate, data in self.Graph.nodes(data=True):
            sim = cosine_sim(data['embedding'], query_graph.nodes[anchor_node]['embedding'])
            if sim <= threshold:
                continue
            GM = DiGraphMatcher(self.Graph, query_graph,
                                node_match=partial(node_edge_match, threshold=threshold),
                                edge_match=partial(node_edge_match, threshold=threshold))
            for mapping in GM.subgraph_isomorphisms_iter():
                if mapping.get(candidate, None) != anchor_node:
                    continue
                pair_sims = []
                for u, v, edge_data in query_graph.edges(data=True):
                    data_target = [k for k, vq in mapping.items() if vq == v]
                    if not data_target:
                        continue
                    data_target = data_target[0]
                    try:
                        path = nx.shortest_path(self.Graph, source=candidate, target=data_target)
                    except nx.NetworkXNoPath:
                        continue
                    path_embs = []
                    # 只取路径上最近的recent_nodes个节点和边
                    path_length = len(path)
                    if recent_nodes > 0:
                        start_idx = max(0, path_length - recent_nodes)
                    else:
                        start_idx = 0

                    for i in range(start_idx, path_length):
                        emb = self.Graph.nodes[path[i]]['embedding']
                        if isinstance(emb, torch.Tensor):
                            emb = emb.detach().cpu().numpy()
                        path_embs.append(emb)
                        if i < path_length - 1:
                            emb = self.Graph[path[i]][path[i + 1]]['embedding']
                            if isinstance(emb, torch.Tensor):
                                emb = emb.detach().cpu().numpy()
                            path_embs.append(emb)

                    if agg_method == "weighted":
                        weights = agg_weights if agg_weights is not None else [1.0] * len(path_embs)
                    else:
                        weights = None
                    path_emb_mean = self.aggregate_embeddings(path_embs, method=agg_method, weights=weights)
                    path_emb_sum = np.sum(path_embs, axis=0)
                    edge_emb = edge_data['embedding']
                    node_emb = query_graph.nodes[v]['embedding']
                    if isinstance(edge_emb, torch.Tensor):
                        edge_emb = edge_emb.detach().cpu().numpy()
                    if isinstance(node_emb, torch.Tensor):
                        node_emb = node_emb.detach().cpu().numpy()
                    if agg_method == "concat":
                        pair_emb = np.concatenate([edge_emb, node_emb], axis=0)
                    elif agg_method == "weighted" and agg_weights is not None and len(agg_weights) == 2:
                        pair_emb = self.aggregate_embeddings([edge_emb, node_emb], method="weighted",
                                                             weights=agg_weights)
                    else:
                        pair_emb = np.mean([edge_emb, node_emb], axis=0)
                    sim_mean = cosine_sim(path_emb_mean, pair_emb)
                    sim_sum = cosine_sim(path_emb_sum, pair_emb)
                    # if sim_mean > threshold or sim_sum > threshold:
                    #     pair_sims.append(max(sim_mean, sim_sum))
                    if sim_mean > threshold:
                        pair_sims.append(max(sim_mean, sim_sum))
                if pair_sims:
                    avg_sim = np.mean(pair_sims)
                    if avg_sim > threshold:
                        sub_nodes = list(mapping.keys())
                        subgraph = self.Graph.subgraph(sub_nodes).copy()
                        matched_subgraphs.append((subgraph, avg_sim))
        if matched_subgraphs:
            matched_subgraphs.sort(key=lambda x: x[1], reverse=True)
            return matched_subgraphs[0][0]
        else:
            return ''

    def _process_candidate_path(self, candidate_data, query_graph, anchor_node, threshold, agg_method, agg_weights,
                                recent_nodes):
        """处理单个候选节点的path matching（用于并行化）"""
        candidate, data = candidate_data

        # 检查anchor node相似度 - 提前过滤
        sim = cosine_sim(data['embedding'], query_graph.nodes[anchor_node]['embedding'])
        if sim <= threshold:
            return None

        GM = DiGraphMatcher(self.Graph, query_graph,
                            node_match=lambda n1, n2: True,
                            edge_match=lambda e1, e2: True)

        matched_subgraphs = []
        for mapping in GM.subgraph_isomorphisms_iter():
            if mapping.get(candidate, None) != anchor_node:
                continue
            pair_sims = []
            for u, v, edge_data in query_graph.edges(data=True):
                data_target = [k for k, vq in mapping.items() if vq == v]
                if not data_target:
                    continue
                data_target = data_target[0]
                try:
                    path = nx.shortest_path(self.Graph, source=candidate, target=data_target)
                except nx.NetworkXNoPath:
                    continue
                path_embs = []
                # 只取路径上最近的recent_nodes个节点和边
                path_length = len(path)
                if recent_nodes > 0:
                    start_idx = max(0, path_length - recent_nodes)
                else:
                    start_idx = 0

                for i in range(start_idx, path_length):
                    emb = self.Graph.nodes[path[i]]['embedding']
                    if isinstance(emb, torch.Tensor):
                        emb = emb.detach().cpu().numpy()
                    path_embs.append(emb)
                    if i < path_length - 1:
                        emb = self.Graph[path[i]][path[i + 1]]['embedding']
                        if isinstance(emb, torch.Tensor):
                            emb = emb.detach().cpu().numpy()
                        path_embs.append(emb)

                if agg_method == "weighted":
                    weights = agg_weights if agg_weights is not None else [1.0] * len(path_embs)
                else:
                    weights = None
                path_emb_mean = self.aggregate_embeddings(path_embs, method=agg_method, weights=weights)
                path_emb_sum = np.sum(path_embs, axis=0)
                edge_emb = edge_data['embedding']
                node_emb = query_graph.nodes[v]['embedding']
                if isinstance(edge_emb, torch.Tensor):
                    edge_emb = edge_emb.detach().cpu().numpy()
                if isinstance(node_emb, torch.Tensor):
                    node_emb = node_emb.detach().cpu().numpy()
                if agg_method == "concat":
                    pair_emb = np.concatenate([edge_emb, node_emb], axis=0)
                elif agg_method == "weighted" and agg_weights is not None and len(agg_weights) == 2:
                    pair_emb = self.aggregate_embeddings([edge_emb, node_emb], method="weighted", weights=agg_weights)
                else:
                    pair_emb = np.mean([edge_emb, node_emb], axis=0)
                sim_mean = cosine_sim(path_emb_mean, pair_emb)
                sim_sum = cosine_sim(path_emb_sum, pair_emb)
                # if sim_mean > threshold or sim_sum > threshold:
                #     pair_sims.append(max(sim_mean, sim_sum))
                if sim_mean > threshold:
                    pair_sims.append(max(sim_mean, sim_sum))
            if pair_sims:
                avg_sim = np.mean(pair_sims)
                if avg_sim > threshold:
                    sub_nodes = list(mapping.keys())
                    subgraph = self.Graph.subgraph(sub_nodes).copy()
                    matched_subgraphs.append((subgraph, avg_sim))

        return matched_subgraphs

    def _graph_matching_exact_parallel(self, query_graph, anchor_node, recent_nodes=5, max_workers=None):
        """
        并行 exact graph matching：
        以 candidate == anchor_node 为锚点，用 DiGraphMatcher 检查 query_graph 在 self.Graph 中的严格同构关系（节点/边 embedding 全等）。
        若并行条件不满足则抛出异常。
        """
        candidates = list(self.Graph.nodes(data=True))
        should_parallelize = (
                self.enable_parallel and
                len(candidates) >= self.parallel_threshold and
                max_workers is not None and
                max_workers > 1
        )
        if not should_parallelize:
            raise NotImplementedError("Sequential _graph_matching_exact fallback not implemented!")
        max_workers = min(max_workers, mp.cpu_count(), 8, len(candidates))

        def process_candidate_exact(candidate_data):
            candidate, data = candidate_data
            if candidate != anchor_node:
                return None
            GM = DiGraphMatcher(
                self.Graph, query_graph,
                node_match=lambda n1, n2: n1['embedding'] is not None and n2['embedding'] is not None and np.allclose(
                    n1['embedding'], n2['embedding']),
                edge_match=lambda e1, e2: e1['embedding'] is not None and e2['embedding'] is not None and np.allclose(
                    e1['embedding'], e2['embedding'])
            )
            matched_subgraphs = []
            for mapping in GM.subgraph_isomorphisms_iter():
                if mapping.get(candidate, None) != anchor_node:
                    continue
                sub_nodes = list(mapping.keys())
                subgraph = self.Graph.subgraph(sub_nodes).copy()
                matched_subgraphs.append((subgraph, 1.0))
            return matched_subgraphs

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_candidate_exact, candidate_data) for candidate_data in candidates]
            all_matched_subgraphs = []
            for future in futures:
                result = future.result()
                if result is not None:
                    all_matched_subgraphs.extend(result)

        if all_matched_subgraphs:
            all_matched_subgraphs.sort(key=lambda x: x[1], reverse=True)
            return all_matched_subgraphs[0][0]
        else:
            return ''

    def _graph_matching_path_parallel(self, query_graph, anchor_node, threshold, agg_method, agg_weights,
                                      recent_nodes=5, max_workers=None):
        """并行化的path graph matching - 优化版本"""
        # 动态决定是否使用并行化
        candidates = list(self.Graph.nodes(data=True))

        # 智能决策是否使用并行化
        should_parallelize = (
                self.enable_parallel and
                len(candidates) >= self.parallel_threshold and
                max_workers is not None and
                max_workers > 1
        )

        if not should_parallelize:
            return self._graph_matching_path(query_graph, anchor_node, threshold, agg_method, agg_weights, recent_nodes)

        # 限制最大线程数
        max_workers = min(max_workers, mp.cpu_count(), 8, len(candidates))

        # 使用线程池并行处理候选节点
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 创建偏函数，固定其他参数
            process_func = partial(self._process_candidate_path,
                                   query_graph=query_graph,
                                   anchor_node=anchor_node,
                                   threshold=threshold,
                                   agg_method=agg_method,
                                   agg_weights=agg_weights,
                                   recent_nodes=recent_nodes)

            # 并行处理所有候选节点
            futures = [executor.submit(process_func, candidate_data) for candidate_data in candidates]

            # 收集结果
            all_matched_subgraphs = []
            for future in futures:
                result = future.result()
                if result is not None:
                    all_matched_subgraphs.extend(result)

        # 排序并返回最佳匹配
        if all_matched_subgraphs:
            all_matched_subgraphs.sort(key=lambda x: x[1], reverse=True)
            return all_matched_subgraphs[0][0]
        else:
            return ''

    def _graph_matching_path_process(self, query_graph, anchor_node, threshold, agg_method, agg_weights, recent_nodes=5,
                                     max_workers=None):
        """进程并行的path graph matching"""
        # 智能决策是否使用并行化
        candidates = list(self.Graph.nodes(data=True))

        should_parallelize = (
                self.enable_parallel and
                len(candidates) >= self.parallel_threshold and
                max_workers is not None and
                max_workers > 1
        )

        if not should_parallelize:
            return self._graph_matching_path(query_graph, anchor_node, threshold, agg_method, agg_weights, recent_nodes)

        # 限制最大进程数
        max_workers = min(max_workers, mp.cpu_count(), 4, len(candidates))  # 进程数限制更严格

        # 序列化图数据
        graph_data = {
            'nodes': {n: dict(self.Graph.nodes[n]) for n in self.Graph.nodes()},
            'edges': [(u, v, dict(self.Graph[u][v])) for u, v in self.Graph.edges(data=True)]
        }

        query_graph_data = {
            'nodes': {n: dict(query_graph.nodes[n]) for n in query_graph.nodes()},
            'edges': [(u, v, dict(query_graph[u][v])) for u, v in query_graph.edges(data=True)]
        }

        # 使用进程池并行处理候选节点
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 准备参数
            args_list = [
                (candidate_data, graph_data, query_graph_data, anchor_node, threshold, agg_method, agg_weights,
                 recent_nodes)
                for candidate_data in candidates
            ]

            # 并行处理所有候选节点
            results = list(executor.map(_process_candidate_path_worker, args_list))

            # 收集结果
            all_matched_subgraphs = []
            pagerank_scores = {}
            for result in results:
                if result is not None:
                    matched_subgraphs, pr_scores = result
                    all_matched_subgraphs.extend(matched_subgraphs)
                    # 合并所有进程计算的PageRank分数（以最后一个为准，实际应用中可能需要更好的合并策略）
                    pagerank_scores.update(pr_scores)

            # 更新PageRank分数
            if pagerank_scores and self.graphrag_evictor is not None:
                self.graphrag_evictor.update_graph_and_pagerank(self.Graph, pagerank_scores)

        # 重建子图对象
        reconstructed_subgraphs = []
        for subgraph_data, similarity in all_matched_subgraphs:
            subgraph = nx.DiGraph()
            for node, data in subgraph_data['nodes'].items():
                subgraph.add_node(node, **data)
            for u, v, data in subgraph_data['edges']:
                subgraph.add_edge(u, v, **data)
            reconstructed_subgraphs.append((subgraph, similarity))

        # 排序并返回最佳匹配
        if reconstructed_subgraphs:
            reconstructed_subgraphs.sort(key=lambda x: x[1], reverse=True)
            return reconstructed_subgraphs[0][0]
        else:
            return ''

    def text_embedding(self, text):
        inputs = self.clip_processor(text=text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.clip_model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
        return outputs

    def image_embedding(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.clip_model.get_image_features(**inputs)
        return outputs

    def output_fail_prompts(self):
        print(f'Failed prompts: {len(self.fail_prompts)}')
        with open('fail_prompts.txt', 'w') as f:
            json.dump(self.fail_prompts, f, indent=2)

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def chat_openai(self,
                    messages,
                    api_key='EMPTY',
                    model="gpt-4o",
                    n=1,
                    patience=3,
                    sleep_time=0):
        self.token_usage += num_tokens_from_messages(messages, model)
        while patience > 0:
            patience -= 1
            try:
                client = AzureOpenAI(
                    api_key=api_key,
                    azure_endpoint="https://hkust.azure-api.net",
                    api_version='2024-10-21',
                )
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    n=n
                )
                if n == 1:
                    prediction = response.choices[0].message.content.strip()
                    if prediction != "" and prediction != None:
                        return prediction
                else:
                    prediction = [choice.message.content.strip() for choice in response.choices]
                    if prediction[0] != "" and prediction[0] != None:
                        return prediction
            except Exception as e:
                print(e)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        self.fail_prompts.append(messages)
        return ""

    def chat_gpt(self, query_text, query_image, support_text, support_text_rag):
        messages = [{"role": "system",
                     "content": system_prompt}]
        query_message = 'Here is the real question:\n' + query_text
        image_list = []
        if query_image is not None:
            for image in query_image:
                if self.args.dataset == 'mmqa':
                    image_path = self.image_path_mmqa + image['path'][0]
                    base64_image = self.encode_image(image_path)
                    image_list.append({"type": "image_url",
                                       "image_url": {
                                           "url": f"data:image/jpeg;base64,{base64_image}",
                                           # "url": image['url'][0],
                                           'detail': 'low',
                                       }})
                else:
                    image_list.append({"type": "image_url",
                                       "image_url": {
                                           "url": image['url'][0],
                                           'detail': 'low',
                                       }})
        sup_text = ''
        beginning = ''
        if support_text:
            support_str = '\n'.join(support_text)
            beginning = 'Please answer question base on support information.\n\n'
            sup_text = f'Here is the supporting material to help you answer the question:\n{support_str}\n\n'
        # else:
        # beginning = 'Please answer this question.\n\n'
        sup_text_rag = ''
        if support_text_rag:
            support_str_rag = '\n'.join(support_text_rag)
            if not beginning:
                beginning = 'Please answer question base on support information.\n\n'
            sup_text_rag = (f'Here is the supporting material retrieved from external sources '
                            f'to help you answer the question:\n{support_str_rag}\n\n')

        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text",
                     "text": beginning + sup_text + sup_text_rag + query_message}
                ],
            }
        )
        if image_list:
            messages[-1]['content'].extend(image_list)
        return self.chat_openai(messages)

    def graph_extraction(self, query: str, retrieved_text: List[str], image_list_query: List[dict]):
        messages = [{"role": "system",
                     "content": sys_prompt_extract}, {
                        "role": "user",
                        "content": []
                    }]

        text_query_input = user_prompt_query.format(query_content=query,
                                                    text_content='\n'.join(retrieved_text),
                                                    tuple_delimiter=self.tuple_delimiter,
                                                    record_delimiter=self.record_delimiter,
                                                    completion_delimiter=self.completion_delimiter)
        messages[-1]['content'].append(
            {
                "type": "text",
                "text": text_query_input
            })
        if image_list_query:
            messages[-1]['content'].extend(image_list_query)
        if 'gpt' in self.args.LLM:
            response = self.chat_openai(messages)
        else:
            response = self.msg_input_qwen(messages)
        records = split_string_by_multi_markers(response, [self.record_delimiter, self.completion_delimiter])
        processed_records = process_extracted_records(records, self.tuple_delimiter)

        return processed_records

    def extract_query_graph(self, query):
        query_graph_extraction_input = user_prompt_query.format(query_content=query,
                                                                tuple_delimiter=self.tuple_delimiter,
                                                                record_delimiter=self.record_delimiter,
                                                                completion_delimiter=self.completion_delimiter)
        query_records = split_string_by_multi_markers(query_graph_extraction_input,
                                                      [self.record_delimiter, self.completion_delimiter])
        processed_query_records = process_extracted_records(query_records, self.tuple_delimiter)
        anchor_node = ''
        encode_list = []
        query_graph = nx.DiGraph()
        removed_idx = []
        for idx, record in enumerate(processed_query_records):
            if 'entity' in record[0].lower().strip():
                encode_list.append(record[1].lower().strip())
            elif 'relation' in record[0].lower().strip():
                relation_name = record[3].lower().strip()
                encode_list.append(relation_name)
            else:
                removed_idx.append(idx)
                print(f'Invalid record format: {record}')
                print('Skipping this record.')
                continue
        for idx in removed_idx:
            del processed_query_records[idx]
        encode_matrix = self.text_embedding(encode_list)
        for idx, record in enumerate(processed_query_records):
            embedding = encode_matrix[idx]
            if 'entity' in record[0].lower().strip():
                if 'anchor' in record[0].lower().strip():
                    anchor_node = record[1].lower().strip()
                query_graph.add_node(record[1].lower().strip(), embedding=embedding)
            elif 'relation' in record[0].lower().strip():
                src = record[1].lower().strip()
                tgt = record[2].lower().strip()
                relation_name = record[3].lower().strip()
                # 确保source/target entity节点都在图中
                if src not in query_graph.nodes:
                    # 用 relation embedding 兜底
                    src_embedding = self.text_embedding(src)[0]
                    query_graph.add_node(src, embedding=src_embedding)
                if tgt not in query_graph.nodes:
                    tgt_embedding = self.text_embedding(tgt)[0]
                    query_graph.add_node(tgt, embedding=tgt_embedding)
                query_graph.add_edge(src, tgt, relation=relation_name, embedding=embedding)
            else:
                print(f'{processed_query_records}')
                print(f'Invalid record format: {record}')
                print('Skipping this record.')
                continue
        del processed_query_records
        return query_graph, anchor_node

    def any_rag_query(self, query: str, threshold=5):
        count = 0
        retrieved_list = []
        marker = False
        graph_str = ''
        dpr_result = []
        self.T = torch.zeros(512).to('cuda')
        query_graph, anchor_node = self.extract_query_graph(query)
        retrieved_docs = self.rag_backbone.query(query, top_k=threshold)
        if not retrieved_docs:
            print('No retrieval')
            return retrieved_list, graph_str, dpr_result
        fine_grain_idx_start = time.time()
        while count < threshold or marker:
            if not retrieved_list and self.args.stop_detect == 'entity':
                break
            messages = [{"role": "system",
                         "content": sys_prompt_extract}, {
                            "role": "user",
                            "content": [
                            ],
                        }]

            image_list, retrieved_text = [], []
            # print(retrieved_img)
            mm_node = retrieved_docs[count]
            if mm_node.metadata['modality'] == 'image':
                if 'webqa' in mm_node.metadata['path'] or 'ywangnx' in mm_node.metadata['path']:
                    image_path = mm_node.metadata['path']
                elif 'http' in mm_node.metadata['path'] or 'https' in mm_node.metadata['path']:
                    image_path = mm_node.metadata['path']
                elif '../' in mm_node.metadata['path']:
                    image_path = mm_node.metadata['path']
                else:
                    image_path = self.image_path_mmqa + mm_node.metadata['path']
                if 'gpt' in self.args.LLM:
                    # print(image)
                    if self.args.dataset == 'mmqa':
                        base64_image = self.encode_image(image_path)
                        image_list.append({"type": "image_url",
                                           "image_url": {
                                               "url": f"data:image/jpeg;base64,{base64_image}",
                                               # "url": image['url'][0],
                                               'detail': 'low',
                                           }})
                    else:
                        image_list.append({"type": "image_url",
                                           "image_url": {
                                               "url": image_path,
                                               'detail': 'low',
                                           }})
                else:
                    image_list.append({"type": "image",
                                       "image": image_path})  # 使用图片路径作为内容标识
                # 准备passage信息
                passage_info = {
                    'id': mm_node.id,
                    'type': 'image',
                    'content': mm_node.metadata['path']  # 使用图片路径作为内容标识
                }
                if getattr(self.args, 'use_dpr', False):
                    sys_prompt_captioning = (f"Describe the image with respect to a provided text query, "
                                             f"clearly point out and describe the image's related feature for answering"
                                             f" the query.")
                    message_captioning = [
                        {
                            'role': 'system',
                            'content': sys_prompt_captioning
                        }]
                    user_prompt_captioning = {'role': 'user',
                                              'content': [{"type": "text",
                                                           "text": f"Here is the text query, please describe "
                                                                   f"the provided image with respect to this query: {query}."},
                                                          {"type": "image",
                                                           "image": image_path}]}
                    message_captioning.append(user_prompt_captioning)
                    try:
                        dpr_result.append(passage_info['content'])
                    except Exception as e:
                        print(f"Error generating caption with QWEN: {e}")
                        passage_info['caption'] = ""
            else:
                retrieved_text.append(mm_node.page_content)
                # 准备passage信息
                passage_info = {
                    'id': mm_node.id,
                    'type': 'text',
                    'content': mm_node.page_content
                }
                dpr_result.append(mm_node.page_content)
            if image_list:
                messages[-1]['content'].extend(image_list)
            text_query_input = user_prompt.format(query_content=query,
                                                  text_content='\n'.join(retrieved_text),
                                                  tuple_delimiter=self.tuple_delimiter,
                                                  record_delimiter=self.record_delimiter,
                                                  completion_delimiter=self.completion_delimiter)
            # print(text_query_input)
            messages[-1]['content'].append(
                {
                    "type": "text",
                    "text": text_query_input
                })
            # print(messages)
            if 'gpt' in self.args.LLM:
                response = self.chat_openai(messages)
            else:
                response = self.msg_input_qwen(messages)
            # print(response)
            records = split_string_by_multi_markers(response, [self.record_delimiter, self.completion_delimiter])
            processed_records = process_extracted_records(records, self.tuple_delimiter)
            if processed_records:
                # 传递passage信息到add_record_to_graph方法
                avg_record_emb = self.add_record_to_graph(processed_records, passage_info)
                if self.args.stop_detect == 'graph_embedding':
                    graph_embedding = self.message_passing()
                    if not isinstance(graph_embedding, list) and not isinstance(avg_record_emb, list):
                        retrievals, marker = self.anyrag_stop_detection(query, avg_record_emb, graph_embedding)
                        retrieved_list.extend(retrievals)
                else:
                    # 根据参数选择是否使用并行化
                    if getattr(self.args, 'enable_parallel', False):
                        max_workers = getattr(self.args, 'max_workers', None)
                        matched_subgraphs = self.graph_matching_stop_detection_parallel(
                            query_graph, anchor_node, threshold=self.threshold, max_workers=max_workers,
                            recent_nodes=self.path_recent_nodes)
                    else:
                        matched_subgraphs = self.graph_matching_stop_detection(query_graph, anchor_node,
                                                                               threshold=self.threshold,
                                                                               recent_nodes=self.path_recent_nodes)
                    retrieved_list.append(mm_node)
                    if not isinstance(matched_subgraphs, str):
                        graph_str = serialize_kg_to_prompt(matched_subgraphs, self.args.add_passage_node)
                        self.find_subgraph += 1
                        marker = True
                        break
            count += 1
        fine_grain_idx_end = time.time()
        self.fine_grain_idx_time += (fine_grain_idx_end - fine_grain_idx_start)
        if self.args.use_dpr and len(dpr_result) < threshold:
            while count < threshold:
                mm_node = retrieved_docs[count]
                if mm_node.metadata['modality'] == 'image':
                    if 'webqa' in mm_node.metadata['path'] or 'ywangnx' in mm_node.metadata['path']:
                        image_path = mm_node.metadata['path']
                    elif 'http' in mm_node.metadata['path'] or 'https' in mm_node.metadata['path']:
                        image_path = mm_node.metadata['path']
                    elif '../' in mm_node.metadata['path']:
                        image_path = mm_node.metadata['path']
                    else:
                        image_path = self.image_path_mmqa + mm_node.metadata['path']
                    dpr_result.append(image_path)
                else:
                    dpr_result.append(mm_node.page_content)
                count += 1

        if self.args.stop_detect == 'graph_embedding' or graph_str == '':
            graph_str = serialize_kg_to_prompt(self.Graph, self.args.add_passage_node)
        if self.args.stop_detect == 'ib':
            self.IBModel.reset()
        self.iteration_end_list.append(count)
        return retrieved_list, graph_str, dpr_result, query_graph

    def generate_graph_summary_from_string(self, query, graph_str):
        # 解析graph_str以提取passage节点信息
        passage_nodes = []
        if graph_str:
            lines = graph_str.split('\n')
            for line in lines:
                if line.startswith('("passage"'):
                    # 提取passage节点信息
                    # 格式: ("passage"[SEP]{node}[SEP]{node_type}[SEP]{desc})[REC/END]
                    parts = line.split('[SEP]')
                    if len(parts) >= 4:
                        passage_type = parts[2]  # 类型 (image/text)
                        passage_desc = parts[3].split(')')[0]  # 描述内容
                        passage_nodes.append({
                            'type': passage_type,
                            'desc': passage_desc
                        })

        messages = [{"role": "system",
                     "content": 'You are an AI assistant that summarizes the multimodal knowledge graph information to answer a given query.'}]

        # 构建用户消息内容
        user_content = []

        # 添加文本内容
        user_content.append({
            "type": "text",
            "text": f'Please summarize the given graph shortly to answer this question query:{query}\nHere is the graph:\n{graph_str}'
        })

        # 添加图像内容（如果有图片passage）
        for passage in passage_nodes:
            if passage['type'] == 'image':
                # 处理图片路径
                image_path = passage['desc']
                if not image_path.startswith('/') and not image_path.startswith('http'):
                    # 相对路径，需要加上基础路径
                    if self.args.dataset == 'mmqa':
                        image_path = self.image_path_mmqa + image_path
                    else:
                        image_path = self.image_path_webqa + image_path

                # 添加图片到消息中
                if 'gpt' in self.args.LLM:
                    try:
                        base64_image = self.encode_image(image_path)
                        user_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "low"
                            }
                        })
                    except Exception as e:
                        print(f"Error encoding image {image_path}: {e}")
                else:
                    user_content.append({
                        "type": "image",
                        "image": image_path
                    })

        messages.append({
            "role": "user",
            "content": user_content
        })

        if 'gpt' in self.args.LLM:
            graph_summary = self.chat_openai(messages)
        else:
            graph_summary = self.msg_input_qwen(messages)
        return graph_summary

    def anyrag_stop_detection(self, query, avg_record_emb, graph_embedding):
        message = [{"role": "system",
                    "content": system_prompt_ner}]
        if self.args.stop_detect == 'entity':
            retrieved_list = []
            message.append(
                {
                    "role": "user",
                    "content": f'Here is the real query, please extract key entities for answering this query: {query}',
                }
            )
            if 'gpt' in self.args.LLM:
                response = self.chat_openai(message, model=self.args.LLM)
            else:
                response = self.msg_input_qwen(message)
            search_ent_list = cut_str(response.lower(), '<ENT>', "</ENT>")
            for entity in search_ent_list:
                if entity.strip() in self.Graph.nodes():
                    entity_desc = self.Graph.nodes[entity.strip()]['desc']
                    retrieved_list.append(f'Entity name: {entity}\nEntity Description: {entity_desc}')
            if retrieved_list:
                return retrieved_list, True
            else:
                return [], False

        elif self.args.stop_detect == 'ib':
            query_embedding = self.text_embedding(query)
            converged = self.IBModel.process_sample(query_embedding, avg_record_emb, graph_embedding)
            if converged:
                return [], True
            else:
                return [], False

    def clean_graph(self):
        self.Graph.clear()

    def add_triple_to_graph(self, graph, record):
        total_records_emb = []
        if record[0].lower().strip() == 'entity':
            if record[1] not in graph.nodes():
                record_emb = self.text_embedding(record[2])
                graph.add_node(record[1],
                               desc=record[2],
                               embedding=record_emb
                               )
            else:
                record_emb = self.text_embedding(record[2])
                graph.nodes[record[1]]['desc'] += f'\n{record[2]}'
                graph.nodes[record[1]]['embedding'] = 0.5 * (
                        record_emb + graph.nodes[record[1]]['embedding'])
            total_records_emb.append(record_emb)

        elif 'relation' in record[0].lower().strip():
            if record[1] not in graph.nodes():
                record_emb = self.text_embedding(record[1])
                graph.add_node(record[1], desc='', embedding=record_emb)
                total_records_emb.append(record_emb)
            if record[2] not in graph.nodes():
                record_emb = self.text_embedding(record[2])
                graph.add_node(record[2], desc='', embedding=record_emb)
                total_records_emb.append(record_emb)
            if (record[1], record[2]) in graph.edges():
                record_emb = self.text_embedding(record[4])
                graph[record[1]][record[2]]['desc'] += f'\n{record[4]}'
                graph[record[1]][record[2]]['embedding'] = 0.5 * (
                        record_emb + graph[record[1]][record[2]]['embedding'])
            else:
                record_emb = self.text_embedding(record[4])
                graph.add_edge(record[1], record[2], relation=record[3], desc=record[4],
                               embedding=record_emb)
            total_records_emb.append(record_emb)
        return total_records_emb

    def add_record_to_graph(self, records: list, passage_info=None):
        total_records_emb = []
        for record in records:
            try:
                # 注意：add_triple_to_graph返回的是一个列表，我们需要扩展而不是替换total_records_emb
                record_embs = self.add_triple_to_graph(self.Graph, record)
                total_records_emb.extend(record_embs)

                # 为GraphRAG evictor注册实体映射
                if self.graphrag_evictor is not None:
                    if record[0].lower().strip() == 'entity':
                        # 为实体节点注册映射
                        entity_name = record[1]
                        cache_key = f"entity_{entity_name}"
                        self.register_kv_chunk_entity_mapping(cache_key, entity_name)
                    elif 'relation' in record[0].lower().strip():
                        # 为关系边注册映射
                        src_entity = record[1]
                        tgt_entity = record[2]
                        cache_key = f"relation_{src_entity}_{record[3]}_{tgt_entity}"
                        self.register_kv_chunk_entity_mapping(cache_key, src_entity)
                        self.register_kv_chunk_entity_mapping(cache_key, tgt_entity)

            except IndexError:
                print(f'IndexError in record: {record}')
                continue

        # 如果提供了passage信息，则添加passage节点
        if passage_info is not None:
            passage_id = passage_info.get('id')
            passage_type = passage_info.get('type')  # 'image' or 'text'
            passage_content = passage_info.get('content')

            if passage_id and passage_type and passage_content:
                # 创建passage节点ID
                passage_node_id = f"passage_{passage_type}_{passage_id}"

                # 如果节点不存在，则添加节点
                if passage_node_id not in self.Graph.nodes():
                    # 根据类型选择合适的嵌入方法Image_embedding
                    if passage_type == 'image':
                        # 确保使用正确的图片路径
                        if '../' in passage_content:
                            image_path = passage_content
                        elif self.args.dataset == 'mmqa':
                            image_path = self.image_path_mmqa + passage_content
                        else:
                            image_path = self.image_path_webqa + passage_content
                        passage_emb = self.image_embedding(image_path)
                    else:
                        passage_emb = self.text_embedding(passage_content)

                    self.Graph.add_node(passage_node_id,
                                        desc=passage_content,
                                        type=passage_type,
                                        embedding=passage_emb)
                    total_records_emb.append(passage_emb)

                    # 为passage节点注册映射
                    if self.graphrag_evictor is not None:
                        cache_key = f"passage_{passage_node_id}"
                        self.register_kv_chunk_entity_mapping(cache_key, passage_node_id)

                # 建立passage节点与所有抽取的三元组之间的"exist in"关系
                for record in records:
                    if record[0].lower().strip() == 'entity':
                        # 建立实体与passage的关系
                        entity_node_id = record[1]
                        if entity_node_id in self.Graph.nodes():
                            # 添加"exist in"关系边
                            if not self.Graph.has_edge(entity_node_id, passage_node_id):
                                relation_emb = self.text_embedding(f"{entity_node_id} exists in {passage_node_id}")
                                self.Graph.add_edge(entity_node_id, passage_node_id,
                                                    relation="exist in",
                                                    desc=f"{entity_node_id} exists in {passage_node_id}",
                                                    embedding=relation_emb)
                                total_records_emb.append(relation_emb)

        if total_records_emb:
            # 检查并统一张量维度
            normalized_embs = []
            for emb in total_records_emb:
                if emb.dim() == 2 and emb.size(0) == 1:
                    # 如果是[1, 512]的形状，压缩第一维
                    normalized_embs.append(emb.squeeze(0))
                elif emb.dim() == 1:
                    # 如果是[512]的形状，保持不变
                    normalized_embs.append(emb)
                else:
                    # 其他情况，保持原样
                    normalized_embs.append(emb)

            # 更新evictor的图信息和PageRank分数
            if self.graphrag_evictor is not None:
                self.compute_and_store_pagerank()

            return torch.mean(torch.stack(normalized_embs), dim=0)
        else:
            return []

    def get_pagerank_scores_tensor(self):
        """
        获取PageRank分数的张量表示，用于传递给LMCache
        """
        if not hasattr(self, 'pagerank_scores') or not self.pagerank_scores:
            # 如果还没有计算PageRank分数，则计算一次
            self.compute_and_store_pagerank()

        if self.pagerank_scores:
            # 按照图中节点的顺序排列PageRank分数
            node_list = list(self.Graph.nodes())
            pagerank_tensor = torch.tensor([self.pagerank_scores.get(node, 0.0) for node in node_list],
                                           dtype=torch.float32)
            # 归一化PageRank分数到[0,1]范围
            if pagerank_tensor.max() > 0:
                pagerank_tensor = pagerank_tensor / pagerank_tensor.max()
            return pagerank_tensor
        else:
            return None

    def get_cache_stats(self):
        """
        获取缓存统计信息
        """
        if self.graphrag_evictor is not None:
            return self.graphrag_evictor.get_cache_stats()
        return {}

    def get_pagerank_scores_for_lmcache(self, triples):
        """
        为LMCache获取PageRank分数张量

        Args:
            triples: 三元组列表 [(source, relation, target), ...]

        Returns:
            torch.Tensor: PageRank分数张量，用于LMCache的graph_attention_scores
        """
        # 确保PageRank分数已计算
        if not hasattr(self, 'pagerank_scores') or not self.pagerank_scores:
            self.compute_and_store_pagerank()

        if self.pagerank_scores and len(self.pagerank_scores) > 0:
            # 为每个三元组获取对应的PageRank分数
            # 这里我们使用源实体和目标实体的PageRank分数的平均值
            pagerank_list = []
            for s, r, t in triples:
                s_score = self.pagerank_scores.get(s, 0.0)
                t_score = self.pagerank_scores.get(t, 0.0)
                # 使用源实体和目标实体PageRank分数的平均值
                avg_score = (s_score + t_score) / 2.0
                pagerank_list.append(avg_score)

            pagerank_tensor = torch.tensor(pagerank_list, dtype=torch.float32)
            # 归一化PageRank分数到[0,1]范围
            if pagerank_tensor.max() > 0:
                pagerank_tensor = pagerank_tensor / pagerank_tensor.max()
            return pagerank_tensor
        else:
            # 如果没有PageRank分数，返回一个默认的张量
            return torch.zeros(len(triples), dtype=torch.float32)


    def get_current_graph_triples(self):
        """
        获取当前图中的所有三元组

        Returns:
            list: 三元组列表 [(source, relation, target), ...]
        """
        triples = []
        for s, t, edge_data in self.Graph.edges(data=True):
            relation = edge_data.get('relation', '')
            triples.append((s, relation, t))
        return triples

    def message_passing(self):
        """
            在 NetworkX 图 self.Graph 上执行消息传递。
            每个节点的 embedding 被更新为其自身、邻居节点和相连边的 embedding 的平均值。
            """
        new_embeddings = {}
        total_embeddings_list = []
        for node in self.Graph.nodes():
            current_emb = self.Graph.nodes[node]['embedding']
            neighbor_embs = []
            edge_embs = []
            # 遍历所有邻居节点及其对应的边
            for neighbor in self.Graph.neighbors(node):
                neighbor_emb = self.Graph.nodes[neighbor]['embedding']
                neighbor_embs.append(neighbor_emb)
                # 获取边的 embedding
                edge_emb = self.Graph[node][neighbor]['embedding']
                edge_embs.append(edge_emb)
            # 合并所有 embedding：自身 + 邻居节点 + 相连边
            all_embeddings = [current_emb] + neighbor_embs + edge_embs
            # 计算平均值
            new_embedding = np.mean(all_embeddings, axis=0)
            # 暂存新 embedding，避免在迭代中修改图结构
            new_embeddings[node] = new_embedding
        # 统一更新所有节点的 embedding
        for node, emb in new_embeddings.items():
            self.Graph.nodes[node]['embedding'] = emb
            total_embeddings_list.append(torch.tensor(emb))
        if total_embeddings_list:
            return torch.mean(torch.stack(total_embeddings_list), dim=0)
        else:
            return []

    def add_caption(self, query_image: dict, query: str, LLM_caption=False):
        if LLM_caption:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text",
                         "text": 'Please shortly describe this image with respect to this question: ' + query},
                        {"type": "image", "image": self.image_path + query_image['path'][0]},
                    ],
                },
            ]
            query_image['caption'] = [self.msg_input_qwen(messages)]
        else:
            query_image['caption'] = self.image_to_text(self.image_path + query_image['path'][0])[0][
                'generated_text']

    def add_caption_title(self, query_image: dict):
        query_image['caption'] = query_image['title']

    def msg_input_qwen(self, messages, ttft=None):
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        self.token_usage += inputs.input_ids.size(1)
        if self.args.use_vllm:
            # 检查是否使用远程vLLM API
            if hasattr(self, 'vllm_client'):
                # 使用远程vLLM API
                try:
                    # 将messages转换为适合API的格式
                    api_messages = []
                    for msg in messages:
                        api_msg = {"role": msg["role"], "content": []}
                        if isinstance(msg["content"], str):
                            api_msg["content"] = msg["content"]
                        elif isinstance(msg["content"], list):
                            for item in msg["content"]:
                                if item["type"] == "text":
                                    api_msg["content"].append(item)
                                elif item["type"] == "image":
                                    # 对于远程API，我们需要将图像转换为URL或base64
                                    if 'http' in item["image"] or 'https' in item["image"]:
                                        api_msg["content"].append({"type": "image_url", "image_url": {
                                            "url": item["image"]
                                        }, })
                                    else:
                                        try:
                                            with open(item["image"], "rb") as f:
                                                encoded_image = base64.b64encode(f.read())
                                            encoded_image_text = encoded_image.decode("utf-8")
                                        except:
                                            print(f"Error reading image file: {item['image']}")
                                            continue
                                        base64_qwen = f"data:image;base64,{encoded_image_text}"
                                        api_msg["content"].append({"type": "image_url", "image_url": {
                                            "url": base64_qwen
                                        }, })
                        api_messages.append(api_msg)
                    if ttft is not None:
                        self.ttft_list.append((time.time() - ttft) / 60)
                    chat_completion = self.vllm_client.chat.completions.create(
                        messages=api_messages,
                        model=self.vllm_model,
                        max_tokens=4096,
                        temperature=0.0,
                        # stream=True,
                    )
                    return chat_completion.choices[0].message.content
                except Exception as e:
                    print(f"Error calling remote vLLM API: {e}")
                    # 出错时回退到默认响应
                    return "Error occurred while processing the request."
            else:
                if hasattr(self, 'lmcache_integrator') and self.lmcache_integrator is not None:

                    # 使用LMCache集成器准备graph attention scores
                    graph_scores = self.lmcache_integrator.prepare_graph_attention_scores()
                    if graph_scores is not None:
                        print(f"LMCache graph attention scores prepared with shape: {graph_scores.shape}")

                sampling_params = SamplingParams(max_tokens=4096)
                generated_ids = self.llm.generate(inputs["input_ids"], sampling_params=sampling_params)
                # 修复生成结果处理
                if isinstance(generated_ids[0], list):  # 处理不同版本的vLLM输出格式
                    output_text = self.processor.batch_decode(
                        [gen_id.outputs[0].token_ids for gen_id in generated_ids],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )
                else:
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in
                        zip(inputs.input_ids, generated_ids[0].outputs[0].token_ids)
                    ]
                    output_text = self.processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )

                return output_text[0]
        else:
            # Inference: Generation of the output
            generated_ids = self.model.generate(**inputs,
                                                max_new_tokens=4096,

                                                )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return output_text[0]

    def chat(self, query_text, query_image=None, support_text='',
             graph_string='',
             dpr_result='',
             ttft=None):
        if self.args.LLM in ['qwen', 'mimo']:
            return self.chat_qwen(query_text, query_image, support_text, graph_string, dpr_result, ttft)
        elif self.args.LLM == 'gpt-4o':
            return self.chat_gpt(query_text, query_image, support_text, graph_string, dpr_result)

    def _prepare_chat_messages(self, query_text, query_image=None, support_text='', support_text_rag=''):
        """
        准备聊天消息，供chat方法使用
        """
        messages = [{"role": "system",
                     "content": system_prompt_cot if self.args.rag == 'none' else system_prompt}]

        if query_image is not None:
            content_list = []
            for image in query_image:
                content_list.append({"type": "image",
                                     "image": image['path'][0] if 'webqa' in image['path'][
                                         0] or 'ywangnx' in image['path'][0] else self.image_path_mmqa +
                                                                                  image['path'][0]})
                if self.args.image_caption:
                    content_list.append({"type": "text",
                                         "text": f'Here is the description of this image:\n' +
                                                 image['caption'][0]})
                messages.append(
                    {
                        "role": "user",
                        "content": content_list,
                    }
                )
            if len(messages) == 1:
                messages.append(
                    {"role": "user",
                     "content": 'Here is the real question:\n' + query_text}
                )
            else:
                messages[-1]["content"].append(
                    {"type": "text",
                     "text": 'Here is the real question:\n' + query_text}
                )
            if support_text:
                sup_text = f'\n\nHere is the supporting materials to help you answer the question:\n{support_text}'
                if isinstance(messages[-1]["content"], list):
                    messages[-1]["content"][-1]['text'] += sup_text
                elif isinstance(messages[-1]["content"], str):
                    messages[-1]["content"] += sup_text
                else:
                    print(messages)
                    raise ValueError('Unsupported type')
            if support_text_rag:
                sup_text = f'\nHere is some graph information that might be helpful:\n{support_text_rag}'
                if isinstance(messages[-1]["content"], list):
                    messages[-1]["content"][-1]['text'] += sup_text
                elif isinstance(messages[-1]["content"], str):
                    messages[-1]["content"] += sup_text
                else:
                    print(messages)
                    raise ValueError('Unsupported type')
        else:
            messages.append(
                {
                    "role": "user",
                    "content": 'Here is the real question:\n' + query_text
                }
            )
            if support_text:
                messages[-1]["content"] += \
                    f'\nHere is the supporting material to help you answer the question:\n{support_text}'

            if support_text_rag:
                messages[-1]["content"] += \
                    f'\nHere is the some graph information that might be helpful:\n{support_text_rag}'

        return messages

    def chat_qwen(self, query_text, query_image=None, support_text='', graph_string='', dpr_result='', ttft=None):
        messages = [{"role": "system",
                     "content": system_prompt_cot if self.args.rag == 'none' else system_prompt_mramg}]
        if query_image is not None:
            content_list = []
            for image in query_image:
                content_list.append({"type": "image",
                                     "image": image['path'][0] if 'webqa' in image['path'][
                                         0] or 'ywangnx' in image['path'][0] else self.image_path_mmqa +
                                                                                  image['path'][0]})
                if self.args.image_caption:
                    content_list.append({"type": "text",
                                         "text": f'Here is the description of this image:\n' +
                                                 image['caption'][0]})
                messages.append(
                    {
                        "role": "user",
                        "content": content_list,
                    }
                )
            if len(messages) == 1:
                messages.append(
                    {"role": "user",
                     "content": 'Here is the real question:\n' + query_text}
                )
            else:
                messages[-1]["content"].append(
                    {"type": "text",
                     "text": 'Here is the real question:\n' + query_text}
                )
            if support_text:
                sup_text = f'\n\nHere is the supporting materials to help you answer the question:\n{support_text}'
                if isinstance(messages[-1]["content"], list):
                    messages[-1]["content"][-1]['text'] += sup_text
                elif isinstance(messages[-1]["content"], str):
                    messages[-1]["content"] += sup_text
                else:
                    print(messages)
                    raise ValueError('Unsupported type')
            if graph_string:
                sup_text = f'\nHere is some graph information that might be helpful:\n{support_text}'
                if isinstance(messages[-1]["content"], list):
                    messages[-1]["content"][-1]['text'] += sup_text
                elif isinstance(messages[-1]["content"], str):
                    messages[-1]["content"] += sup_text
                else:
                    print(messages)
                    raise ValueError('Unsupported type')
                # messages[0]['content'][-1]['text'] = shrink_input_size(messages[0]['content'][-1]['text'], max_token)
        else:
            messages.append(
                {
                    "role": "user",
                    "content": 'Here is the real question:\n' + query_text
                    # + ' Direct response me with the answer, not any other words.'
                    ,
                }
            )
            if support_text:
                messages[-1]["content"] += \
                    f'\nHere is the supporting material to help you answer the question:\n{support_text}'

            if graph_string:
                messages[-1]["content"] += \
                    f'\nHere is the some graph information that might be helpful:\n{support_text}'
        # Preparation for inference
        if dpr_result:
            dpr_str = []
            image_list = []
            for dpr_chunk in dpr_result:
                if is_path(dpr_chunk) or is_url(dpr_chunk):
                    image_list.append(
                        {"type": "image",
                         "image": dpr_chunk}
                    )
                else:
                    dpr_str.append(dpr_chunk)
            dpr_total = '\n\n'.join(dpr_str)
            dpr_string = '\nHere are some dpr retrieved texts and images that might be helpful:\n' + dpr_total

            if isinstance(messages[-1]["content"], list):
                messages[-1]["content"][-1]['text'] += dpr_string
                messages[-1]["content"].extend(image_list)
            elif isinstance(messages[-1]["content"], str):
                messages[-1]["content"] += dpr_string
                messages.append({
                    "role": "user",
                    "content": image_list,
                })

            else:
                print(messages)
                raise ValueError('Unsupported type')
        try:
            return self.msg_input_qwen(messages, ttft)
        except TypeError:
            json.dump(messages, open('messages.json', 'w'), indent=2)
            print(messages)
            raise TypeError('Error in message format')

