from utils import *
import networkx as nx
import numpy as np
import torch
from networkx.algorithms.isomorphism import DiGraphMatcher


# 独立函数，用于进程并行
def _process_candidate_pairwise_worker(args):
    """进程并行的工作函数 - pairwise matching"""
    candidate_data, graph_data, query_graph_data, anchor_node, threshold = args
    # 重建主图
    G = nx.DiGraph()
    for node, data in graph_data['nodes'].items():
        G.add_node(node, **data)
    for u, v, data in graph_data['edges']:
        G.add_edge(u, v, **data)

    # 重建查询图
    query_graph = nx.DiGraph()
    for node, data in query_graph_data['nodes'].items():
        query_graph.add_node(node, **data)
    for u, v, data in query_graph_data['edges']:
        query_graph.add_edge(u, v, **data)

    # 节点和边匹配函数
    def node_match(n1, n2):
        return cosine_sim(n1['embedding'], n2['embedding']) > threshold

    def edge_match(e1, e2):
        return cosine_sim(e1['embedding'], e2['embedding']) > threshold

    candidate, data = candidate_data

    # 检查anchor node相似度
    sim = cosine_sim(data['embedding'], query_graph.nodes[anchor_node]['embedding'])
    if sim <= threshold:
        return None

    # 执行子图同构匹配
    GM = DiGraphMatcher(G, query_graph, node_match=node_match, edge_match=edge_match)
    matched_subgraphs = []

    for mapping in GM.subgraph_isomorphisms_iter():
        if mapping.get(candidate, None) != anchor_node:
            continue
        sub_nodes = list(mapping.keys())
        subgraph = G.subgraph(sub_nodes).copy()
        similarity = calculate_subgraph_similarity(query_graph, subgraph, mapping)
        # 序列化子图以便进程间传输
        subgraph_data = {
            'nodes': {n: dict(subgraph.nodes[n]) for n in subgraph.nodes()},
            'edges': [(u, v, dict(subgraph[u][v])) for u, v in subgraph.edges(data=True)]
        }
        matched_subgraphs.append((subgraph_data, similarity))

    # 计算并返回PageRank分数
    try:
        pagerank_scores = nx.pagerank(G)
    except Exception as e:
        print(f"Failed to compute PageRank scores in worker: {e}")
        pagerank_scores = {}

    return matched_subgraphs, pagerank_scores


def _process_candidate_path_worker(args):
    """进程并行的工作函数 - path matching"""
    candidate_data, graph_data, query_graph_data, anchor_node, threshold, agg_method, agg_weights, recent_nodes = args

    # 重建主图
    G = nx.DiGraph()
    for node, data in graph_data['nodes'].items():
        G.add_node(node, **data)
    for u, v, data in graph_data['edges']:
        G.add_edge(u, v, **data)

    # 重建查询图
    query_graph = nx.DiGraph()
    for node, data in query_graph_data['nodes'].items():
        query_graph.add_node(node, **data)
    for u, v, data in query_graph_data['edges']:
        query_graph.add_edge(u, v, **data)

    # 余弦相似度函数
    def cosine_sim(a, b):
        if isinstance(a, torch.Tensor):
            a = a.detach().cpu().numpy()
        if isinstance(b, torch.Tensor):
            b = b.detach().cpu().numpy()
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # 聚合函数
    def aggregate_embeddings(embeddings, method="mean", weights=None):
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

    candidate, data = candidate_data

    # 检查anchor node相似度
    sim = cosine_sim(data['embedding'], query_graph.nodes[anchor_node]['embedding'])
    if sim <= threshold:
        return None

    GM = DiGraphMatcher(G, query_graph,
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
                path = nx.shortest_path(G, source=candidate, target=data_target)
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
                emb = G.nodes[path[i]]['embedding']
                if isinstance(emb, torch.Tensor):
                    emb = emb.detach().cpu().numpy()
                path_embs.append(emb)
                if i < path_length - 1:
                    emb = G[path[i]][path[i + 1]]['embedding']
                    if isinstance(emb, torch.Tensor):
                        emb = emb.detach().cpu().numpy()
                    path_embs.append(emb)

            if agg_method == "weighted":
                weights = agg_weights if agg_weights is not None else [1.0] * len(path_embs)
            else:
                weights = None
            path_emb_mean = aggregate_embeddings(path_embs, method=agg_method, weights=weights)
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
                pair_emb = aggregate_embeddings([edge_emb, node_emb], method="weighted", weights=agg_weights)
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
                subgraph = G.subgraph(sub_nodes).copy()
                # 序列化子图以便进程间传输
                subgraph_data = {
                    'nodes': {n: dict(subgraph.nodes[n]) for n in subgraph.nodes()},
                    'edges': [(u, v, dict(subgraph[u][v])) for u, v in subgraph.edges(data=True)]
                }
                matched_subgraphs.append((subgraph_data, avg_sim))

    # 计算并返回PageRank分数
    try:
        pagerank_scores = nx.pagerank(G)
    except Exception as e:
        print(f"Failed to compute PageRank scores in worker: {e}")
        pagerank_scores = {}

    return matched_subgraphs, pagerank_scores
