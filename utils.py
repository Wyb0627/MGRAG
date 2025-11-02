import os

import PIL
import networkx as nx
import torch
import numpy as np
import random
import json
import gzip
import io
import re
import tiktoken
from math import isclose
from typing import Union
from PIL import Image
import base64
from io import BytesIO
import requests
import html
import spacy
from spacy.symbols import NOUN  # 也可选择NOUN或其他词性
from tqdm import trange
from pathlib import Path
from urllib.parse import urlparse
from typing import List


def read_jsonl(file_name: str) -> List:
    """
    Read a jsonl file and return a list of dictionaries.
    """
    data_list = []
    with open(file_name, 'r') as f:
        for line in f:
            data = json.loads(line)
            data_list.append(data)
    return data_list


def cosine_sim(a, b):
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    if isinstance(b, torch.Tensor):
        b = b.detach().cpu().numpy()
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def calculate_subgraph_similarity(query_graph, subgraph, mapping):
    total_sim = 0.0
    count = 0
    for data_node, query_node in mapping.items():
        sim = cosine_sim(subgraph.nodes[data_node]['embedding'],
                         query_graph.nodes[query_node]['embedding'])
        total_sim += sim
        count += 1
    for u, v, data in subgraph.edges(data=True):
        query_u = mapping[u]
        query_v = mapping[v]
        if query_graph.has_edge(query_u, query_v):
            sim = cosine_sim(data['embedding'],
                             query_graph.edges[query_u, query_v]['embedding'])
            total_sim += sim
            count += 1
    return total_sim / count if count > 0 else 0.0


def img_compression(img, compression_rate):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    # 创建内存缓冲区
    output_buffer = io.BytesIO()
    # 保存到内存缓冲区，指定 JPEG 格式和质量
    img.save(output_buffer, format='JPEG', quality=compression_rate, optimize=True)
    # 将指针重置到缓冲区开头
    output_buffer.seek(0)
    compressed_image = Image.open(output_buffer)
    return compressed_image


def node_edge_match(n1, n2, threshold):
    return cosine_sim(n1['embedding'], n2['embedding']) > threshold


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["GLOBAL_SEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def read_jsonl(filename):
    with open(filename, 'r') as f:
        data = [json.loads(l.strip()) for l in f.readlines()]
    return data


def read_jsonl_gz(filename):
    with gzip.open(filename, 'rt') as f:
        data = [json.loads(l.strip()) for l in f.readlines()]
    return data


def cut_str(input_str: str, start: str, end: str):
    pattern = re.escape(start.lower()) + r'(.*?)' + re.escape(end.lower())
    matches = re.findall(pattern, input_str.lower(), flags=re.DOTALL)
    # start_idx = input_str.index(start) + len(start)
    # end_idx = input_str.index(end)
    # return [input_str[start_idx:end_idx]]
    return [match.strip() for match in matches]


def clean_str(input) -> str:
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
    # If we get non-string input, just give it back
    if not isinstance(input, str):
        return input

    result = html.unescape(input.strip())
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)


def load_image_from_base64_and_save(img_base64, out_path: str, fail_list: list, image_url: str = ''):
    try:
        image = Image.open(BytesIO(base64.b64decode(img_base64)))
        image = image.convert('RGB')
    except:
        response = requests.get(image_url,
                                headers={'User-Agent': 'MMRAG/1.0 (Linux; ywangnx@connect.ust.hk)'},
                                stream=True
                                )
        print(f'Failed to load image from base64, try to download from {image_url}')
        try:
            image = Image.open(response.raw)
            image = image.convert('RGB')

        except:
            fail_list.append(image_url)
            print(f'Failed to open image from {image_url}')
            return False
    image.save(out_path, format='JPEG')
    #             image.verify()
    image.close()
    return True


def download_image_and_save(image_url: str, out_path: str):
    response = requests.get(image_url,
                            headers={'User-Agent': 'MMRAG/1.0 (Linux; ywangnx@connect.ust.hk)'},
                            stream=True
                            )
    image = Image.open(response.raw)
    image = image.convert('RGB')
    image.save(out_path, format='JPEG')
    #             image.verify()
    image.close()



def convert_tensors_to_lists(obj):
    """递归将PyTorch张量转换为列表，支持处理嵌套结构

    Args:
        obj: 输入对象，可以是dict/list/tuple/PyTorch张量等

    Returns:
        所有PyTorch张量被替换为列表的等效结构
    """
    if isinstance(obj, torch.Tensor):  # 基础情况：遇到张量时转换
        return obj.tolist()

    elif isinstance(obj, dict):  # 处理字典类型
        return {k: convert_tensors_to_lists(v) for k, v in obj.items()}

    elif isinstance(obj, (list, tuple)):  # 处理列表/元组类型
        return type(obj)(convert_tensors_to_lists(elem) for elem in obj)

    else:  # 其他类型直接返回（int/float/str等）
        return obj


def is_path(s: str) -> bool:
    if not isinstance(s, str) or not s.strip():
        return False
    # 检查是否包含路径分隔符，或是否是绝对路径，或是否是有效的相对路径组件
    # 一个路径至少应包含一个路径分隔符，或是一个有效的相对路径（如 "."、".."、"folder" 等）
    try:
        p = Path(s)
        # 尝试解析路径
        _ = p.parts  # 触发路径解析
        # 如果是绝对路径，直接返回 True
        if p.is_dir() or p.is_file():
            return True
        # 如果是相对路径，检查是否包含至少一个有效部分（排除空或仅分隔符）
    except Exception:
        return False
    return False


def is_url(s):
    """
    严格判断字符串 s 是否为完整的 URL（支持 http, https, ftp 等常见协议）
    要求整个字符串必须是一个合法 URL，不能包含前缀或后缀文本。
    """
    if not isinstance(s, str):
        return False

    s = s.strip()
    if not s:
        return False

    # 正则匹配完整 URL（从头到尾）
    # 支持协议：http, https, ftp, ftps 等
    # 允许端口、路径、查询参数、锚点
    url_pattern = re.compile(
        r'^(?:http|ftp)s?://'  # 协议
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # 域名
        r'localhost|'  # localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|'  # IPv4
        r'\[?[A-F0-9]*:[A-F0-9:]+\]?)'  # IPv6
        r'(?::\d+)?'  # 端口
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

    if not url_pattern.match(s):
        return False

    # 二次校验：使用 urlparse 确保结构合法
    try:
        result = urlparse(s)
        return bool(result.scheme and result.netloc)
    except Exception:
        return False


def remove_quotes(s):
    quotes = ('"', "'")
    if s and s[0] in quotes:
        s = s[1:]
    if s and s[-1] in quotes:
        s = s[:-1]
    return s


def shrink_input_size(llm_input: str, size: int) -> str:
    processed_input = llm_input.strip().split(' ')
    return ' '.join(processed_input[:size])


def test():
    # train = read_jsonl_gz('../dataset/multimodalqa/dataset/MMQA_train.jsonl.gz')
    dev = read_jsonl_gz('../dataset/multimodalqa/dataset/MMQA_dev.jsonl.gz')
    count_train = 0
    count_dev = 0
    imageq_list = []
    # for data in train:
    #     if data['metadata']['type'].lower() in ['imageq', 'textq']:
    #        count_train += 1
    #       imageq_list.append(data)
    for data in dev:
        if data['metadata']['type'].lower() in ['imageq']:
            count_dev += 1
            imageq_list.append(data)
        if 'Milo Murphy' in data['question']:
            print(data)
    print(count_train)
    print(count_dev)
    return imageq_list


def safe_equal(prediction: Union[bool, float, str],
               reference: Union[float, str],
               include_percentage: bool = False,
               is_close: float = False) -> bool:
    if prediction is None:
        return False
    elif type(prediction) == bool:
        # bool questions
        if prediction:
            return reference == 'yes'
        else:
            return reference == 'no'
    elif type(reference) == str and type(prediction) == str:
        # string questions
        prediction = prediction.strip().lower()
        reference = reference.strip().lower()
        return prediction == reference
    else:
        # number questions
        if include_percentage:
            gt_result = [reference / 100, reference, reference * 100]
        else:
            gt_result = [reference]
        for item in gt_result:
            try:
                if is_close:
                    if isclose(item, prediction, rel_tol=0.001):
                        return True
                precision = min(get_precision(prediction), get_precision(item))
                if round(prediction, precision) == round(item, precision):
                    return True
            except Exception:
                continue
        return False


def get_precision(gt_ans: float) -> int:
    precision = 5
    if '.' in str(gt_ans):
        precision = len(str(gt_ans).split('.')[-1])
    return precision


def read_txt_prompt(file_path):
    with open(file_path, 'r') as f:
        data = f.readlines()
    return data


def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """Split a string by multiple markers"""
    if not markers:
        return [content]
    content = content if content is not None else ""
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]


def num_tokens_from_messages(messages, model="gpt-4o", image_token=85):
    """Returns the number of tokens used by a list of messages."""
    encoding = tiktoken.encoding_for_model(model)
    total_tokens = 0
    for message in messages:
        # 每个消息的基础 token 开销
        total_tokens += 3

        # 处理 role 字段
        role = message.get("role", "")
        total_tokens += len(encoding.encode(role))

        # 处理 content 字段
        content = message.get("content", "")
        if isinstance(content, str):
            # 纯文本内容
            total_tokens += len(encoding.encode(content))
        else:
            # 多模态内容，遍历每个部分
            for part in content:
                if part["type"] == "text":
                    total_tokens += len(encoding.encode(part["text"]))
                elif part["type"] == "image_url":
                    total_tokens += image_token
    return total_tokens


def process_extracted_records(records, tuple_delimiter):
    processed_list = []
    for record in records:
        record = re.search(r"\((.*)\)", record)
        if record is None:
            continue
        record = record.group(1)
        record_attributes = split_string_by_multi_markers(
            record, [tuple_delimiter])
        record_attributes_after = unify_and_deduplicate(record_attributes)
        processed_list.append(clean_str(record_attributes_after))
    return processed_list


def serialize_kg_to_prompt(Graph: nx.DiGraph, add_passage_node=False):
    lines = []
    # 序列化实体节点
    for node, data in Graph.nodes(data=True):
        desc = data.get('desc', '')  # 获取描述，若不存在则为空字符串
        node_type = data.get('type', '')  # 获取节点类型

        # 如果是passage_node，使用不同的格式
        if node.startswith('passage_'):
            if add_passage_node:
                line = f'("passage"[SEP]{node}[SEP]{node_type}[SEP]{desc})[REC]'
            else:
                continue
        else:
            line = f'("entity"[SEP]{node}[SEP]{desc})[REC]'
        lines.append(line)

    # 序列化关系边
    for u, v, data in Graph.edges(data=True):
        relation = data.get('relation', '')  # 获取关系类型
        desc = data.get('desc', '')  # 获取关系描述
        line = f'("relation"[SEP]{u}[SEP]{v}[SEP]{relation}[SEP]{desc})[REC]'
        lines.append(line)
    if lines:
        lines[-1] = lines[-1].replace('[REC]', '[END]')  # 替换最后一行的结束标记
        # 拼接所有行，形成最终字符串
        return '\n'.join(lines)
    else:
        return ''


def unify_and_deduplicate(keywords, target_pos=NOUN):
    nlp = spacy.load("en_core_web_sm")
    lemma_dict = {}
    lemma_list = []
    for word in keywords:
        if not word.strip():  # 跳过空字符串或纯空格
            continue

        doc = nlp(word)
        if len(doc) == 0:  # 处理spacy解析失败的情况（如纯标点）
            lemma = word.lower()
        else:
            # 对每个Token强制统一词性并提取词根
            lemmas = []
            for token in doc:
                token.pos = target_pos  # 强制设置目标词性
                lemmas.append(token.lemma_.lower())
            lemma = " ".join(lemmas).strip()  # 处理多词短语
        if lemma not in lemma_list:
            # Remove all symbols
            lemma = re.sub(r'[^\w\s]', '', lemma)
            if lemma:
                lemma_list.append(lemma.strip())
        # 保留首个出现的词汇
        # if lemma not in lemma_dict:
        #     lemma_dict[lemma] = word.lower()

    return lemma_list
