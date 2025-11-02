from argparse import ArgumentParser
from ragas.dataset_schema import SingleTurnSample
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider_scorer import CiderScorer
from ragas.metrics import AnswerAccuracy, FaithfulnesswithHHEM, SemanticSimilarity
from faithfulness import compute_faithfulness_score
from utils import *
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
import asyncio

evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="Qwen/Qwen2.5-VL-7B-Instruct",
                                               openai_api_key="EMPTY",
                                               openai_api_base='http://localhost:8001/v1',
                                               max_tokens=4096,
                                               temperature=0, ))

vllm_llm = ChatOpenAI(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    base_url='http://localhost:8001/v1',
    api_key="sk-local-test",  # 这里不要用空字符串
    temperature=0.0,
    max_retries=5,
    timeout=60
)

embedder = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={'device': 'cuda'},
    encode_kwargs={"normalize_embeddings": True}
))


def calculate_bleu(reference, candidate):
    reference_tokens = [reference.split()]
    candidate_tokens = candidate.split()
    smoothie = SmoothingFunction().method1
    bleu_scores = {
        f"BLEU-{i}": sentence_bleu(reference_tokens, candidate_tokens, weights=(1 / i,) * i,
                                   smoothing_function=smoothie)
        for i in range(1, 5)
    }
    return bleu_scores


# 计算 ROUGE
def calculate_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return {k: v.fmeasure for k, v in scores.items()}


# 计算 CIDEr
def calculate_cider(references, candidates):
    scorer = CiderScorer(n=4, sigma=6.0)
    for ref, cand in zip(references, candidates):
        scorer += (cand, [ref])
    cider, _ = scorer.compute_score()
    return cider


async def calculate_metrics(answer_list,  # Model generated answer
                            gt_list,  # Ground truth answers
                            q_type_list,  # Query type
                            query_list,  # Query text
                            response_list,  # Model total response
                            support_text_list,  # Support text in query
                            query_image_list,  # Images in query
                            dpr_result_list,  # Retrieved documents
                            g_string_list,  # Generated string from MMKG
                            args):
    total_metric_dict = {
        'BLEU-1': 0,
        'BLEU-2': 0,
        'BLEU-3': 0,
        'BLEU-4': 0,
        'rouge1': 0,
        'rouge2': 0,
        'rougeL': 0,
        'acc': 0,
        'EM': 0,
        'semantic sim': 0,
        "faithfulness": 0,
    }
    results = []
    for answer, gt, q_type, query, response, support_text, query_image, dpr_result, g_string in zip(answer_list,
                                                                                                    gt_list,
                                                                                                    q_type_list,
                                                                                                    query_list,
                                                                                                    response_list,
                                                                                                    support_text_list,
                                                                                                    query_image_list,
                                                                                                    dpr_result_list,
                                                                                                    g_string_list):
        case_dict = {}
        metric_dict = {
            'BLEU-1': [],
            'BLEU-2': [],
            'BLEU-3': [],
            'BLEU-4': [],
            'rouge1': [],
            'rouge2': [],
            'rougeL': [],
            'acc': [],
            'EM': 0,
            'semantic sim': [],
            "faithfulness": [],
        }
        content_list = dpr_result.copy()
        # if g_string:
        #     content_list.append(g_string)
        for gt_answer in gt:
            if q_type == 'yesno':
                if answer.lower() == 'yes' and 'yes' in answer.lower():
                    metric_dict['EM'] += 1
            elif answer.lower() == gt_answer.lower():
                metric_dict['EM'] += 1
            for key, value in calculate_rouge(gt_answer.lower(), answer.lower()).items():
                if value:
                    metric_dict[key].append(value)
                else:
                    metric_dict[key].append(0)
            for key, value in calculate_bleu(gt_answer.lower(), answer.lower()).items():
                if value:
                    metric_dict[key].append(value)
                else:
                    metric_dict[key].append(0)
            sample = SingleTurnSample(
                user_input=query,
                response=answer,
                reference=gt_answer,
                retrieved_contexts=content_list
            )
            acc_scorer = AnswerAccuracy(llm=evaluator_llm)
            sim_scorer = SemanticSimilarity(embeddings=embedder)
            try:
                acc_score, sim_score = await asyncio.gather(
                    acc_scorer.single_turn_ascore(sample),
                    sim_scorer.single_turn_ascore(sample),
                )
            except Exception as e:
                print(f'Error {e} in acc or sim calculation')
                acc_score, sim_score = 0.0, 0.0
            try:
                faithfulness_score = await compute_faithfulness_score(
                    query, answer, content_list, vllm_llm
                )
            except Exception as e:
                print(f'Error {e} in faithfulness calculation')
                faithfulness_score = 0.0
            # evaluator_llm wrapped with ragas LLM Wrapper
            # acc_score = await acc_scorer.single_turn_ascore(sample)
            metric_dict['acc'].append(acc_score)

            # sim_score = await sim_scorer.single_turn_ascore(sample)
            metric_dict['semantic sim'].append(sim_score)
            # faithfulness_score = await compute_faithfulness_score(
            #    query, answer, content_list, vllm_llm
            # )
            metric_dict['faithfulness'].append(faithfulness_score)
            '''
            try:
                faithfulness_scorer = FaithfulnesswithHHEM(llm=evaluator_llm)
                faithfulness_score = await faithfulness_scorer.single_turn_ascore(sample)
                metric_dict['faithfulness'].append(faithfulness_score)
            except Exception as e:
                print(f'Error {e} in faithfulness calculation')
                metric_dict['faithfulness'].append(0.0)
            '''
        case_dict['query'] = query
        case_dict['gt_answer'] = gt
        case_dict['response'] = response
        case_dict['support_text'] = support_text
        case_dict['gen_answer'] = answer
        case_dict['query_image'] = query_image
        case_dict['metric'] = metric_dict
        case_dict['q_type'] = q_type
        case_dict['g_string'] = g_string
        if dpr_result:
            case_dict['dpr_result'] = dpr_result
        results.append(case_dict)
        for key, value in total_metric_dict.items():
            if key != 'EM':
                total_metric_dict[key] += max(metric_dict[key])
            else:
                total_metric_dict[key] += metric_dict[key] / len(metric_dict['acc'])
    return total_metric_dict, results
    # if not correct_answer:


async def main():
    with open(f'./results/result_{args.rag}_{args.dataset}.json', 'r') as f:
        result_dict = json.load(f)
    total_metric_dict, results = await calculate_metrics(answer_list=result_dict['ans_list'],
                                                         gt_list=result_dict['gt_list'],
                                                         q_type_list=result_dict['q_type_list'],
                                                         query_list=result_dict['query_list'],
                                                         response_list=result_dict['response_list'],
                                                         support_text_list=result_dict['support_text_list'],
                                                         query_image_list=result_dict['query_image_list'],
                                                         dpr_result_list=result_dict['dpr_result_list'],
                                                         g_string_list=result_dict['g_string_list'],
                                                         args=args)

    with open(f'./results/result_{args.rag}_{args.dataset}.json', 'w') as f:
        json.dump(convert_tensors_to_lists(results), f, indent=2)
    for key, value in total_metric_dict.items():
        print(f'{key}: {value / len(results)}')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default='mmqa')
    parser.add_argument("--rag", type=str, default='none')
    parser.add_argument("--use_dpr", action="store_true")
    args = parser.parse_args()
    asyncio.run(main())
