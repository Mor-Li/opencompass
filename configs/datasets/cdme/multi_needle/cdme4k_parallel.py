from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.cdme.cdme_parallel import CDMEDataset
from opencompass.datasets.cdme.cdme_parallel import CDMEEvaluator
from opencompass.datasets.cdme.cdme_parallel import cdme_postprocess
from opencompass.datasets.cdme.cdme_parallel import cdme_dataset_postprocess
import math


def logistic(x, L=100, x0=50, k=0.1):
    return round(L / (1 + math.exp(-k * (x - x0))), 3)


def generate_linear_space(start, end, num):
    if num == 1:
        return [start]
    elif num < 1:
        raise ValueError("num must be at least 1.")
    step = (end - start) / (num - 1)
    return [start + step * i for i in range(num)]


def generate_depth_percents(intervals, interval_type):
    if interval_type == 'linear':
        return generate_linear_space(0, 100, intervals)
    elif interval_type == 'sigmoid':
        linear_space = generate_linear_space(0, 100, intervals)
        return [logistic(x) for x in linear_space]
    else:
        raise ValueError('Unsupported interval type')


cdme_reader_cfg = dict(input_columns=['prompt'], output_column='answer')

cdme_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='{prompt}'),
                dict(role='BOT', prompt='{answer}\n'),
            ]
        )
        ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

cdme_eval_cfg = dict(
    evaluator=dict(type=CDMEEvaluator),
    pred_postprocessor=dict(type=cdme_postprocess),
    dataset_postprocessor=dict(type=cdme_dataset_postprocess),
    pred_role='BOT')

context_lengths = list(range(1000, 5000, 1000))
document_depth_percent_intervals = 20
document_depth_percent_interval_type = "linear"

base_path = './data/CDME'
file_list = ['zh_finance.jsonl']
cdme4k_parallel_datasets = []

needle_file_name = 'predefined_examples.jsonl'

for original_context_length in context_lengths:
    dataset_dict = {
        'abbr': f'CDME_Length{original_context_length}',
        'type': CDMEDataset,
        'path': base_path,
        'needle_file_name': needle_file_name,
        'length': original_context_length,
        'depths': [int(i) for i in generate_depth_percents(
            document_depth_percent_intervals,
            document_depth_percent_interval_type)],
        'tokenizer_model': 'gpt-4',
        'file_list': file_list,
        'num_repeats_per_file': 10,
        'length_buffer': 200,
        'guide': True,
        'language': 'Chinese',
        'reader_cfg': cdme_reader_cfg,
        'infer_cfg': cdme_infer_cfg,
        'eval_cfg': cdme_eval_cfg
    }
    cdme4k_parallel_datasets.append(dataset_dict)
