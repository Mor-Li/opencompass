from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.needlebench.origin import NeedleBenchOriginDataset
from opencompass.datasets.needlebench.origin import NeedleBenchOriginEvaluator
from opencompass.datasets.needlebench.origin import needlebench_postprocess
from opencompass.datasets.needlebench.origin import needlebench_dataset_postprocess

needlebench_reader_cfg = dict(input_columns=['prompt'], output_column='answer')

needlebench_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='{prompt}'),
                dict(role='BOT', prompt='{answer}\n'),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

needlebench_eval_cfg = dict(
    evaluator=dict(type=NeedleBenchOriginEvaluator),
    pred_postprocessor=dict(type=needlebench_postprocess),
    dataset_postprocessor=dict(type=needlebench_dataset_postprocess),
    pred_role='BOT',
)

context_lengths = list([9000, 13000, 17000, 21000, 25000, 29000, 31000, 32000])
depths_list = [0, 10, 21, 31, 42, 52, 63, 73, 84, 94, 100]
document_depth_percent_intervals = 20
document_depth_percent_interval_type = 'linear'

base_path = 'opencompass/needlebench'
file_list = ['PaulGrahamEssays.jsonl']
needlebench_en_datasets = []
needle_file_name = 'needles.jsonl'

for original_context_length in context_lengths:
    for depth_percent in depths_list:
        dataset_dict = {
            'abbr': f'Length{original_context_length}'
            f'Depth{int(depth_percent)}_origin_en_32k',
            'type': NeedleBenchOriginDataset,
            'path': base_path,
            'length': original_context_length,
            'depth': int(depth_percent),
            'tokenizer_model': 'gpt-4',
            'file_list': file_list,
            'num_repeats_per_file': 10,
            'length_buffer': 3000,
            'guide': True,
            'language': 'English',
            'needle_file_name': needle_file_name,
            'reader_cfg': needlebench_reader_cfg,
            'infer_cfg': needlebench_infer_cfg,
            'eval_cfg': needlebench_eval_cfg,
        }
        needlebench_en_datasets.append(dataset_dict)

file_list = ['zh_finance.jsonl']
needlebench_zh_datasets = []
needle_file_name = 'needles.jsonl'

for original_context_length in context_lengths:
    for depth_percent in depths_list:
        dataset_dict = {
            'abbr': f'Length{original_context_length}'
            f'Depth{int(depth_percent)}_origin_zh_32k',
            'type': NeedleBenchOriginDataset,
            'path': base_path,
            'length': original_context_length,
            'depth': int(depth_percent),
            'tokenizer_model': 'gpt-4',
            'file_list': file_list,
            'num_repeats_per_file': 10,
            'length_buffer': 200,
            'guide': True,
            'language': 'Chinese',
            'needle_file_name': needle_file_name,
            'reader_cfg': needlebench_reader_cfg,
            'infer_cfg': needlebench_infer_cfg,
            'eval_cfg': needlebench_eval_cfg,
        }
        needlebench_zh_datasets.append(dataset_dict)
