from mmengine.config import read_base
with read_base():
    from ...datasets.needlebench.needlebench_4k.needlebench import needlebench_datasets as needlebench_4k_datasets
    from ..base import chat_models_200k_compare_needle_data
    from ..base import infer as infer, eval
    from ...summarizers.needlebench_4k import summarizer

datasets = sum([v for k, v in locals().items() if (k.endswith("_datasets") or k == 'datasets')], [])

models = [*chat_models_200k_compare_needle_data]
work_dir = './outputs/needlebench/compare_needle_data'
