from mmengine.config import read_base
with read_base():
    from ...datasets.needlebench.needlebench_200k.needlebench import needlebench_datasets
    from ..base import chat_models_200k
    from ..base import infer as infer, eval
    from ...summarizers.needlebench_200k import summarizer

datasets = [*needlebench_datasets]
models = [*chat_models_200k]
work_dir = './outputs/needlebench'
