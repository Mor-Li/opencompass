from mmengine.config import read_base
with read_base():
    from ...datasets.needlebench.needlebench_32k.needlebench import needlebench_datasets
    from ..base import chat_models_32k, chat_models_200k
    from ..base import infer as infer, eval
    from ...summarizers.needlebench_32k import summarizer

datasets = [*needlebench_datasets]
models = [*chat_models_32k, *chat_models_200k]
work_dir = './outputs/needlebench'
