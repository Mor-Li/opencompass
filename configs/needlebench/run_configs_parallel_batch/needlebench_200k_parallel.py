from mmengine.config import read_base
with read_base():
    from ...datasets.needlebench.needlebench_8k.parallel.needlebench_parallel_diffbatch import needlebench_datasets
    from ..base import chat_models_8k, chat_models_32k, chat_models_200k
    from ..base import infer as infer, eval

datasets = [*needlebench_datasets]
models = [*chat_models_8k, *chat_models_32k, *chat_models_200k]
work_dir = './outputs/needlebench/parallel_batch'
