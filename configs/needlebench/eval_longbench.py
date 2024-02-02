from mmengine.config import read_base
with read_base():
    from ..datasets.longbench.longbench import longbench_datasets
    from .base import chat_models_32k
    from .base import infer as infer, eval
    # from ..summarizers.needlebench_200k import summarizer

datasets = [*longbench_datasets]
models = [*chat_models_32k]
work_dir = './outputs/longbench'
