from mmengine.config import read_base
with read_base():
    from ..datasets.needlebench.needlebench import needlebench_datasets_32k
    from .base import chatglm3_6b_32k_model
    from .base import infer as infer, eval
    from ..summarizers.needlebench_200k import summarizer

datasets = [*needlebench_datasets_32k]
models = [chatglm3_6b_32k_model]
work_dir = './outputs/needlebench'
