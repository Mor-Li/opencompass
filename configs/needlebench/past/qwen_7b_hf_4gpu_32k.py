from mmengine.config import read_base
with read_base():
    from ..datasets.needlebench.needlebench_32k.parallel.needlebench_parallel_en import needlebench_datasets
    from .base import qwen_7b_chat_hf_model_4gpu
    from .base import parallel_infer as infer, eval
    from ..summarizers.needlebench_32k import summarizer

datasets = [*needlebench_datasets]
models = [qwen_7b_chat_hf_model_4gpu]
work_dir = './outputs/needlebench'
