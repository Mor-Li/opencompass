from mmengine.config import read_base
with read_base():
    from ..base import cdme200k_parallel_datasets_en
    from ..base import chat_models_200k_en
    from ..base import parallel_infer as infer
    from ..base import eval as eval

datasets = [*cdme200k_parallel_datasets_en]
models = [*chat_models_200k_en]