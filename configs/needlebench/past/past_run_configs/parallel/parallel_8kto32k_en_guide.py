from mmengine.config import read_base
with read_base():
    from ..base import cdme8kto32k_parallel_datasets_en
    from ..base import chat_models_8k_en, chat_models_32k_en
    from ..base import parallel_infer as infer
    from ..base import eval as eval

datasets = [*cdme8kto32k_parallel_datasets_en]
models = [*chat_models_32k_en]