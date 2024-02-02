from mmengine.config import read_base
with read_base():
    from ..base import cdme8kto32k_datasets_cn
    from ..base import chat_models_32k_cn
    from ..base import infer, eval

datasets = [*cdme8kto32k_datasets_cn]
models = [*chat_models_32k_cn]
