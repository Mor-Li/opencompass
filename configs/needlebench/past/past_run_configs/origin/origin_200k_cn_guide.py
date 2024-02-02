from mmengine.config import read_base
with read_base():
    from ..base import cdme200k_datasets_cn
    from ..base import chat_models_200k_cn
    from ..base import infer, eval

datasets = [*cdme200k_datasets_cn]
models = [*chat_models_200k_cn]
