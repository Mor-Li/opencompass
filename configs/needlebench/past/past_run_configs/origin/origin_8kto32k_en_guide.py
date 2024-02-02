from mmengine.config import read_base
with read_base():
    from ..base import cdme8kto32k_datasets_en
    from ..base import chat_models_32k_en
    from ..base import infer, eval

datasets = [*cdme8kto32k_datasets_en]
models = [*chat_models_32k_en]
