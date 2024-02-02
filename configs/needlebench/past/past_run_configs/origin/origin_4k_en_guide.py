from mmengine.config import read_base
with read_base():
    from ..base import cdme4k_datasets_en
    from ..base import chat_models_4k_en, chat_models_8k_en, chat_models_32k_en
    from ..base import infer, eval

datasets = [*cdme4k_datasets_en]
models = [*chat_models_4k_en, *chat_models_8k_en, *chat_models_32k_en]
