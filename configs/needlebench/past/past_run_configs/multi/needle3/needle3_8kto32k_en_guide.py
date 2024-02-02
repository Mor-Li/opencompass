from mmengine.config import read_base
with read_base():
    from ...base import needle3_datasets_8kto32k_en
    from ...base import chat_models_32k_en
    from ...base import infer, eval

datasets = [*needle3_datasets_8kto32k_en]
models = [*chat_models_32k_en]
