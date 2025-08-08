import os
from opencompass.models import OpenAI
from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.pi_llm.pi_llm_gen import pi_llm_datasets

# Pick only one minimal configuration from test1
datasets = [
    dict(
        abbr='pi_llm_tiny',
        type=pi_llm_datasets[0]['type'],
        source_dict_path=pi_llm_datasets[0]['source_dict_path'],
        n_tracked_keys=[3],     # Minimal: just 3 keys
        n_tracked_updates=[2],  # Minimal: just 2 updates
        n_untracked_keys=0,
        n_untracked_updates=0,
        n_samples_per_config=2, # Just 2 samples total
        seed=42,
        reader_cfg=pi_llm_datasets[0]['reader_cfg'],
        infer_cfg=pi_llm_datasets[0]['infer_cfg'],
        eval_cfg=pi_llm_datasets[0]['eval_cfg']
    )
]

# Kimi-K2 model
models = [
    dict(
        type=OpenAI,
        abbr='kimi-k2',
        path='kimi-k2',
        key='ENV',  # Will get from $OPENAI_API_KEY
        # openai_api_base will use default which reads from $OPENAI_BASE_URL
        max_out_len=16384,
        batch_size=1,
        run_cfg=dict(num_gpus=0),
    )
]