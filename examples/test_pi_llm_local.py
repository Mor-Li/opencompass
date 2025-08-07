from mmengine.config import read_base

from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

with read_base():
    # Import the simple PI-LLM dataset configuration
    from opencompass.configs.datasets.pi_llm.pi_llm_simple import pi_llm_datasets

# Use a dummy model for testing
from opencompass.models import HuggingFace

# Simple test model configuration
models = [
    dict(
        abbr='dummy-model',
        type=HuggingFace,
        path='facebook/opt-125m',  # Small model for testing
        tokenizer_path='facebook/opt-125m',
        model_kwargs=dict(device_map='auto'),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
        ),
        max_out_len=512,
        max_seq_len=2048,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    )
]

# Override datasets to use absolute path
datasets = pi_llm_datasets
for dataset in datasets:
    dataset['source_dict_path'] = '/mnt/moonfs/limo-m2/PI-LLM-Opencompass/opencompass/data/pi_llm/dict_category_double-word_46-400_v1-1.json'
    dataset['path'] = dataset['source_dict_path']

# Inference configuration
infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=1,
        task=dict(type=OpenICLInferTask)
    ),
)

# Simple summarizer
summarizer = dict(
    dataset_abbrs=['pi_llm_simple'],
    summary_groups=[
        {
            'name': 'pi_llm',
            'subsets': ['pi_llm_simple'],
        }
    ],
)