from mmengine.config import read_base

from opencompass.models import OpenAI
from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

with read_base():
    # Import the PI-LLM dataset configuration
    from opencompass.configs.datasets.pi_llm.pi_llm_simple import pi_llm_datasets

# Define the summarizer for PI-LLM evaluation
summarizer = dict(
    dataset_abbrs=['pi_llm_simple'],
    summary_groups=[
        {
            'name': 'pi_llm',
            'subsets': ['pi_llm_simple'],
        }
    ],
)

# API template for GPT-4.1
api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
])

# Model configuration for GPT-4.1
models = [
    dict(
        abbr='gpt-4-0125-preview',  # GPT-4.1 model name
        type=OpenAI,
        path='gpt-4-0125-preview',
        key='ENV',  # The key will be obtained from $OPENAI_API_KEY
        meta_template=api_meta_template,
        query_per_second=1,
        max_out_len=2048,
        max_seq_len=8192,  # GPT-4.1 supports longer context
        batch_size=1,  # PI-LLM requires sequential processing
        temperature=0,  # Use deterministic output for evaluation
    ),
]

# Use the imported pi_llm_datasets
datasets = pi_llm_datasets

# Inference configuration
infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=1,  # Run sequentially for API calls
        task=dict(type=OpenICLInferTask)
    ),
)