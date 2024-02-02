from opencompass.models import HuggingFaceCausalLM
from opencompass.models import HuggingFace
from opencompass.models.turbomind import TurboMindModel
from opencompass.models import HuggingFaceChatGLM3
from mmengine.config import read_base
with read_base():
    # original version
    from ..datasets.needlebench.original.cdme4k import cdme4k_datasets
    from ..datasets.needlebench.original.cdme4kto8k import cdme4kto8k_datasets
    from ..datasets.needlebench.original.cdme8k import cdme8k_datasets
    from ..datasets.needlebench.original.cdme8kto32k import cdme8kto32k_datasets
    from ..datasets.needlebench.original.cdme32k import cdme32k_datasets
    from ..datasets.needlebench.original.cdme200k import cdme200k_datasets
    from ..datasets.needlebench.original.cdme200k import cdme200k_trim_datasets

    # parallel version
    from ..datasets.needlebench.parallel_needle.cdme4k_parallel import cdme4k_parallel_datasets
    from ..datasets.needlebench.parallel_needle.cdme4kto8k_parallel import cdme4kto8k_parallel_datasets
    from ..datasets.needlebench.parallel_needle.cdme8k_parallel import cdme8k_parallel_datasets
    from ..datasets.needlebench.parallel_needle.cdme8kto32k_parallel import cdme8kto32k_parallel_datasets    
    from ..datasets.needlebench.parallel_needle.cdme32k_parallel import cdme32k_parallel_datasets
    from ..datasets.needlebench.parallel_needle.cdme200k_parallel import cdme200k_parallel_datasets

    # 2-needle version
    from ..datasets.needlebench.multi_needle.cdme4k_cot2_italy import cdme4k_cot2_italy_datasets
    from ..datasets.needlebench.multi_needle.cdme4kto8k_cot2_italy import cdme4kto8k_cot2_italy_datasets
    from ..datasets.needlebench.multi_needle.cdme8kto32k_cot2_italy import cdme8kto32k_cot2_italy_datasets

    # 3-needle version
    from ..datasets.needlebench.multi_needle.cdme4k_cot3_italy import cdme4k_cot3_italy_datasets
    from ..datasets.needlebench.multi_needle.cdme4kto8k_cot3_italy import cdme4kto8k_cot3_italy_datasets
    from ..datasets.needlebench.multi_needle.cdme8kto32k_cot3_italy import cdme8kto32k_cot3_italy_datasets

    # 3-needle-poem version
    from ..datasets.needlebench.multi_needle.cdme4k_cot3_poem import cdme4k_cot3_poem_datasets
    from ..datasets.needlebench.multi_needle.cdme4kto8k_cot3_poem import cdme4kto8k_cot3_poem_datasets
    from ..datasets.needlebench.multi_needle.cdme8kto32k_cot3_poem import cdme8kto32k_cot3_poem_datasets

datasets = [*cdme8kto32k_parallel_datasets]

qwen_7b_chat_hf_meta_template = dict(
    round=[
        dict(role="HUMAN", begin='\nuser\n', end=''),
        dict(role="BOT", begin="\nassistant\n", end='', generate=True),
    ],
)

qwen_7b_chat_turbomind_model = dict(
    type=TurboMindModel,
    abbr='qwen-7b-chat-turbomind',
    path="Qwen/Qwen-7B-Chat",
    meta_template=qwen_7b_chat_hf_meta_template,
    engine_config=dict(session_len=33000,
                       max_batch_size=1,
                       tp=1,
                       # rope_scaling_factor=3.0,
                       model_name="qwen-7b",),
    gen_config=dict(top_k=1, top_p=0.8,
                    temperature=1.0,
                    max_new_tokens=2000),
    max_out_len=2000,
    max_seq_len=33000,
    batch_size=1,
    concurrency=8,
    run_cfg=dict(num_gpus=1, num_procs=1)
)

qwen_72b_chat_hf_meta_template = dict(
    round=[
        dict(role="HUMAN", begin='\nuser\n', end=''),
        dict(role="BOT", begin="\nassistant\n", end='', generate=True),
    ],
)
qwen_72b_chat_hf_model = dict(
    type=HuggingFaceCausalLM,
    abbr='qwen-72b-chat-hf',
    path="Qwen/Qwen-72B-Chat",
    tokenizer_path='Qwen/Qwen-72B-Chat',
    model_kwargs=dict(
        device_map='auto',
        trust_remote_code=True
    ),
    tokenizer_kwargs=dict(
        padding_side='left',
        truncation_side='left',
        trust_remote_code=True,
        use_fast=False,
    ),
    pad_token_id=151643,
    max_out_len=2000,
    max_seq_len=33000,
    batch_size=1,
    meta_template=qwen_72b_chat_hf_meta_template,
    run_cfg=dict(num_gpus=4, num_procs=1),
    end_str='',
)
qwen_72b_chat_turbomind_model = dict(
    type=TurboMindModel,
    abbr='qwen-72b-chat-turbomind',
    path="Qwen/Qwen-72B-Chat",
    meta_template=qwen_72b_chat_hf_meta_template,
    engine_config=dict(session_len=33000,
                       max_batch_size=1,
                       tp=4,
                       # rope_scaling_factor=3.0,
                       cache_max_entry_count=280,
                       model_name="qwen-7b",),
    gen_config=dict(top_k=1, top_p=0.8,
                    temperature=1.0,
                    max_new_tokens=2000),
    max_out_len=2000,
    max_seq_len=33000,
    batch_size=1,
    concurrency=8,
    run_cfg=dict(num_gpus=4, num_procs=1)
)

models = [
    qwen_72b_chat_turbomind_model,
    # qwen_7b_chat_turbomind_model
]
