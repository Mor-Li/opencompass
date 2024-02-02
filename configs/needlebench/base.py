from opencompass.models import HuggingFaceCausalLM
from opencompass.models import HuggingFace
from opencompass.models.turbomind import TurboMindModel
from opencompass.models import HuggingFaceChatGLM3
from opencompass.runners import SlurmSequentialRunner
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import VLLM

datasets = []

internlm2_chat_internal_meta_template = dict(
    begin="""""",
    round=[
        dict(role='HUMAN',
             begin='[UNUSED_TOKEN_146]user\n',
             end='[UNUSED_TOKEN_145]\n'),
        dict(role='SYSTEM',
             begin='[UNUSED_TOKEN_146]system\n',
             end='[UNUSED_TOKEN_145]\n'),
        dict(role='BOT',
             begin='[UNUSED_TOKEN_146]assistant\n',
             end='[UNUSED_TOKEN_145]\n',
             generate=True),
    ],
    eos_token_id=92542)


internlm2_7b_no_needle_data_turbomind = dict(
        type=TurboMindModel,
        abbr='internlm2-chat-7b-no-needle-data-turbomind',
        path="/mnt/petrelfs/share_data/zhangwenwei/models/llmit/exps/20231224/ampere_7B_3_0_0_FT_0_18rc32_32k_hf",
        meta_template=internlm2_chat_internal_meta_template,
        engine_config=dict(session_len=210000,
                           max_batch_size=8,
                           rope_scaling_factor=2.0,
                           model_name="internlm2-chat-7b"),
        gen_config=dict(top_k=1, top_p=0.8,
                        temperature=1.0,
                        max_new_tokens=2000,),
        max_out_len=2000,
        max_seq_len=210000,
        batch_size=8,
        concurrency=8,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )

internlm2_7b_add_needle_data_turbomind = dict(
        type=TurboMindModel,
        abbr='internlm2-chat-7b-add-needle-data-turbomind',
        path="/mnt/petrelfs/share_data/zhangwenwei/models/llmit/exps/20231224/ampere_7B_3_0_0_FT_0_18rc33_32k_hf",
        meta_template=internlm2_chat_internal_meta_template,
        engine_config=dict(session_len=210000,
                           max_batch_size=8,
                           rope_scaling_factor=2.0,
                           model_name="internlm2-chat-7b"),
        gen_config=dict(top_k=1, top_p=0.8,
                        temperature=1.0,
                        max_new_tokens=2000,),
        max_out_len=2000,
        max_seq_len=210000,
        batch_size=8,
        concurrency=8,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )

internlm2_hf_7b_base = dict(
        type=HuggingFaceCausalLM,
        abbr='internlm2-7b-hf-base',
        path="internlm/internlm2-7b",
        tokenizer_path='internlm/internlm2-7b',
        model_kwargs=dict(
            trust_remote_code=True,
            device_map='auto',
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            use_fast=False,
            trust_remote_code=True,
        ),
        max_out_len=2000,
        min_out_len=3,
        max_seq_len=32768,
        batch_size=8,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )

chatglm3_6b_hf_pi_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ]
)
chatglm3_6b_hf_cn = dict(
        type=HuggingFaceChatGLM3,
        abbr='chatglm3-6b-hf',
        path='THUDM/chatglm3-6b',
        tokenizer_path='THUDM/chatglm3-6b',
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True,
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
        ),
        meta_template=chatglm3_6b_hf_pi_meta_template,
        max_out_len=2000,
        max_seq_len=32768,
        batch_size=32,
        run_cfg=dict(num_gpus=1, num_procs=1)
    )

chatglm3_6b_hf_en = dict(
        type=HuggingFaceChatGLM3,
        abbr='chatglm3-6b-hf',
        path='THUDM/chatglm3-6b',
        tokenizer_path='THUDM/chatglm3-6b',
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True,
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
        ),
        meta_template=chatglm3_6b_hf_pi_meta_template,
        max_out_len=2000,
        max_seq_len=32768,
        batch_size=1,
        run_cfg=dict(num_gpus=2, num_procs=1)
    )

chatglm3_6b_32k_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ]
)
chatglm3_6b_32k_model = dict(
        type=HuggingFaceChatGLM3,
        abbr='chatglm3-6b-32k-hf',
        path='THUDM/chatglm3-6b-32k',
        tokenizer_path='THUDM/chatglm3-6b-32k',
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True,
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
        ),
        meta_template=chatglm3_6b_32k_meta_template,
        max_out_len=2000,
        max_seq_len=32768,
        batch_size=1,
        run_cfg=dict(num_gpus=2, num_procs=1)
    )

chatglm3_6b_32k_vllm = dict(
        type=VLLM,
        abbr='chatglm3-6b-32k-vllm',
        path='THUDM/chatglm3-6b-32k',
        max_out_len=2000,
        max_seq_len=32768,
        batch_size=4,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )


hf_internlm2_chat_7b_model_meta_template = dict(
    round=[
        dict(role='HUMAN',
             begin='<|im_start|>user\n', end='<|im_end|>\n'),
        dict(role='BOT', begin='<|im_start|>assistant\n',
             end='<|im_end|>\n', generate=True),
    ],
)
hf_internlm2_chat_7b_model = dict(
        type=HuggingFaceCausalLM,
        abbr='internlm2-chat-7b-hf',
        path="internlm/internlm2-chat-7b",
        tokenizer_path='internlm/internlm2-chat-7b',
        model_kwargs=dict(
            trust_remote_code=True,
            device_map='auto',
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            use_fast=False,
            trust_remote_code=True,
        ),
        max_out_len=2000,
        max_seq_len=32768,
        batch_size=8,
        meta_template=hf_internlm2_chat_7b_model_meta_template,
        run_cfg=dict(num_gpus=1, num_procs=1),
        end_str='<|im_end|>',
        )

hf_internlm2_chat_20b_model_meta_template = dict(
    round=[
        dict(role='HUMAN',
             begin='<|im_start|>user\n', end='<|im_end|>\n'),
        dict(role='BOT', begin='<|im_start|>assistant\n',
             end='<|im_end|>\n', generate=True),
    ],
)
hf_internlm2_chat_20b_model = dict(
        type=HuggingFaceCausalLM,
        abbr='internlm2-chat-20b-hf',
        path="internlm/internlm2-chat-20b",
        tokenizer_path='internlm/internlm2-chat-20b',
        model_kwargs=dict(
            trust_remote_code=True,
            device_map='auto',
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            use_fast=False,
            trust_remote_code=True,
        ),
        max_out_len=2000,
        max_seq_len=32768,
        batch_size=8,
        meta_template=hf_internlm2_chat_20b_model_meta_template,
        run_cfg=dict(num_gpus=2, num_procs=1),
        end_str='<|im_end|>',
    )

hf_internlm_chat_7b_model_meta_template = dict(
    round=[
        dict(role='HUMAN', begin='<|User|>:', end='\n'),
        dict(role='BOT', begin='<|Bot|>:', end='<eoa>\n', generate=True),
    ],
)
hf_internlm_chat_7b_model = dict(
        type=HuggingFaceCausalLM,
        abbr='internlm-chat-7b-hf',
        path="internlm/internlm-chat-7b",
        tokenizer_path='internlm/internlm-chat-7b',
        model_kwargs=dict(
            trust_remote_code=True,
            device_map='auto',
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            use_fast=False,
            trust_remote_code=True,
        ),
        max_out_len=2000,
        max_seq_len=32768,
        batch_size=8,
        meta_template=hf_internlm_chat_7b_model_meta_template,
        run_cfg=dict(num_gpus=1, num_procs=1),
        end_str='<eoa>',
    )

yi_6b_chat_hf_meta_template = dict(
    round=[
        dict(role="HUMAN", begin='<|im_start|>user\n', end='<|im_end|>\n'),
        dict(role="BOT", begin="<|im_start|>assistant\n", end='<|im_end|>\n', generate=True),
    ],
)
yi_6b_chat_hf_model = dict(
        type=HuggingFace,
        abbr='yi-6b-chat-hf',
        path='01-ai/Yi-6B-Chat',
        tokenizer_path='01-ai/Yi-6B-Chat',
        model_kwargs=dict(
            trust_remote_code=True,
            device_map='auto',
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
        ),
        meta_template=yi_6b_chat_hf_meta_template,
        max_out_len=2000,
        max_seq_len=8192,
        batch_size=8,
        run_cfg=dict(num_gpus=1, num_procs=1),
        end_str='<|im_end|>',
    )

yi_34b_chat_hf_meta_template = dict(
    round=[
        dict(role="HUMAN", begin='<|im_start|>user\n', end='<|im_end|>\n'),
        dict(role="BOT", begin="<|im_start|>assistant\n", end='<|im_end|>\n', generate=True),
    ],
)
yi_34b_chat_hf_model = dict(
        type=HuggingFace,
        abbr='yi-34b-chat-hf',
        path='01-ai/Yi-34B-Chat',
        tokenizer_path='01-ai/Yi-34B-Chat',
        model_kwargs=dict(
            trust_remote_code=True,
            device_map='auto',
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
        ),
        meta_template=yi_34b_chat_hf_meta_template,
        max_out_len=2000,
        max_seq_len=8192,
        batch_size=8,
        run_cfg=dict(num_gpus=2, num_procs=1),
        end_str='<|im_end|>',
    )

yi_6b_hf_base_model = dict(
    type=HuggingFace,
    abbr='yi-6b-hf',
    path='01-ai/Yi-6B',
    tokenizer_path='01-ai/Yi-6B',
    model_kwargs=dict(
        trust_remote_code=True,
        device_map='auto',
    ),
    tokenizer_kwargs=dict(
        padding_side='left',
        truncation_side='left',
        trust_remote_code=True,
    ),
    max_out_len=2000,
    max_seq_len=8192,
    batch_size=8,
    run_cfg=dict(num_gpus=1, num_procs=1),
)
yi_34b_hf_base_model = dict(
    type=HuggingFace,
    abbr='yi-34b-hf',
    path='01-ai/Yi-34B',
    tokenizer_path='01-ai/Yi-34B',
    model_kwargs=dict(
        trust_remote_code=True,
        device_map='auto',
    ),
    tokenizer_kwargs=dict(
        padding_side='left',
        truncation_side='left',
        trust_remote_code=True,
    ),
    max_out_len=2000,
    max_seq_len=8192,
    batch_size=8,
    run_cfg=dict(num_gpus=2, num_procs=1),
)

baichuan2_7b_chat_meta_template = dict(
    round=[
        dict(role='HUMAN', begin='<reserved_106>'),
        dict(role='BOT', begin='<reserved_107>', generate=True),
    ],
)
baichuan2_7b_chat_model = dict(
        type=HuggingFaceCausalLM,
        abbr='baichuan2-7b-chat-hf',
        path="baichuan-inc/Baichuan2-7B-Chat",
        tokenizer_path='baichuan-inc/Baichuan2-7B-Chat',
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
            use_fast=False,
        ),
        meta_template=baichuan2_7b_chat_meta_template,
        max_out_len=2000,
        max_seq_len=8192,
        batch_size=8,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
baichuan2_7b_base_model = dict(
        type=HuggingFaceCausalLM,
        abbr='baichuan2-7b-base-hf',
        path="baichuan-inc/Baichuan2-7B-Base",
        tokenizer_path='baichuan-inc/Baichuan2-7B-Base',
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
            use_fast=False,
        ),
        max_out_len=2000,
        max_seq_len=8192,
        batch_size=8,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )

baichuan2_13b_chat_meta_template = dict(
    round=[
        dict(role='HUMAN', begin='<reserved_106>'),
        dict(role='BOT', begin='<reserved_107>', generate=True),
    ],
)
baichuan2_13b_chat_model = dict(
        type=HuggingFaceCausalLM,
        abbr='baichuan2-13b-chat-hf',
        path="baichuan-inc/Baichuan2-13B-Chat",
        tokenizer_path='baichuan-inc/Baichuan2-13B-Chat',
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
            use_fast=False,
        ),
        meta_template=baichuan2_13b_chat_meta_template,
        max_out_len=2000,
        max_seq_len=8192,
        batch_size=8,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=2, num_procs=1),
    )
baichuan2_13b_base_model = dict(
        type=HuggingFaceCausalLM,
        abbr='baichuan2-13b-base-hf',
        path="baichuan-inc/Baichuan2-13B-Base",
        tokenizer_path='baichuan-inc/Baichuan2-13B-Base',
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
            use_fast=False,
        ),
        max_out_len=2000,
        max_seq_len=8192,
        batch_size=8,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=2, num_procs=1),
    )

llama_2_7b_chat_hf_meta_template = dict(
    round=[
        dict(role="HUMAN", begin=' [INST] ', end=' [/INST] '),
        dict(role="BOT", begin='', end='', generate=True),
    ],
)
llama_2_7b_chat_hf_model = dict(
        type=HuggingFaceCausalLM,
        abbr='llama-2-7b-chat-hf',
        path="meta-llama/Llama-2-7b-chat-hf",
        tokenizer_path='meta-llama/Llama-2-7b-chat-hf',
        model_kwargs=dict(
            device_map='auto'
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            use_fast=False,
        ),
        meta_template=llama_2_7b_chat_hf_meta_template,
        max_out_len=2000,
        max_seq_len=8192,
        batch_size=8,
        run_cfg=dict(num_gpus=1, num_procs=1),
        end_str='[INST]',
    )

llama_2_13b_chat_hf_meta_template = dict(
    round=[
        dict(role="HUMAN", begin=' [INST] ', end=' [/INST] '),
        dict(role="BOT", begin='', end='', generate=True),
    ],
)
llama_2_13b_chat_hf_model = dict(
    type=HuggingFaceCausalLM,
    abbr='llama-2-13b-chat-hf',
    path="meta-llama/Llama-2-13b-chat-hf",
    tokenizer_path='meta-llama/Llama-2-13b-chat-hf',
    model_kwargs=dict(
        device_map='auto'
    ),
    tokenizer_kwargs=dict(
        padding_side='left',
        truncation_side='left',
        use_fast=False,
    ),
    meta_template=llama_2_13b_chat_hf_meta_template,
    max_out_len=2000,
    max_seq_len=8192,
    batch_size=8,
    run_cfg=dict(num_gpus=2, num_procs=1),
    end_str='[INST]',
)

llama_2_70b_chat_hf_meta_template = dict(
    round=[
        dict(role="HUMAN", begin=' [INST] ', end=' [/INST] '),
        dict(role="BOT", begin='', end='', generate=True),
    ],
)
llama_2_70b_chat_hf_model = dict(
    type=HuggingFaceCausalLM,
    abbr='llama-2-70b-chat-hf',
    path="meta-llama/Llama-2-70b-chat-hf",
    tokenizer_path='meta-llama/Llama-2-70b-chat-hf',
    model_kwargs=dict(
        device_map='auto'
    ),
    tokenizer_kwargs=dict(
        padding_side='left',
        truncation_side='left',
        use_fast=False,
    ),
    meta_template=llama_2_70b_chat_hf_meta_template,
    max_out_len=2000,
    max_seq_len=8192,
    batch_size=8,
    run_cfg=dict(num_gpus=2, num_procs=1),
    end_str='[INST]',
)

qwen_7b_chat_hf_meta_template = dict(
    round=[
        dict(role="HUMAN", begin='\n<|im_start|>user\n', end='<|im_end|>'),
        dict(role="BOT", begin="\n<|im_start|>assistant\n", end='<|im_end|>', generate=True),
    ],
)
qwen_7b_chat_hf_model = dict(
    type=HuggingFaceCausalLM,
    abbr='qwen-7b-chat-hf',
    path="Qwen/Qwen-7B-Chat",
    tokenizer_path='Qwen/Qwen-7B-Chat',
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
    max_seq_len=32768,
    batch_size=1,
    meta_template=qwen_7b_chat_hf_meta_template,
    run_cfg=dict(num_gpus=2, num_procs=1),
    end_str='',
)
qwen_7b_chat_hf_model_4gpu = dict(
    type=HuggingFaceCausalLM,
    abbr='qwen-7b-chat-hf',
    path="Qwen/Qwen-7B-Chat",
    tokenizer_path='Qwen/Qwen-7B-Chat',
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
    max_seq_len=32768,
    batch_size=1,
    meta_template=qwen_7b_chat_hf_meta_template,
    run_cfg=dict(num_gpus=4, num_procs=1),
    end_str='',
)
qwen_14b_chat_hf_meta_template = dict(
    round=[
        dict(role="HUMAN", begin='\n<|im_start|>user\n', end='<|im_end|>'),
        dict(role="BOT", begin="\n<|im_start|>assistant\n", end='<|im_end|>', generate=True),
    ],
)
qwen_14b_chat_hf_model = dict(
    type=HuggingFaceCausalLM,
    abbr='qwen-14b-chat-hf',
    path="Qwen/Qwen-14B-Chat",
    tokenizer_path='Qwen/Qwen-14B-Chat',
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
    max_seq_len=8192,
    batch_size=8,
    meta_template=qwen_14b_chat_hf_meta_template,
    run_cfg=dict(num_gpus=1, num_procs=1),
    end_str='',
)

qwen_72b_chat_hf_meta_template = dict(
    round=[
        dict(role="HUMAN", begin='\n<|im_start|>user\n', end='<|im_end|>'),
        dict(role="BOT", begin="\n<|im_start|>assistant\n", end='<|im_end|>', generate=True),
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
    max_seq_len=32768,
    batch_size=1,
    meta_template=qwen_72b_chat_hf_meta_template,
    run_cfg=dict(num_gpus=4, num_procs=1),
    end_str='',
)

qwen_72b_meta_template = dict(
    round=[
        dict(role="HUMAN", begin='\n<|im_start|>user\n', end='<|im_end|>'),
        dict(role="BOT", begin="\n<|im_start|>assistant\n", end='<|im_end|>', generate=True),
    ],
)

qwen_7b_chat_vllm = dict(
        type=VLLM,
        abbr='qwen-7b-chat-vllm',
        path="Qwen/Qwen-7B-Chat",
        model_kwargs=dict(tensor_parallel_size=2),
        meta_template=qwen_72b_meta_template,
        max_out_len=2000,
        max_seq_len=32768,
        batch_size=32,
        generation_kwargs=dict(temperature=0),
        end_str='<|im_end|>',
        run_cfg=dict(num_gpus=2, num_procs=1),
    )

qwen_72b_chat_vllm = dict(
        type=VLLM,
        abbr='qwen-72b-chat-vllm',
        path="Qwen/Qwen-72B-Chat",
        model_kwargs=dict(tensor_parallel_size=4),
        meta_template=qwen_72b_meta_template,
        max_out_len=2000,
        max_seq_len=32768,
        batch_size=32,
        generation_kwargs=dict(temperature=0),
        end_str='<|im_end|>',
        run_cfg=dict(num_gpus=4, num_procs=1),
    )


chatglm3_6b_32k_vllm = dict(
        type=VLLM,
        abbr='chatglm3-6b-32k-vllm',
        path='THUDM/chatglm3-6b-32k',
        max_out_len=2000,
        max_seq_len=32768,
        batch_size=32,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )


# sus_32b_chat_hf_meta_template = dict(
#     round=[
#         dict(role="HUMAN", begin='\nuser\n', end=''),
#         dict(role="BOT", begin="\nassistant\n", end='', generate=True),
#     ],
# )
# sus_32b_chat_hf_model = dict(
#     type=HuggingFaceCausalLM,
#     abbr='qwen-14b-chat-hf',
#     path="Qwen/Qwen-14B-Chat",
#     tokenizer_path='Qwen/Qwen-14B-Chat',
#     model_kwargs=dict(
#         device_map='auto',
#         trust_remote_code=True
#     ),
#     tokenizer_kwargs=dict(
#         padding_side='left',
#         truncation_side='left',
#         trust_remote_code=True,
#         use_fast=False,
#     ),
#     pad_token_id=151643,
#     max_out_len=2000,
#     max_seq_len=8192,
#     batch_size=8,
#     meta_template=sus_32b_chat_hf_meta_template,
#     run_cfg=dict(num_gpus=1, num_procs=1),
#     end_str='',
# )

skywork_13b_hf_base_model = dict(
    type=HuggingFaceCausalLM,
    abbr='skywork-13b-hf',
    path="Skywork/Skywork-13B-base",
    tokenizer_path='Skywork/Skywork-13B-base',
    model_kwargs=dict(
        device_map='auto',
        trust_remote_code=True,
    ),
    tokenizer_kwargs=dict(
        padding_side='left',
        truncation_side='left',
        trust_remote_code=True,
        use_fast=False,
    ),
    max_out_len=2000,
    max_seq_len=8192,
    batch_size=8,
    run_cfg=dict(num_gpus=1, num_procs=1),
)



# Turbomind model
internlm2_7b_turobomind_chat_cn = dict(
        type=TurboMindModel,
        abbr='internlm2-chat-7b-turbomind',
        path="internlm/internlm2-chat-7b",
        meta_template=hf_internlm2_chat_7b_model_meta_template,
        engine_config=dict(session_len=210000,
                           max_batch_size=8,
                           rope_scaling_factor=2.0,
                           model_name="internlm2-chat-7b"),
        gen_config=dict(top_k=1, top_p=0.8,
                        temperature=1.0,
                        max_new_tokens=2000),
        max_out_len=2000,
        max_seq_len=210000,
        batch_size=8,
        concurrency=8,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )

internlm2_20b_turobomind_chat_cn = dict(
        type=TurboMindModel,
        abbr='internlm2-chat-20b-turbomind',
        path="internlm/internlm2-chat-20b",
        meta_template=hf_internlm2_chat_7b_model_meta_template,
        engine_config=dict(session_len=210000,
                           max_batch_size=8,
                           rope_scaling_factor=3.0,
                           model_name="internlm2-chat-20b",
                           tp=2),
        gen_config=dict(top_k=1, top_p=0.8,
                        temperature=1.0,
                        max_new_tokens=2000,),
        max_out_len=2000,
        max_seq_len=210000,
        batch_size=1,
        concurrency=8,
        run_cfg=dict(num_gpus=2, num_procs=1),
    )

internlm2_7b_turobomind_chat_en = dict(
        type=TurboMindModel,
        abbr='internlm2-chat-7b-turbomind',
        path="internlm/internlm2-chat-7b",
        meta_template=hf_internlm2_chat_7b_model_meta_template,
        engine_config=dict(session_len=210000,
                           max_batch_size=8,
                           rope_scaling_factor=2.0,
                           model_name="internlm2-chat-7b"),
        gen_config=dict(top_k=1, top_p=0.8,
                        temperature=1.0,
                        max_new_tokens=2000),
        max_out_len=2000,
        max_seq_len=210000,
        batch_size=8,
        concurrency=8,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )

internlm2_20b_turobomind_chat_en = dict(
        type=TurboMindModel,
        abbr='internlm2-chat-20b-turbomind',
        path="internlm/internlm2-chat-20b",
        meta_template=hf_internlm2_chat_7b_model_meta_template,
        engine_config=dict(session_len=210000,
                           max_batch_size=8,
                           rope_scaling_factor=3.0,
                           model_name="internlm2-chat-20b",
                           tp=2),
        gen_config=dict(top_k=1, top_p=0.8,
                        temperature=1.0,
                        max_new_tokens=2000,),
        max_out_len=2000,
        max_seq_len=210000,
        batch_size=1,
        concurrency=8,
        run_cfg=dict(num_gpus=2, num_procs=1),
    )


mixtral_8_7b_instruct_v01_meta_template = dict(
    begin="<s>",
    round=[
        dict(role="HUMAN", begin='[INST]', end='[/INST]'),
        dict(role="BOT", begin="", end='</s>', generate=True),
    ],
    eos_token_id=2
)


mixtral_8_7b_instruct_v0_1_vllm = dict(
        type=VLLM,
        abbr='mixtral-8x7b-instruct-v0.1-vllm',
        path='mistralai/Mixtral-8x7B-Instruct-v0.1',
        model_kwargs=dict(tensor_parallel_size=2),
        meta_template=mixtral_8_7b_instruct_v01_meta_template,
        max_out_len=2000,
        max_seq_len=32768,
        batch_size=32,
        generation_kwargs=dict(temperature=0),
        end_str='</s>',
        run_cfg=dict(num_gpus=2, num_procs=1),
    )


zephyr_7b_beta_meta_template = dict(
    round=[
        dict(role="HUMAN", begin='<|user|>\n', end='</s>'),
        dict(role="BOT", begin="<|assistant|>\n", end='</s>', generate=True),
    ],
)

zephyr_7b_chat = dict(
        type=VLLM,
        abbr='zephyr-7b-beta-vllm',
        path='HuggingFaceH4/zephyr-7b-beta',
        meta_template=zephyr_7b_beta_meta_template,
        max_out_len=2000,
        max_seq_len=32768,
        batch_size=32,
        generation_kwargs=dict(temperature=0),
        end_str='</s>',
        run_cfg=dict(num_gpus=1, num_procs=1),
    )


mistral_7b_instruct_v0_2_meta_template = dict(
    begin="<s>",
    round=[
        dict(role="HUMAN", begin='[INST]', end='[/INST]'),
        dict(role="BOT", begin="", end='</s>', generate=True),
    ],
    eos_token_id=2
)

mistral_7b_instruct_v0_2 = dict(
        type=VLLM,
        abbr='mistral-7b-instruct-v0.2-vllm',
        path='mistralai/Mistral-7B-Instruct-v0.2',
        meta_template=mistral_7b_instruct_v0_2_meta_template,
        max_out_len=2000,
        max_seq_len=32768,
        batch_size=32,
        generation_kwargs=dict(temperature=0),
        end_str='</s>',
        run_cfg=dict(num_gpus=1, num_procs=1),
    )



deepseek_67b_chat_meta_template = dict(
    round=[
        dict(role="HUMAN", begin='User: ', end='\n\n'),
        dict(role="BOT", begin="Assistant: ", end='<｜end▁of▁sentence｜>', generate=True),
    ],
)

deepseek_67b_chat = dict(
        type=HuggingFaceCausalLM,
        abbr='deepseek-67b-chat-hf',
        path="deepseek-ai/deepseek-llm-67b-chat",
        tokenizer_path='deepseek-ai/deepseek-llm-67b-chat',
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True,
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
            use_fast=False,
        ),
        meta_template=deepseek_67b_chat_meta_template,
        max_out_len=2000,
        max_seq_len=4096,
        batch_size=8,
        run_cfg=dict(num_gpus=4, num_procs=1),
        end_str='<｜end▁of▁sentence｜>',
    )

wizard_70b_vllm_meta_template = dict(
    round=[
        dict(role="HUMAN", begin='USER: ', end=' '),
        dict(role="BOT", begin="ASSISTANT: ", end='</s>', generate=True),
    ],
)

wizard_70b_vllm = dict(
        type=VLLM,
        abbr='wizardlm-70b-v1.0-vllm',
        path='WizardLM/WizardLM-70B-V1.0',
        model_kwargs=dict(tensor_parallel_size=4),
        meta_template=wizard_70b_vllm_meta_template,
        max_out_len=2000,
        max_seq_len=4096,
        batch_size=32,
        generation_kwargs=dict(temperature=0),
        end_str='</s>',
        run_cfg=dict(num_gpus=4, num_procs=1),
    )

orionstar_yi_34b_meta_template = dict(
    begin='<|startoftext|>',
    round=[
        dict(role="HUMAN", begin='Human: ', end='\n\n'),
        dict(role="BOT", begin="Assistant: <|endoftext|>", end='<|endoftext|>', generate=True),
    ],
    eos_token_id=2
)

orionstar_14b_chat_meta_template = dict(
    begin='<s>',
    round=[
        dict(role="HUMAN", begin='Human: ', end='\n'),
        dict(role="BOT", begin="Assistant: ", end='</s>', generate=True),
    ],
    eos_token_id=2
)

orionstar_yi_34b = dict(
        abbr='orionstar-yi-34b-chat-hf',
        type=HuggingFaceCausalLM,
        path='OrionStarAI/OrionStar-Yi-34B-Chat',
        tokenizer_path='OrionStarAI/OrionStar-Yi-34B-Chat',
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True,
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
        ),
        meta_template=orionstar_yi_34b_meta_template,
        max_out_len=2000,
        max_seq_len=4096,
        batch_size=8,
        run_cfg=dict(num_gpus=4, num_procs=1),
        end_str='<|endoftext|>',
    )

orionstar_14b_chat = dict(
        abbr='orionstar-14b-long-chat-hf',
        type=HuggingFaceCausalLM,
        path='OrionStarAI/Orion-14B-LongChat',
        tokenizer_path='OrionStarAI/Orion-14B-LongChat',
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True,
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
        ),
        meta_template=orionstar_14b_chat_meta_template,
        max_out_len=2000,
        max_seq_len=320000,
        batch_size=1,
        run_cfg=dict(num_gpus=4, num_procs=1),
        end_str='<|endoftext|>',
)


base_models_4k = [baichuan2_7b_base_model,
                  baichuan2_13b_base_model,
                  yi_6b_hf_base_model,
                  yi_34b_hf_base_model,
                  skywork_13b_hf_base_model,]

# chat_models_4k_en = [
#     llama_2_7b_chat_hf_model,
#     llama_2_13b_chat_hf_model,
#     llama_2_70b_chat_hf_model,
#     baichuan2_7b_chat_model,
#     baichuan2_13b_chat_model,
#     yi_6b_chat_hf_model,
#     yi_34b_chat_hf_model,]

# chat_models_4k_cn = [
#     baichuan2_7b_chat_model,
#     baichuan2_13b_chat_model,
#     yi_6b_chat_hf_model,
#     yi_34b_chat_hf_model,]

chat_models_4k = [
    llama_2_7b_chat_hf_model,
    llama_2_13b_chat_hf_model,
    llama_2_70b_chat_hf_model,
    baichuan2_7b_chat_model,
    baichuan2_13b_chat_model,
    yi_6b_chat_hf_model,
    yi_34b_chat_hf_model,
    deepseek_67b_chat,
    wizard_70b_vllm,
    orionstar_yi_34b,
    ]

# chat_models_8k_cn = [chatglm3_6b_hf_cn,
#                      hf_internlm2_chat_7b_model,
#                      hf_internlm2_chat_20b_model,
#                      hf_internlm_chat_7b_model,
#                      qwen_14b_chat_hf_model,]

# chat_models_8k_en = [chatglm3_6b_hf_en,
#                      hf_internlm2_chat_7b_model,
#                      hf_internlm2_chat_20b_model,
#                      hf_internlm_chat_7b_model,
#                      qwen_7b_chat_hf_model,
#                      qwen_72b_chat_hf_model,
#                      qwen_14b_chat_hf_model,]


chat_models_8k = [
    chatglm3_6b_hf_en,
    hf_internlm2_chat_7b_model,
    hf_internlm2_chat_20b_model,
    hf_internlm_chat_7b_model,
    qwen_14b_chat_hf_model,
    qwen_72b_chat_hf_model,

]
# chat_models_32k_cn = [ 
#                         internlm2_7b_turobomind_chat_cn,
#                         internlm2_20b_turobomind_chat_cn,
#                         qwen_7b_chat_hf_model,
#                         qwen_72b_chat_vllm,
#                         chatglm3_6b_32k_model,
#                         ]


# chat_models_32k_en = [
#                         internlm2_7b_turobomind_chat_en,
#                         internlm2_20b_turobomind_chat_en,
#                         qwen_7b_chat_vllm,# 这就是个8k模型 被vllm搞成8k 本来是32k虽然只能en跑到25k
#                         # qwen_7b_chat_hf_model,
#                         # qwen_72b_chat_hf_model,
#                         # qwen_72b_chat_hf_model,
#                         qwen_72b_chat_vllm,
#                         chatglm3_6b_32k_model,
#                         #chatglm3_6b_32k_vllm,这个没有必要也没有意义
#                      ]

chat_models_32k = [
    mixtral_8_7b_instruct_v0_1_vllm,
    internlm2_7b_turobomind_chat_cn,
    internlm2_20b_turobomind_chat_cn,
    qwen_7b_chat_hf_model,
    qwen_72b_chat_vllm,
    chatglm3_6b_32k_model,
    zephyr_7b_chat,
    mistral_7b_instruct_v0_2,

]

chat_models_200k_cn = [internlm2_7b_turobomind_chat_cn,
                       internlm2_20b_turobomind_chat_cn,]

chat_models_200k_en = [internlm2_7b_turobomind_chat_en,
                       internlm2_20b_turobomind_chat_en,]

chat_models_200k = [
    # 中文英文都一样的，没事不用在意en cn
    internlm2_7b_turobomind_chat_en,
    internlm2_20b_turobomind_chat_en,
    orionstar_14b_chat
]

chat_models_200k_compare_needle_data = [
    internlm2_7b_no_needle_data_turbomind,
    internlm2_7b_add_needle_data_turbomind,
]


infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=30000),
    runner=dict(
        type=SlurmSequentialRunner,
        partition='llm_dev2',
        quotatype='auto',
        max_num_workers=256,
        task=dict(type=OpenICLInferTask),
        retry=5
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner, n=2000),
    runner=dict(
        type=SlurmSequentialRunner,
        partition='llm_dev2',
        quotatype='auto',
        max_num_workers=256,
        task=dict(type=OpenICLEvalTask),
        retry=0
    ),
)

parallel_infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=400),
    runner=dict(
        type=SlurmSequentialRunner,
        partition='llm_dev2',
        quotatype='auto',
        max_num_workers=256,
        task=dict(type=OpenICLInferTask)
    ),
)