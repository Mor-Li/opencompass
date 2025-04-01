from opencompass.models import VLLM

models = [
    dict(
        type=VLLM,
        abbr='llama-3.1-8b-instruct-vllm',
        path='meta-llama/Meta-Llama-3.1-8B-Instruct',
        model_kwargs=dict(tensor_parallel_size=2),
        max_out_len=4096,
        max_seq_len=140000,
        batch_size=1,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=2),
    )
]
