from opencompass.models import VLLM

settings = [
    ('llama-3-8b-vllm', '/fs-computility/llm/shared/llmeval/models/opencompass_hf_hub/models--meta-llama--Meta-Llama-3-8B/snapshots/62bd457b6fe961a42a631306577e622c83876cb6', 1),
]

models = []
for abbr, path, num_gpus in settings:
    models.append(
        dict(
            type=VLLM,
            abbr=abbr,
            path=path,
            model_kwargs=dict(tensor_parallel_size=num_gpus),
            max_out_len=100,
            max_seq_len=2048,
            batch_size=32,
            generation_kwargs=dict(temperature=0),
            run_cfg=dict(num_gpus=num_gpus, num_procs=1),
        )
    )
