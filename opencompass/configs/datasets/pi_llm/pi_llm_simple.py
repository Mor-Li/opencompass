from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import PILLMDataset, pi_llm_postprocess, PILLMEvaluator

# Basic configuration for PI-LLM dataset - simple test version
pi_llm_reader_cfg = dict(
    input_columns=['instruction'],  # Only instruction column now
    output_column='output'
)

# Inference configuration
pi_llm_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='{instruction}'),  # Only instruction, no input
            ],
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512)
)

# Evaluation configuration
pi_llm_eval_cfg = dict(
    evaluator=dict(type=PILLMEvaluator),
    pred_postprocessor=dict(type=pi_llm_postprocess)
)

# Simple test with fewer configurations for quick testing
pi_llm_datasets = [
    dict(
        abbr='pi_llm_simple',
        type=PILLMDataset,
        path='data/pi_llm/dict_category_double-word_46-400_v1-1.json',
        source_dict_path='data/pi_llm/dict_category_double-word_46-400_v1-1.json',
        n_tracked_keys=[10],  # Just 10 keys
        n_tracked_updates=[5, 10, 20],  # Just 3 update configurations
        n_untracked_keys=0,
        n_untracked_updates=0,
        random_update=1,
        prompt_updating='colon',
        prompt_forgetting='none',
        n_samples_per_config=5,  # Fewer samples per config
        reader_cfg=pi_llm_reader_cfg,
        infer_cfg=pi_llm_infer_cfg,
        eval_cfg=pi_llm_eval_cfg
    )
]