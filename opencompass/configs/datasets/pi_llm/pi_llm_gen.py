from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import PILLMDataset, pi_llm_postprocess, PILLMEvaluator

# Basic configuration for PI-LLM dataset
pi_llm_reader_cfg = dict(
    input_columns=['instruction', 'input'],
    output_column='output'
)

# Inference configuration
pi_llm_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='{instruction}\n{input}'),
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

# Test 1: Varying number of updates
pi_llm_test1_datasets = [
    dict(
        abbr='pi_llm_test1',
        type=PILLMDataset,
        path='data/pi_llm/dict_category_double-word_46-400_v1-1.json',
        source_dict_path='data/pi_llm/dict_category_double-word_46-400_v1-1.json',
        n_tracked_keys=[46],
        n_tracked_updates=[2, 3, 4, 6, 8, 12, 17, 24, 34, 48, 68, 97, 139, 197, 281, 400],
        n_untracked_keys=0,
        n_untracked_updates=0,
        random_update=1,
        prompt_updating='colon',
        prompt_forgetting='none',
        n_samples_per_config=10,
        reader_cfg=pi_llm_reader_cfg,
        infer_cfg=pi_llm_infer_cfg,
        eval_cfg=pi_llm_eval_cfg
    )
]

# Test 2: Varying number of keys
pi_llm_test2_datasets = [
    dict(
        abbr='pi_llm_test2',
        type=PILLMDataset,
        path='data/pi_llm/dict_category_double-word_46-400_v1-1.json',
        source_dict_path='data/pi_llm/dict_category_double-word_46-400_v1-1.json',
        n_tracked_keys=[2, 5, 10, 15, 20, 25, 30, 35, 40, 46],
        n_tracked_updates=[46],
        n_untracked_keys=0,
        n_untracked_updates=0,
        random_update=1,
        prompt_updating='colon',
        prompt_forgetting='none',
        n_samples_per_config=10,
        reader_cfg=pi_llm_reader_cfg,
        infer_cfg=pi_llm_infer_cfg,
        eval_cfg=pi_llm_eval_cfg
    )
]

# Test 3: Mix of tracked and untracked keys
pi_llm_test3_datasets = [
    dict(
        abbr='pi_llm_test3',
        type=PILLMDataset,
        path='data/pi_llm/dict_category_double-word_46-400_v1-1.json',
        source_dict_path='data/pi_llm/dict_category_double-word_46-400_v1-1.json',
        n_tracked_keys=[23],
        n_tracked_updates=[46],
        n_untracked_keys=23,
        n_untracked_updates=46,
        random_update=1,
        prompt_updating='colon',
        prompt_forgetting='none',
        n_samples_per_config=10,
        reader_cfg=pi_llm_reader_cfg,
        infer_cfg=pi_llm_infer_cfg,
        eval_cfg=pi_llm_eval_cfg
    )
]

# Combined dataset list (for running all tests)
pi_llm_datasets = pi_llm_test1_datasets + pi_llm_test2_datasets + pi_llm_test3_datasets