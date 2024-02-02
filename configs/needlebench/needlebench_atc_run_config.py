from mmengine.config import read_base
with read_base():
    from ..datasets.needlebench.atc.atc_zh import needlebench_atc_datasets as needlebench_atc_zh_datasets
    from ..datasets.needlebench.atc.atc_en import needlebench_atc_datasets as needlebench_atc_en_datasets
    from .base import chat_models_4k, chat_models_8k, chat_models_32k, chat_models_200k
    # from ..summarizers.needlebench_4k import summarizer

from opencompass.runners import SlurmSequentialRunner
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

datasets = sum([v for k, v in locals().items() if (k.endswith("_datasets") or k == 'datasets')], [])

models = [*chat_models_4k, *chat_models_8k, *chat_models_32k, *chat_models_200k]
work_dir = './outputs/needlebench/atc'


infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=3000),
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