from mmengine.config import read_base

with read_base():
    from .multi.needlebench_multi_2needle_en import needlebench_datasets as needlebench_multi_2needle_en_datasets
    from .multi.needlebench_multi_3needle_en import needlebench_datasets as needlebench_multi_3needle_en_datasets
    from .multi.needlebench_multi_4needle_en import needlebench_datasets as needlebench_multi_4needle_en_datasets
    from .multi.needlebench_multi_5needle_en import needlebench_datasets as needlebench_multi_5needle_en_datasets
    from .multi.needlebench_multi_2needle_zh import needlebench_datasets as needlebench_multi_2needle_zh_datasets
    from .multi.needlebench_multi_3needle_zh import needlebench_datasets as needlebench_multi_3needle_zh_datasets
    from .multi.needlebench_multi_4needle_zh import needlebench_datasets as needlebench_multi_4needle_zh_datasets
    from .multi.needlebench_multi_5needle_zh import needlebench_datasets as needlebench_multi_5needle_zh_datasets

    from .original.needlebench_origin_en import needlebench_datasets as needlebench_origin_en_datasets
    from .original.needlebench_origin_zh import needlebench_datasets as needlebench_origin_zh_datasets
    from .parallel.needlebench_parallel_en import needlebench_datasets as needlebench_parallel_en_datasets
    from .parallel.needlebench_parallel_zh import needlebench_datasets as needlebench_parallel_zh_datasets

needlebench_datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
