from opencompass.summarizers.needlebench import NeedleBenchSummarizer
context_lengths_8k = list(range(5000, 9000, 1000))
depths = [0, 5, 10, 15, 21, 26, 31, 36, 42, 47, 52, 57, 63, 68, 73, 78, 84, 89, 94, 100]

# Initialize the lists
_needlebench_8k_2needle_en = []
_needlebench_8k_3needle_en = []
_needlebench_8k_4needle_en = []
_needlebench_8k_5needle_en = []
_needlebench_8k_2needle_zh = []
_needlebench_8k_3needle_zh = []
_needlebench_8k_4needle_zh = []
_needlebench_8k_5needle_zh = []
_needlebench_8k_origin_en = []
_needlebench_8k_origin_zh = []

# Fill the lists using nested loops
for original_context_length in context_lengths_8k:
    for depth_percent in depths:
        _needlebench_8k_2needle_en.append(f'Length{original_context_length}Depth{int(depth_percent)}_2needle_en_8k')
        _needlebench_8k_3needle_en.append(f'Length{original_context_length}Depth{int(depth_percent)}_3needle_en_8k')
        _needlebench_8k_4needle_en.append(f'Length{original_context_length}Depth{int(depth_percent)}_4needle_en_8k')
        _needlebench_8k_5needle_en.append(f'Length{original_context_length}Depth{int(depth_percent)}_5needle_en_8k')
        _needlebench_8k_2needle_zh.append(f'Length{original_context_length}Depth{int(depth_percent)}_2needle_zh_8k')
        _needlebench_8k_3needle_zh.append(f'Length{original_context_length}Depth{int(depth_percent)}_3needle_zh_8k')
        _needlebench_8k_4needle_zh.append(f'Length{original_context_length}Depth{int(depth_percent)}_4needle_zh_8k')
        _needlebench_8k_5needle_zh.append(f'Length{original_context_length}Depth{int(depth_percent)}_5needle_zh_8k')

        _needlebench_8k_origin_en.append(f'Length{original_context_length}Depth{int(depth_percent)}_origin_en_8k')
        _needlebench_8k_origin_zh.append(f'Length{original_context_length}Depth{int(depth_percent)}_origin_zh_8k')

# Concatenate the multi-needle and origin lists
_needlebench_8k_multi_needle_en = _needlebench_8k_2needle_en + _needlebench_8k_3needle_en + _needlebench_8k_4needle_en + _needlebench_8k_5needle_en
_needlebench_8k_multi_needle_zh = _needlebench_8k_2needle_zh + _needlebench_8k_3needle_zh + _needlebench_8k_4needle_zh + _needlebench_8k_5needle_zh
_needlebench_8k_origin = _needlebench_8k_origin_en + _needlebench_8k_origin_zh
_needlebench_8k_multi_needle = _needlebench_8k_multi_needle_en + _needlebench_8k_multi_needle_zh

# Repeating the same process for parallel (assuming it's similar to origin_en)
_needlebench_8k_parallel_en = []
_needlebench_8k_parallel_zh = []
for original_context_length in context_lengths_8k:
    _needlebench_8k_parallel_en.append(f'Length{original_context_length}_parallel_en_8k')
for original_context_length in context_lengths_8k:
    _needlebench_8k_parallel_zh.append(f'Length{original_context_length}_parallel_zh_8k')
_needlebench_8k_parallel = _needlebench_8k_parallel_en + _needlebench_8k_parallel_zh

needlebench_summary_groups = [
    {'name': 'original_version', 'subsets': _needlebench_8k_origin},
    {'name': 'original_version_zh', 'subsets': _needlebench_8k_origin_zh},
    {'name': 'original_version_en', 'subsets': _needlebench_8k_origin_en},

    {'name': 'multi_needle_en', 'subsets': _needlebench_8k_multi_needle_en},
    {'name': 'multi_needle2_en', 'subsets': _needlebench_8k_2needle_en},
    {'name': 'multi_needle3_en', 'subsets': _needlebench_8k_3needle_en},
    {'name': 'multi_needle4_en', 'subsets': _needlebench_8k_4needle_en},
    {'name': 'multi_needle5_en', 'subsets': _needlebench_8k_5needle_en},

    {'name': 'multi_needle_zh', 'subsets': _needlebench_8k_multi_needle_zh},
    {'name': 'multi_needle2_zh', 'subsets': _needlebench_8k_2needle_zh},
    {'name': 'multi_needle3_zh', 'subsets': _needlebench_8k_3needle_zh},
    {'name': 'multi_needle4_zh', 'subsets': _needlebench_8k_4needle_zh},
    {'name': 'multi_needle5_zh', 'subsets': _needlebench_8k_5needle_zh},

    {'name': 'multi_needle', 'subsets': _needlebench_8k_multi_needle},

    {'name': 'parallel_version', 'subsets': _needlebench_8k_parallel},
    {'name': 'parallel_version_zh', 'subsets': _needlebench_8k_parallel_zh},
    {'name': 'parallel_version_en', 'subsets': _needlebench_8k_parallel_en},


    {'name': 'overall',
     'subsets': [['original_version', 'naive_average'],
                 ['multi_needle', 'naive_average'],
                 ['parallel_version', 'average_score']],
     'weights': {'original_version': 0.4,
                 'multi_needle': 0.3,
                 'parallel_version': 0.3}},
]
summarizer = dict(
    type=NeedleBenchSummarizer,
    dataset_abbrs=[
        'overall',
        '--------- NeedleBench-8k Single-Needle ---------',  # category
        'original_version',
        'original_version_zh',
        'original_version_en',
        '--------- NeedleBench-8k Parallel-Needles ---------',  # category
        'parallel_version',
        'parallel_version_zh',
        'parallel_version_en',
        '--------- NeedleBench-8k Multi-Needles ---------',  # category
        'multi_needle',
        'multi_needle_en',
        'multi_needle_zh',
        'multi_needle2_en',
        'multi_needle3_en',
        'multi_needle4_en',
        'multi_needle5_en',
        'multi_needle2_zh',
        'multi_needle3_zh',
        'multi_needle4_zh',
        'multi_needle5_zh',

        # *_needlebench_8k_origin, *_needlebench_8k_multi_needle, *_needlebench_8k_parallel,
    ],
    summary_groups=needlebench_summary_groups,
)
