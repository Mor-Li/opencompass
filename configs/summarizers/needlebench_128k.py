context_lengths_128k = list([16000, 32000, 48000, 64000, 80000, 96000, 112000, 128000])
depths = [0, 5, 10, 15, 21, 26, 31, 36, 42, 47, 52, 57, 63, 68, 73, 78, 84, 89, 94, 100]

# Initialize the lists
_needlebench_128k_2needle_en = []
_needlebench_128k_3needle_en = []
_needlebench_128k_4needle_en = []
_needlebench_128k_5needle_en = []
_needlebench_128k_2needle_zh = []
_needlebench_128k_3needle_zh = []
_needlebench_128k_4needle_zh = []
_needlebench_128k_5needle_zh = []
_needlebench_128k_origin_en = []
_needlebench_128k_origin_zh = []

# Fill the lists using nested loops
for original_context_length in context_lengths_128k:
    for depth_percent in depths:
        _needlebench_128k_2needle_en.append(f'Length{original_context_length}Depth{int(depth_percent)}_2needle_en_128k')
        _needlebench_128k_3needle_en.append(f'Length{original_context_length}Depth{int(depth_percent)}_3needle_en_128k')
        _needlebench_128k_4needle_en.append(f'Length{original_context_length}Depth{int(depth_percent)}_4needle_en_128k')
        _needlebench_128k_5needle_en.append(f'Length{original_context_length}Depth{int(depth_percent)}_5needle_en_128k')
        _needlebench_128k_2needle_zh.append(f'Length{original_context_length}Depth{int(depth_percent)}_2needle_zh_128k')
        _needlebench_128k_3needle_zh.append(f'Length{original_context_length}Depth{int(depth_percent)}_3needle_zh_128k')
        _needlebench_128k_4needle_zh.append(f'Length{original_context_length}Depth{int(depth_percent)}_4needle_zh_128k')
        _needlebench_128k_5needle_zh.append(f'Length{original_context_length}Depth{int(depth_percent)}_5needle_zh_128k')

        _needlebench_128k_origin_en.append(f'Length{original_context_length}Depth{int(depth_percent)}_origin_en_128k')
        _needlebench_128k_origin_zh.append(f'Length{original_context_length}Depth{int(depth_percent)}_origin_zh_128k')

# Concatenate the multi-needle and origin lists
_needlebench_128k_multi_needle_en = _needlebench_128k_2needle_en + _needlebench_128k_3needle_en + _needlebench_128k_4needle_en + _needlebench_128k_5needle_en
_needlebench_128k_multi_needle_zh = _needlebench_128k_2needle_zh + _needlebench_128k_3needle_zh + _needlebench_128k_4needle_zh + _needlebench_128k_5needle_zh
_needlebench_128k_origin = _needlebench_128k_origin_en + _needlebench_128k_origin_zh
_needlebench_128k_multi_needle = _needlebench_128k_multi_needle_en + _needlebench_128k_multi_needle_zh

# Repeating the same process for parallel (assuming it's similar to origin_en)
_needlebench_128k_parallel_en = []
_needlebench_128k_parallel_zh = []
for original_context_length in context_lengths_128k:
    _needlebench_128k_parallel_en.append(f'Length{original_context_length}_parallel_en_128k')
for original_context_length in context_lengths_128k:
    _needlebench_128k_parallel_zh.append(f'Length{original_context_length}_parallel_zh_128k')
_needlebench_128k_parallel = _needlebench_128k_parallel_en + _needlebench_128k_parallel_zh

needlebench_summary_groups = [
    {'name': 'needlebench_128k_original_version', 'subsets': _needlebench_128k_origin},
    {'name': 'needlebench_128k_original_version_zh', 'subsets': _needlebench_128k_origin_zh},
    {'name': 'needlebench_128k_original_version_en', 'subsets': _needlebench_128k_origin_en},

    {'name': 'needlebench_128k_multi_needle_en', 'subsets': _needlebench_128k_multi_needle_en},
    {'name': 'needlebench_128k_multi_needle2_en', 'subsets': _needlebench_128k_2needle_en},
    {'name': 'needlebench_128k_multi_needle3_en', 'subsets': _needlebench_128k_3needle_en},
    {'name': 'needlebench_128k_multi_needle4_en', 'subsets': _needlebench_128k_4needle_en},
    {'name': 'needlebench_128k_multi_needle5_en', 'subsets': _needlebench_128k_5needle_en},

    {'name': 'needlebench_128k_multi_needle_zh', 'subsets': _needlebench_128k_multi_needle_zh},
    {'name': 'needlebench_128k_multi_needle2_zh', 'subsets': _needlebench_128k_2needle_zh},
    {'name': 'needlebench_128k_multi_needle3_zh', 'subsets': _needlebench_128k_3needle_zh},
    {'name': 'needlebench_128k_multi_needle4_zh', 'subsets': _needlebench_128k_4needle_zh},
    {'name': 'needlebench_128k_multi_needle5_zh', 'subsets': _needlebench_128k_5needle_zh},

    {'name': 'needlebench_128k_multi_needle', 'subsets': _needlebench_128k_multi_needle},

    {'name': 'needlebench_128k_parallel_version', 'subsets': _needlebench_128k_parallel},
    {'name': 'needlebench_128k_parallel_version_zh', 'subsets': _needlebench_128k_parallel_zh},
    {'name': 'needlebench_128k_parallel_version_en', 'subsets': _needlebench_128k_parallel_en},


    {'name': 'needlebench_128k', 'subsets': _needlebench_128k_origin + _needlebench_128k_multi_needle_en + _needlebench_128k_parallel},

]
summarizer = dict(
    dataset_abbrs=[
        'needlebench_128k',
        'needlebench_128k_parallel_version',
        'needlebench_128k_parallel_version_zh',
        'needlebench_128k_parallel_version_en',
        'needlebench_128k_multi_needle',
        'needlebench_128k_multi_needle_en',
        'needlebench_128k_multi_needle_zh',
        'needlebench_128k_multi_needle2_en',
        'needlebench_128k_multi_needle3_en',
        'needlebench_128k_multi_needle4_en',
        'needlebench_128k_multi_needle5_en',
        'needlebench_128k_multi_needle2_zh',
        'needlebench_128k_multi_needle3_zh',
        'needlebench_128k_multi_needle4_zh',

        *_needlebench_128k_origin, *_needlebench_128k_multi_needle, *_needlebench_128k_parallel,
    ],
    summary_groups=needlebench_summary_groups,
)
