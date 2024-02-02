context_lengths_8k = list(range(5000, 9000, 1000))
depths = [0, 5, 10, 15, 21, 26, 31, 36, 42, 47, 52, 57, 63, 68, 73, 78, 84, 89, 94, 100]

# Repeating the same process for parallel (assuming it's similar to origin_en)
_needlebench_8k_parallel_en_batch1 = []
_needlebench_8k_parallel_en_batch5 = []
_needlebench_8k_parallel_en_batch10 = []
_needlebench_8k_parallel_en_batch15 = []
_needlebench_8k_parallel_en_batch20 = []
_needlebench_8k_parallel_zh_batch1 = []
_needlebench_8k_parallel_zh_batch5 = []
_needlebench_8k_parallel_zh_batch10 = []
_needlebench_8k_parallel_zh_batch15 = []
_needlebench_8k_parallel_zh_batch20 = []
for original_context_length in context_lengths_8k:
    _needlebench_8k_parallel_en_batch1.append(f'Length{original_context_length}_parallel_en_8k_batch1')
    _needlebench_8k_parallel_en_batch5.append(f'Length{original_context_length}_parallel_en_8k_batch5')
    _needlebench_8k_parallel_en_batch10.append(f'Length{original_context_length}_parallel_en_8k_batch10')
    _needlebench_8k_parallel_en_batch15.append(f'Length{original_context_length}_parallel_en_8k_batch15')
    _needlebench_8k_parallel_en_batch20.append(f'Length{original_context_length}_parallel_en_8k_batch20')
    _needlebench_8k_parallel_zh_batch1.append(f'Length{original_context_length}_parallel_zh_8k_batch1')
    _needlebench_8k_parallel_zh_batch5.append(f'Length{original_context_length}_parallel_zh_8k_batch5')
    _needlebench_8k_parallel_zh_batch10.append(f'Length{original_context_length}_parallel_zh_8k_batch10')
    _needlebench_8k_parallel_zh_batch15.append(f'Length{original_context_length}_parallel_zh_8k_batch15')
    _needlebench_8k_parallel_zh_batch20.append(f'Length{original_context_length}_parallel_zh_8k_batch20')


_needlebench_8k_parallel_batch1 = _needlebench_8k_parallel_en_batch1 + _needlebench_8k_parallel_zh_batch1
_needlebench_8k_parallel_batch5 = _needlebench_8k_parallel_en_batch5 + _needlebench_8k_parallel_zh_batch5
_needlebench_8k_parallel_batch10 = _needlebench_8k_parallel_en_batch10 + _needlebench_8k_parallel_zh_batch10
_needlebench_8k_parallel_batch15 = _needlebench_8k_parallel_en_batch15 + _needlebench_8k_parallel_zh_batch15
_needlebench_8k_parallel_batch20 = _needlebench_8k_parallel_en_batch20 + _needlebench_8k_parallel_zh_batch20

needlebench_summary_groups = [
    {'name': 'parallel_version_batch1', 'subsets': _needlebench_8k_parallel_batch1},
    {'name': 'parallel_version_zh_batch1', 'subsets': _needlebench_8k_parallel_zh_batch1},
    {'name': 'parallel_version_en_batch1', 'subsets': _needlebench_8k_parallel_en_batch1},
    {'name': 'parallel_version_batch5', 'subsets': _needlebench_8k_parallel_batch5},
    {'name': 'parallel_version_zh_batch5', 'subsets': _needlebench_8k_parallel_zh_batch5},
    {'name': 'parallel_version_en_batch5', 'subsets': _needlebench_8k_parallel_en_batch5},
    {'name': 'parallel_version_batch10', 'subsets': _needlebench_8k_parallel_batch10},
    {'name': 'parallel_version_zh_batch10', 'subsets': _needlebench_8k_parallel_zh_batch10},
    {'name': 'parallel_version_en_batch10', 'subsets': _needlebench_8k_parallel_en_batch10},
    {'name': 'parallel_version_batch15', 'subsets': _needlebench_8k_parallel_batch15},
    {'name': 'parallel_version_zh_batch15', 'subsets': _needlebench_8k_parallel_zh_batch15},
    {'name': 'parallel_version_en_batch15', 'subsets': _needlebench_8k_parallel_en_batch15},
    {'name': 'parallel_version_batch20', 'subsets': _needlebench_8k_parallel_batch20},
    {'name': 'parallel_version_zh_batch20', 'subsets': _needlebench_8k_parallel_zh_batch20},
    {'name': 'parallel_version_en_batch20', 'subsets': _needlebench_8k_parallel_en_batch20},
]
summarizer = dict(
    dataset_abbrs=[
        '--------- NeedleBench-8k Parallel-Needles ---------',  # category
        'parallel_version_batch1',
        'parallel_version_zh_batch1',
        'parallel_version_en_batch1',
        'parallel_version_batch5',
        'parallel_version_zh_batch5',
        'parallel_version_en_batch5',
        'parallel_version_batch10',
        'parallel_version_zh_batch10',
        'parallel_version_en_batch10',
        'parallel_version_batch15',
        'parallel_version_zh_batch15',
        'parallel_version_en_batch15',
        'parallel_version_batch20',
        'parallel_version_zh_batch20',
        'parallel_version_en_batch20',
        # *_needlebench_8k_origin, *_needlebench_8k_multi_needle, *_needlebench_8k_parallel,
    ],
    summary_groups=needlebench_summary_groups,
)
