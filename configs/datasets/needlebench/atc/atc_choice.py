from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import CircularEvaluator
from opencompass.datasets.needlebench.atc_choice import NeedleBenchATCDataset
from opencompass.utils.text_postprocessors import first_option_postprocess

# ----------------------- Prompt Content----------------------- #

few_shot_prompts = {
    'single_choice_prompts': {
        "single_choice_cn": [
            dict(role='HUMAN', prompt='问题: 毕东作为刘红的爸爸，对刘红的成长有重要影响。 \n在上面提供的打乱的家族关系文本中，"刘红"的能够向上追溯到的最年长的亲人是谁？\nA. 毕东\nB. 刘红\nC. 李清亮\nD. 王展\n'),
            dict(role='BOT', prompt='回答: A'),
            dict(role='HUMAN', prompt='问题: 梅建平作为黄玉兰的姥姥，对黄玉兰的成长有重要影响。梅建平的妈妈是朱丽丽。蓝鑫把黄玉兰视为其母亲。焦慧不仅是朱丽丽的妈妈，还是朱丽丽的榜样。 \n在上面提供的打乱的家族关系文本中，"蓝鑫"的能够向上追溯到的最年长的亲人是谁？\nA. 梅建平\nB. 朱丽丽\nC. 蓝鑫\nD. 焦慧\n'),
            dict(role='BOT', prompt='回答: D'),
            dict(role='HUMAN', prompt='问题: 毕东把柳金凤视为其姥姥。奉兵作为柳金凤的妈妈，对柳金凤的成长有重要影响。余萍把杨颖视为其爸爸。毕东在郭建华的生命中扮演着父亲的角色。常宁的外公是余萍。刘慧是郭建华所生的孩子。刘慧在杨颖的生命中扮演着外公的角色。 \n在上面提供的打乱的家族关系文本中，"常宁"的能够向上追溯到的最年长的亲人是谁？\nA. 柳金凤\nB. 毕东\nC. 奉兵\nD. 余萍\n'),
            dict(role='BOT', prompt='回答: C'),
            dict(role='HUMAN', prompt='问题: 魏丽丽在谢平的生命中扮演着奶奶的角色。郭兵是魏阳的姥姥。谢平是郑玉珍的外婆。丁颖把武波视为其外公。丁颖在李建国的生命中扮演着外婆的角色。武波的父亲是刘瑜。许玲把余桂芳视为其父亲。刘瑜把许玲视为其爷爷。李建国对郭兵来说，不只是一个爷爷，还是一个朋友。魏丽丽的外公是魏阳。 \n在上面提供的打乱的家族关系文本中，"郑玉珍"的能够向上追溯到的最年长的亲人是谁？\nA. 魏丽丽\nB. 刘瑜\nC. 李建国\nD. 余桂芳\n'),
            dict(role='BOT', prompt='回答: D'),
            dict(role='HUMAN', prompt='问题: {question}'),
        ],
    },
}

# ----------------------- Prompt Settings ----------------------- #
needle_num_list = list(range(2, 20, 1))

names_path = './data/needlebench/names.json'

repeats = 10

# Use Zero-Shot or not
with_few_shot = True

# Max for this dataset is 4, should be set with `with_few_shot`
few_shot_samples = 4

# Generate reasoning path or not, only for single choice
with_reasoning = True

# Use circular evaluation or not
with_circular_eval = True

needlebench_prompts = few_shot_prompts
single_choice_prompts = needlebench_prompts['single_choice_prompts']

# Set few shot prompt number
for _name in list(single_choice_prompts.keys()):
    if with_few_shot:
        assert few_shot_samples > 0 and few_shot_samples <= 4
        single_choice_prompts[_name] = single_choice_prompts[_name][- few_shot_samples * 2 - 1:]

# ----------------------- Dataset Settings ----------------------- #

needlebench_datasets = []


needlebench_atc_reader_cfg = dict(input_columns=["question"], output_column="answer")

needlebench_atc_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=(single_choice_prompts['single_choice_cn'])),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer,),
)

needlebench_atc_eval_cfg = dict(
    evaluator=dict(type=CircularEvaluator),
    pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'))

for num_needles in needle_num_list:
    # ordered Chinese version
    dataset_dict = {
        'abbr': f'NeedleBenchATCDataset-'
                f'{num_needles}Needle-ZH',
        'type': NeedleBenchATCDataset,
        'path': names_path,
        'num_needles': num_needles,
        'language': 'Chinese',
        'repeats': repeats,
        'with_circular': with_circular_eval,
        'reader_cfg': needlebench_atc_reader_cfg,
        'infer_cfg': needlebench_atc_infer_cfg,
        'eval_cfg': needlebench_atc_eval_cfg
    }
    needlebench_datasets.append(dataset_dict)

