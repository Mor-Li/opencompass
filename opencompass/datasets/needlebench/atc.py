# flake8: noqa
import json
import os
import random
import re
from enum import Enum
from datasets import Dataset

from opencompass.datasets.base import BaseDataset
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET, TEXT_POSTPROCESSORS
from opencompass.openicl.icl_evaluator import BaseEvaluator

from opencompass.utils import get_data_path
from opencompass.datasets.math import extract_boxed_answer

# 定义问题类型枚举
class QuestionType(Enum):
    ELDEST_ANCESTOR = 0       # 最年长祖先
    NTH_ANCESTOR = 1          # N级祖先
    NTH_DESCENDANT = 2        # N级子节点
    RELATIONSHIP_DISTANCE = 3 # 关系距离

# 定义关系术语的代数映射（一代关系还是两代关系）
relationship_generation_map_zh = {
    '父亲': 1,
    '母亲': 1,
    '爸爸': 1,
    '妈妈': 1,
    '爷爷': 2,
    '奶奶': 2,
    '姥姥': 2,
    '姥爷': 2,
    '外公': 2,
    '外婆': 2,
}

relationship_generation_map_en = {
    'father': 1,
    'mother': 1,
    'dad': 1,
    'mom': 1,
    'grandfather': 2,
    'grandmother': 2,
    'maternal grandmother': 2,
    'maternal grandfather': 2,
    'paternal grandfather': 2,
    'paternal grandmother': 2,
}

relationship_templates_zh_CN = [
    '{A}是{B}的{relationship}。',
    '{B}的{relationship}是{A}。',
    '{A}作为{B}的{relationship}，对{B}的成长有重要影响。',
    '{A}不仅是{B}的{relationship}，还是{B}的榜样。',
    '{A}在{B}的成长过程中，不仅仅是{B}的{relationship}，还是{B}的监护人。',
    '{A}对{B}来说，不只是一个{relationship}，还是一个朋友。',
]

relationship_terms_zh_CN = [
    '父亲',
    '母亲',
    '爸爸',
    '妈妈',
    '爷爷',
    '奶奶',
    '姥姥',
    '姥爷',
    '外公',
    '外婆',
]

relationship_terms_en = [
    'father',
    'mother',
    'dad',
    'mom',
    'grandfather',
    'grandmother',
    'maternal grandmother',
    'maternal grandfather',
    'paternal grandfather',
    'paternal grandmother',
]

relationship_templates_en = [
    "{A} is {B}'s {relationship}.",
    "{B}'s {relationship} is {A}.",
    ("{A}, as {B}'s {relationship}, "
     "has a significant impact on {B}'s upbringing."),
    ("{A} is not only {B}'s {relationship} "
     "but also {B}'s role model."),
    ("During {B}'s upbringing, {A} was not only {B}'s {relationship}, "
     "but also {B}'s guardian."),
    ('For {B}, {A} is not just a {relationship}, '
     'but also a friend.'),
    "For {B}, {A} is more than just a {relationship}; {A} is a lifelong mentor of {B}.",
]

# 最年长祖先问题模板
shuffled_story_with_prompt_zh_CN = """下面是对你的多步推理能力的测试，这个测试叫做祖先追溯测试，我们会模拟不同人的家庭亲属关系，你的任务是在其中不断推理，直到找到最年长的祖先。
                
例如：
例子1.如果张强的父亲是马克，除此以外提供的文本中没有更多关于亲属关系的信息，那么在提供的文本中张强能够向上追溯到的最年长的亲人就是马克。
例子2.如果李明的姥姥是张红，而张红的父亲是张强，除此以外提供的文本中没有更多关于亲属关系的信息，那么在提供的文本中李明能够向上追溯到的最年长的亲人就是张强。
例子3.如果小明是张红的曾孙女，张红的祖母是王华，王华的父亲是王刚，除此以外提供的文本中没有更多关于亲属关系的信息，那么小明能够向上追溯到的最年长的亲人就是王刚。

注意：
1. 你不必纠结这个测试中的人名的性别关系，例如，一个通常被视为女性化的名字仍然可以是其他人的父亲，我们的重点是谁更年长。
2. 忽略这个测试中的姓氏遗传问题，例如，李明仍然可能是王鹏的亲生父亲，我们只关注谁更年长，不必纠结孩子是否应该继承父亲或母亲的性别。
3. 在回答的最后，将你的答案放在\\boxed{{}}中，例如："所以{last_person}能向上追溯到的最年长的亲人就是\\boxed{{某人（你的答案）}}"

现在，打乱的家族关系文本如下：
{shuffled_story}

在上面提供的打乱的家族关系文本中，'{last_person}'的能够向上追溯到的最年长的亲人是谁？
"""

shuffled_story_with_prompt_en = """Here is a test for multi-step reasoning ability called the Ancestral Trace Challenge. In this test, we will simulate different people's familial relationships, and your task is to continuously reason through them until you identify the eldest ancestor.

For example:
Example 1: If James Hill's father is Jasmine Lane, and no further information about familial relationships is provided in the text, then the oldest relative James Hill can trace back to in the provided text is \\boxed{{Jasmine Lane}}.
Example 2: If Andrew Williams's grandmother is Dan Newton, and Dan Newton's father is James Hill, and no further information about familial relationships is provided in the text, then the oldest relative Andrew Williams can trace back to in the provided text is \\boxed{{James Hill}}.
Example 3: If Jeff White's father is Kevin Le, Dan Newton's grandmother is Jeff White, and Jeff White's father is Kevin Le, and Shelley Mills is Dan Newton's great-granddaughter, and no further information about familial relationships is provided in the text, then the oldest relative Shelley Mills can trace back to in the provided text is \\boxed{{Kevin Le}}.

Notes:
1. You do not need to worry about the gender consistency of names in this test. For example, a name that is typically considered feminine can still be the father of another person. Our primary focus is on who is older.
2. Ignore surname inheritance issues. For instance, Andrew Williams could still be the biological father of Christopher Baker. We only care about who is older and do not need to consider whether a child should inherit the father's or mother's surname.
3. At the end of your response, remember to put your final answer within \\boxed{{}}. For example: "So the oldest relative '{last_person}' can trace back to in the provided text is \\boxed{{somebody (your answer here)}}."

Now, the scrambled family relationships are provided below:
{shuffled_story}

Given the scrambled family relationships described above, who is the eldest relative that '{last_person}' can trace back to in the context?
"""

# 新增的N级祖先问题模板
nth_ancestor_prompt_zh_CN = """下面是对你的多步推理能力的测试，这个测试叫做祖先追溯测试，我们会模拟不同人的家庭亲属关系，你的任务是在其中不断推理，找到指定人物的特定代祖先。

例如：
例子1.如果张强的父亲是马克，我们说马克是张强的1代祖先。
例子2.如果李明的姥姥是张红（姥姥算两代关系），而张红的父亲是张强，那么张红是李明的2代祖先，张强是李明的3代祖先。
例子3.如果小明的奶奶是王华（奶奶算两代关系），王华的妈妈是刘芳，那么王华是小明的2代祖先，刘芳是小明的3代祖先。

注意：
1. 你不必纠结这个测试中的人名的性别关系，我们只关注辈分关系。
2. 忽略这个测试中的姓氏遗传问题，我们只关注亲属关系。
3. 父亲/母亲/爸爸/妈妈算1代关系，爷爷/奶奶/姥姥/姥爷/外公/外婆算2代关系。
4. 在回答的最后，将你的答案放在\\boxed{{}}中，例如："所以{person}的{n}代祖先就是\\boxed{{某人（你的答案）}}"

现在，打乱的家族关系文本如下：
{shuffled_story}

在上面提供的打乱的家族关系文本中，'{person}'的{n}代祖先是谁？
"""

nth_ancestor_prompt_en = """Here is a test for multi-step reasoning ability called the Ancestral Trace Challenge. In this test, we will simulate different people's familial relationships, and your task is to identify a specific ancestor of a given person.

For example:
Example 1: If James Hill's father is Jasmine Lane, then Jasmine Lane is James Hill's 1st generation ancestor.
Example 2: If Andrew Williams's grandmother is Dan Newton (grandmother counts as 2 generations), and Dan Newton's father is James Hill, then Dan Newton is Andrew Williams's 2nd generation ancestor, and James Hill is Andrew Williams's 3rd generation ancestor.
Example 3: If Shelley Mills's grandfather is Jeff White (grandfather counts as 2 generations), and Jeff White's mother is Mary Johnson, then Jeff White is Shelley Mills's 2nd generation ancestor, and Mary Johnson is Shelley Mills's 3rd generation ancestor.

Notes:
1. You do not need to worry about the gender consistency of names in this test. We only care about generational relationships.
2. Ignore surname inheritance issues. We only care about familial relationships.
3. Father/mother/dad/mom count as 1 generation, while grandfather/grandmother/maternal grandmother/maternal grandfather/paternal grandfather/paternal grandmother count as 2 generations.
4. At the end of your response, remember to put your final answer within \\boxed{{}}. For example: "So the {n}th generation ancestor of '{person}' is \\boxed{{somebody (your answer here)}}."

Now, the scrambled family relationships are provided below:
{shuffled_story}

Given the scrambled family relationships described above, who is the {n}th generation ancestor of '{person}'?
"""

# 新增的N级子节点问题模板
nth_descendant_prompt_zh_CN = """下面是对你的多步推理能力的测试，这个测试叫做家族关系追溯测试，我们会模拟不同人的家庭亲属关系，你的任务是在其中不断推理，找到指定人物的特定代子孙。

例如：
例子1.如果马克是张强的父亲，我们说张强是马克的1代子孙。
例子2.如果张红是李明的姥姥（姥姥算两代关系），而张强是张红的父亲，那么李明是张红的2代子孙，李明是张强的3代子孙。
例子3.如果王华是小明的爷爷（爷爷算两代关系），刘芳是王华的妈妈，那么小明是王华的2代子孙，小明是刘芳的3代子孙。

注意：
1. 你不必纠结这个测试中的人名的性别关系，我们只关注辈分关系。
2. 忽略这个测试中的姓氏遗传问题，我们只关注亲属关系。
3. 父亲/母亲/爸爸/妈妈算1代关系，爷爷/奶奶/姥姥/姥爷/外公/外婆算2代关系。
4. 在回答的最后，将你的答案放在\\boxed{{}}中，例如："所以{person}的{n}代子孙就是\\boxed{{某人（你的答案）}}"

现在，打乱的家族关系文本如下：
{shuffled_story}

在上面提供的打乱的家族关系文本中，'{person}'的{n}代子孙是谁？
"""

nth_descendant_prompt_en = """Here is a test for multi-step reasoning ability called the Ancestral Trace Challenge. In this test, we will simulate different people's familial relationships, and your task is to identify a specific descendant of a given person.

For example:
Example 1: If Jasmine Lane is James Hill's father, then James Hill is Jasmine Lane's 1st generation descendant.
Example 2: If Dan Newton is Andrew Williams's grandmother (grandmother counts as 2 generations), and James Hill is Dan Newton's father, then Andrew Williams is Dan Newton's 2nd generation descendant, and Andrew Williams is James Hill's 3rd generation descendant.
Example 3: If Jeff White is Shelley Mills's grandfather (grandfather counts as 2 generations), and Mary Johnson is Jeff White's mother, then Shelley Mills is Jeff White's 2nd generation descendant, and Shelley Mills is Mary Johnson's 3rd generation descendant.

Notes:
1. You do not need to worry about the gender consistency of names in this test. We only care about generational relationships.
2. Ignore surname inheritance issues. We only care about familial relationships.
3. Father/mother/dad/mom count as 1 generation, while grandfather/grandmother/maternal grandmother/maternal grandfather/paternal grandfather/paternal grandmother count as 2 generations.
4. At the end of your response, remember to put your final answer within \\boxed{{}}. For example: "So the {n}th generation descendant of '{person}' is \\boxed{{somebody (your answer here)}}."

Now, the scrambled family relationships are provided below:
{shuffled_story}

Given the scrambled family relationships described above, who is the {n}th generation descendant of '{person}'?
"""

# 新增的关系距离问题模板
relationship_distance_prompt_zh_CN = """下面是对你的多步推理能力的测试，这个测试叫做家族关系追溯测试，我们会模拟不同人的家庭亲属关系，你的任务是在其中不断推理，计算两个人之间的关系距离。

关系距离定义为：家族图中从一个人到另一个人所需的最少代数差距。注意不同关系有不同的代数差距，例如：
例子1.如果马克是张强的父亲（父亲算1代关系），那么张强和马克之间的关系距离是1。
例子2.如果张红是李明的姥姥（姥姥算2代关系），而张强是张红的父亲（父亲算1代关系），那么李明和张红之间的关系距离是2，李明和张强之间的关系距离是3。
例子3.如果小明的爷爷是王华（爷爷算2代关系），王华的妈妈是刘芳（妈妈算1代关系），那么小明和王华之间的关系距离是2，小明和刘芳之间的关系距离是3。

注意：
1. 你不必纠结这个测试中的人名的性别关系，我们只关注辈分关系。
2. 忽略这个测试中的姓氏遗传问题，我们只关注亲属关系。
3. 父亲/母亲/爸爸/妈妈算1代关系，爷爷/奶奶/姥姥/姥爷/外公/外婆算2代关系。
4. 在回答的最后，将你的答案放在\\boxed{{}}中，例如："所以{person_a}和{person_b}之间的关系距离是\\boxed{{5}}"

现在，打乱的家族关系文本如下：
{shuffled_story}

在上面提供的打乱的家族关系文本中，'{person_a}'和'{person_b}'之间的关系距离是多少？
"""

relationship_distance_prompt_en = """Here is a test for multi-step reasoning ability called the Ancestral Trace Challenge. In this test, we will simulate different people's familial relationships, and your task is to calculate the relationship distance between two individuals.

The relationship distance is defined as the minimum number of generational gaps needed to go from one person to another in the family graph. Note that different relationships have different generational gaps. For example:
Example 1: If Jasmine Lane is James Hill's father (father counts as 1 generation), then the relationship distance between James Hill and Jasmine Lane is 1.
Example 2: If Dan Newton is Andrew Williams's grandmother (grandmother counts as 2 generations), and James Hill is Dan Newton's father (father counts as 1 generation), then the relationship distance between Andrew Williams and Dan Newton is 2, and the relationship distance between Andrew Williams and James Hill is 3.
Example 3: If Jeff White is Shelley Mills's grandfather (grandfather counts as 2 generations), and Mary Johnson is Jeff White's mother (mother counts as 1 generation), then the relationship distance between Shelley Mills and Jeff White is 2, and the relationship distance between Shelley Mills and Mary Johnson is 3.

Notes:
1. You do not need to worry about the gender consistency of names in this test. We only care about relationship connections.
2. Ignore surname inheritance issues. We only care about familial relationships.
3. Father/mother/dad/mom count as 1 generation, while grandfather/grandmother/maternal grandmother/maternal grandfather/paternal grandfather/paternal grandmother count as 2 generations.
4. At the end of your response, remember to put your final answer within \\boxed{{}}. For example: "So the relationship distance between '{person_a}' and '{person_b}' is \\boxed{{5}}."

Now, the scrambled family relationships are provided below:
{shuffled_story}

Given the scrambled family relationships described above, what is the relationship distance between '{person_a}' and '{person_b}'?
"""

@LOAD_DATASET.register_module()
class NeedleBenchATCDataset(BaseDataset):

    @staticmethod
    def load(
        path,
        file_name: str,
        num_needles: int,
        language: str,
        repeats: int,
        question_types=None,  # 支持指定问题类型列表
    ):
        data = {'prompt': [], 'answer': []}
        path = get_data_path(path)
        if os.environ.get('DATASET_SOURCE') == 'HF':
            from huggingface_hub import snapshot_download

            path = snapshot_download(repo_id=path, repo_type='dataset')
        file_path = os.path.join(path, file_name)

        with open(file_path, 'r', encoding='utf-8') as file:
            names_data = json.load(file)

        all_names = names_data[language].split(',')
        # 确保question_types非空
        if not question_types:
            question_types = [QuestionType.ELDEST_ANCESTOR]
        
        # 确保question_types是QuestionType枚举类型
        converted_question_types = []
        for qt in question_types:
            if isinstance(qt, QuestionType):
                converted_question_types.append(qt)
            elif isinstance(qt, str) and hasattr(QuestionType, qt):
                # 处理字符串类型的question_type
                converted_question_types.append(getattr(QuestionType, qt))
            elif hasattr(qt, "name") and hasattr(QuestionType, qt.name):
                # 处理具有name属性的对象
                converted_question_types.append(getattr(QuestionType, qt.name))
            else:
                # 尝试从值转换回QuestionType
                try:
                    qt_value = int(qt) if not isinstance(qt, int) else qt
                    for enum_qt in QuestionType:
                        if enum_qt.value == qt_value:
                            converted_question_types.append(enum_qt)
                            break
                    else:
                        print(f"Warning: Cannot convert {qt} to QuestionType, using ELDEST_ANCESTOR instead")
                        converted_question_types.append(QuestionType.ELDEST_ANCESTOR)
                except (ValueError, TypeError):
                    print(f"Warning: Cannot convert {qt} to QuestionType, using ELDEST_ANCESTOR instead")
                    converted_question_types.append(QuestionType.ELDEST_ANCESTOR)
        
        if not converted_question_types:
            converted_question_types = [QuestionType.ELDEST_ANCESTOR]
            
        # print(f"Using question types: {[qt.name for qt in converted_question_types]}")

        for question_type in converted_question_types:
            # print(f"Generating examples for question type: {question_type.name}")
            # 为每个问题类型生成指定数量的示例
            for i in range(repeats):
                # 为每个问题类型和重复次数设置不同的种子
                # 使用问题类型的枚举值乘以10000作为基础，确保不同问题类型的种子范围不重叠
                seed = (i + 1) + (10000 * question_type.value)
                random.seed(seed)
                
                # 从所有名字中随机选择指定数量的名字
                names = random.sample(all_names, num_needles)
                
                # 根据语言选择相应的关系术语和模板
                if language == 'Chinese':
                    relationship_terms = relationship_terms_zh_CN
                    relationship_templates = relationship_templates_zh_CN
                    relationship_map = relationship_generation_map_zh
                elif language == 'English':
                    relationship_terms = relationship_terms_en
                    relationship_templates = relationship_templates_en
                    relationship_map = relationship_generation_map_en
                else:
                    raise ValueError('Unsupported language specified. '
                                    'Please choose either "Chinese" or "English".')

                def generate_chain_family_story(names, templates, relationship_terms, relationship_map):
                    story = ''
                    relationships = []
                    total_generations = 0  # 跟踪总代数差异

                    for i in range(len(names) - 1):
                        template = random.choice(templates)
                        relation_term = random.choice(relationship_terms)
                        relation = template.format(A=names[i],
                                                  B=names[i + 1],
                                                  relationship=relation_term)
                        story += f'{relation}*'

                        # 获取这种关系对应的代数
                        gen_diff = relationship_map.get(relation_term, 1)  # 默认为1代
                        # print(f"[Generation Calculation] {names[i]} 是 {names[i+1]} 的 {relation_term}，本次代数差为 {gen_diff}，累计总代数: {total_generations} -> {total_generations + gen_diff}")
                        total_generations += gen_diff

                        # 记录关系信息以便后续使用
                        relationships.append((names[i], names[i + 1], relation_term, gen_diff))

                    return story, relationships, total_generations

                chain_story, relationships, total_generations = generate_chain_family_story(
                    names, relationship_templates, relationship_terms, relationship_map)

                # Splitting the chain_story into a list of fragments
                family_story_fragments = chain_story.split('*')
                # 移除空字符串
                family_story_fragments = [f for f in family_story_fragments if f]

                # Shuffling the list of fragments
                random.shuffle(family_story_fragments)

                # Joining the shuffled fragments back into a string
                shuffled_story = ''.join(family_story_fragments)

                # 根据问题类型生成相应的提示和答案
                if question_type == QuestionType.ELDEST_ANCESTOR:
                    # 最年长祖先问题
                    last_person = names[-1]
                    if language == 'Chinese':
                        prompt = shuffled_story_with_prompt_zh_CN.format(
                            shuffled_story=shuffled_story, last_person=last_person)
                    else:
                        prompt = shuffled_story_with_prompt_en.format(
                            shuffled_story=shuffled_story, last_person=last_person)
                    answer = names[0]  # 第一个人是最年长的祖先

                elif question_type == QuestionType.NTH_ANCESTOR:
                    # N级祖先问题 - 从最年轻的人开始，向上追溯到最老的人
                    person = names[-1]  # 最年轻的人（链条末尾）
                    n = total_generations  # 使用计算出的总代数差异
                    if language == 'Chinese':
                        prompt = nth_ancestor_prompt_zh_CN.format(
                            shuffled_story=shuffled_story, person=person, n=n)
                    else:
                        prompt = nth_ancestor_prompt_en.format(
                            shuffled_story=shuffled_story, person=person, n=n)
                    answer = names[0]  # 最老的人（链条开头）是第n代祖先

                elif question_type == QuestionType.NTH_DESCENDANT:
                    # N级子孙问题 - 从最老的人开始，向下追溯到最年轻的人
                    person = names[0]  # 最老的人（链条开头）
                    n = total_generations  # 使用计算出的总代数差异
                    if language == 'Chinese':
                        prompt = nth_descendant_prompt_zh_CN.format(
                            shuffled_story=shuffled_story, person=person, n=n)
                    else:
                        prompt = nth_descendant_prompt_en.format(
                            shuffled_story=shuffled_story, person=person, n=n)
                    answer = names[-1]  # 最年轻的人（链条末尾）是第n代子孙

                elif question_type == QuestionType.RELATIONSHIP_DISTANCE:
                    # 关系距离问题 - 计算链条最远两端的人之间的关系距离
                    person_a = names[0]  # 最老的人
                    person_b = names[-1]  # 最年轻的人
                    if language == 'Chinese':
                        prompt = relationship_distance_prompt_zh_CN.format(
                            shuffled_story=shuffled_story, person_a=person_a, person_b=person_b)
                    else:
                        prompt = relationship_distance_prompt_en.format(
                            shuffled_story=shuffled_story, person_a=person_a, person_b=person_b)
                    # 使用计算出的总代际数作为关系距离
                    answer = str(total_generations)

                else:
                    # 默认回退到最年长祖先问题
                    last_person = names[-1]
                    if language == 'Chinese':
                        prompt = shuffled_story_with_prompt_zh_CN.format(
                            shuffled_story=shuffled_story, last_person=last_person)
                    else:
                        prompt = shuffled_story_with_prompt_en.format(
                            shuffled_story=shuffled_story, last_person=last_person)
                    answer = names[0]  # 第一个人是最年长的祖先

                data['prompt'].append(prompt)
                data['answer'].append(answer)

        # 将每个样本的问题类型（question_type）加入到dataset中
        # 这里将其以字符串形式存储，便于后续处理
        # 假设每个样本都按顺序append到data中，因此可以直接生成对应的question_type列表
        # 由于外层循环是 for question_type in question_types: for i in range(repeats): append
        # 所以每个question_type会有repeats个样本，顺序与data['prompt']一致
        question_type_list = []
        for question_type in converted_question_types:
            for i in range(repeats):
                question_type_list.append(str(question_type.name if hasattr(question_type, "name") else str(question_type)))

        # 如果有多种question_type，question_type_list长度应与data['prompt']一致
        assert len(question_type_list) == len(data['prompt']), "question_type_list and data length mismatch"

        dataset = Dataset.from_dict({
            'prompt': data['prompt'],
            'answer': data['answer'],
            'question_type': question_type_list,
        })
        return dataset


def clean_atc_answer(text: str) -> str:
    """Clean answer format specifically for QwQ-32B-Preview model
    
    Args:
        text: Raw prediction text
        
    Returns:
        Standardized name format after cleaning
    """
    if not text or text == "None":
        return "None"
    
    # Remove LaTeX commands but keep content
    text = re.sub(r'\\text\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\\boxed\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\\[\[\]]', '', text)
    
    # Remove extra backslashes
    text = text.replace('\\\\', '').replace('\\', '')
    
    # Handle extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove quotes
    text = text.replace('"', '').replace("'", '')
    # Remove tildes (波浪符号)
    text = text.replace('~', ' ')
        
    return text

@TEXT_POSTPROCESSORS.register_module('needlebench_atc_postprocess_v2')
def needlebench_atc_postprocess_v2(text: str) -> str:

    cand_ans = extract_boxed_answer(text, strip_double_curly_brace=True)
    
    if cand_ans:
        return clean_atc_answer(cand_ans)
    return "None"


@ICL_EVALUATORS.register_module("needlebench_atc_evaluator")
class NeedleBenchATCEvaluator(BaseEvaluator):

    def score(self, predictions, gold):
        if len(predictions) != len(gold):
            return {'error': 'predictions and gold have different lengths'}

        correct_count = 0
        details = []
        
        for prediction, reference in zip(predictions, gold):
            reference_name = reference
            if prediction.strip() == reference_name.strip():
                correct_count += 1
            
            detail = {
                'pred': prediction,
                'answer': reference_name,
                'correct': prediction.strip() == reference_name.strip()
            }
            details.append(detail)

        accuracy = (correct_count / len(predictions)) * 100 if predictions else 0
        result = {'score': accuracy, 'details': details}
        return result
    
if __name__ == '__main__':
    import argparse
    import os
    import json
    
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='生成NeedleBench ATC数据集示例')
    parser.add_argument('--num_needles', type=int, default=2, help='链条中的人物数量')
    parser.add_argument('--language', type=str, default='English', choices=['English', 'Chinese'], help='语言')
    parser.add_argument('--repeats', type=int, default=2, help='每种问题类型生成的示例数量')
    parser.add_argument('--output_dir', type=str, default='needlebench_examples', help='输出目录')
    parser.add_argument('--all_types', action='store_true', help='生成所有问题类型的示例')
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 准备名字数据
    names_data = {
        'English': 'Alice,Bob,Charlie,David,Emma,Frank,Grace,Henry,Ivy,Jack,Kate,Leo,Mia,Noah,Olivia,Peter,Quinn,Ryan,Sophia,Thomas,Uma,Victor,Wendy,Xavier,Yara,Zack',
        'Chinese': '张伟,王芳,李娜,刘洋,陈明,杨丽,赵强,周红,吴刚,徐静,孙伟,马丽,胡明,朱红,谢强,郑静,黄伟,杨明,林丽,叶强,宋静,韩伟,唐丽,董强,梁静,冯伟,程丽,曹强,袁静,许伟'
    }
    
    # 如果找不到名字数据，就使用默认名字
    try:
        file_path = os.path.join('opencompass/needlebench', 'names.json')
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                names_data = json.load(file)
    except Exception as e:
        print(f"使用默认名字数据，无法加载names.json: {e}")
    
    # 设置问题类型
    if args.all_types:
        # 只包含前四种类型
        question_types = [
            QuestionType.ELDEST_ANCESTOR,
            QuestionType.NTH_ANCESTOR,
            QuestionType.NTH_DESCENDANT,
            QuestionType.RELATIONSHIP_DISTANCE
        ]
    else:
        question_types = [
            QuestionType.ELDEST_ANCESTOR,
            QuestionType.NTH_ANCESTOR,
            QuestionType.NTH_DESCENDANT,
            QuestionType.RELATIONSHIP_DISTANCE
        ]
    
    # 为每种问题类型生成示例
    examples = {}
    for qt in question_types:
        print(f"生成问题类型: {qt.name}")
        dataset = NeedleBenchATCDataset.load(
            path='opencompass/needlebench',
            file_name='names.json',
            num_needles=args.num_needles,
            language=args.language,
            repeats=args.repeats,
            question_types=[qt]
        )
        
        examples[qt.name] = []
        for i in range(len(dataset)):
            example = {
                'prompt': dataset[i]['prompt'],
                'answer': dataset[i]['answer']
            }
            examples[qt.name].append(example)
            
            # 打印一个示例
            if i == 0:
                print("\n" + "="*50)
                print(f"问题类型: {qt.name}")
                print("="*50)
                print(f"问题:\n{example['prompt']}")
                print("-"*50)
                print(f"标准答案: {example['answer']}")
                print("="*50 + "\n")
    
    # 将示例保存到文件
    output_file = os.path.join(args.output_dir, f'needlebench_atc_{args.language}_{args.num_needles}names.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    
    print(f"已保存所有示例到: {output_file}")
    
    # 汇总统计
    total_examples = sum(len(exs) for exs in examples.values())
    print(f"\n生成完成! 总共生成了 {total_examples} 个示例，涵盖 {len(examples)} 种问题类型")
    for qt_name, exs in examples.items():
        print(f"  - {qt_name}: {len(exs)} 个示例")