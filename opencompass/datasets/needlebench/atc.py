import random

from datasets import Dataset
from faker import Faker

from opencompass.datasets.base import BaseDataset
from opencompass.registry import LOAD_DATASET


@LOAD_DATASET.register_module()
class NeedleBenchATCDataset(BaseDataset):

    @staticmethod
    def load(
        # path: str,
        num_needles: int,
        language: str,
        repeats: int,
    ):
        data = {'prompt': [], 'answer': []}

        for _ in range(repeats):
            if language == 'Chinese':
                fake = Faker('zh_CN')
                names = [fake.unique.name() for _ in range(num_needles)]

                relationship_terms = [
                    '父亲', '母亲', '爸爸', '妈妈', '爷爷', '奶奶', '姥姥', '姥爷', '外公', '外婆'
                ]

                relationship_templates = [
                    '{A}是{B}的{relationship}。',
                    '{B}的{relationship}是{A}。',
                    '{A}作为{B}的{relationship}，对{B}的成长有重要影响。',
                    '{A}不仅是{B}的{relationship}，还是{B}的榜样。',
                    '{B}是{A}所生的孩子。',
                    '{A}对{B}来说，不只是一个{relationship}，还是一个朋友。',
                    '{A}在{B}的生命中扮演着{relationship}的角色。',
                    '{B}把{A}视为其{relationship}。',
                ]
            elif language == 'English':
                fake = Faker('en-US')
                names = [fake.unique.name() for _ in range(num_needles)]

                relationship_terms = [
                    'father', 'mother', 'dad', 'mom', 'grandfather',
                    'grandmother', 'maternal grandmother',
                    'maternal grandfather', 'paternal grandfather',
                    'paternal grandmother'
                ]

                relationship_templates = [
                    "{A} is {B}'s {relationship}.",
                    "{B}'s {relationship} is {A}.",
                    ("{A}, as {B}'s {relationship}, "
                     "has a significant impact on {B}'s upbringing."),
                    ("{A} is not only {B}'s {relationship} "
                     "but also {B}'s role model."),
                    '{B} is the child of {A}.',
                    ('For {B}, {A} is not just a {relationship}, '
                     'but also a friend.'),
                    ("{A} plays the role of {B}'s {relationship} "
                     "in {B}'s life."),
                    '{B} considers {A} as their {relationship}.',
                ]

            def generate_chain_family_story(names, templates,
                                            relationship_terms):
                story = ''
                for i in range(len(names) - 1):
                    template = random.choice(templates)
                    relation_term = random.choice(relationship_terms)
                    relation = template.format(A=names[i],
                                               B=names[i + 1],
                                               relationship=relation_term)
                    story += f'{relation}*'
                return story

            chain_story = generate_chain_family_story(names,
                                                      relationship_templates,
                                                      relationship_terms)

            # Splitting the chain_story into a list of fragments
            family_story_fragments = chain_story.split('*')

            # Shuffling the list of fragments
            random.shuffle(family_story_fragments)

            # Joining the shuffled fragments back into a string
            shuffled_story = ''.join(family_story_fragments)

            last_person = names[-1]

            # Generating the prompt based on the language
            if language == 'Chinese':
                prompt = f"\n在上面这个打乱的家族关系中，'{last_person}'的最原始祖宗是谁？"
            elif language == 'English':
                prompt = (f'\n In the above disrupted family relationship, who'
                          f" is '{last_person}'s most ancient ancestor?")
            else:
                prompt = 'Language not supported.'

            # Combine story and prompt
            shuffled_story_with_prompt = shuffled_story + ' ' + prompt

            # Output the shuffled story with prompt and the original ancestor
            # print(shuffled_story_with_prompt, )

            data['prompt'].append(shuffled_story_with_prompt)
            data['answer'].append(names[0] + '*' + names[0])

        dataset = Dataset.from_dict({
            'prompt': data['prompt'],
            'answer': data['answer'],
        })
        return dataset
