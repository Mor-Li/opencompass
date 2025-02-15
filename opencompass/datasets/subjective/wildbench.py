# flake8: noqa: F401, F403
import json
import re
from collections import defaultdict

from datasets import Dataset, DatasetDict

from opencompass.datasets.subjective.compass_arena_subjective_bench import \
    get_element_counts
from opencompass.registry import DICT_POSTPROCESSORS, LOAD_DATASET
from opencompass.utils import get_data_path

from ..base import BaseDataset
from .utils import get_judgeanswer_and_reference

score_prompt = """# Instruction

You are an expert evaluator. Your task is to evaluate the quality of \
the responses generated by AI models.
We will provide you with the user query and an AI-generated responses.
You should first read the user query and the conversation history \
carefully for analyzing the task, and then evaluate the quality of \
the responses based on and rules provided below.

# Conversation between User and AI

## History
<|begin_of_history|>

{history}

<|end_of_history|>

## Current User Query
<|begin_of_query|>

{user_query}

<|end_of_query|>

## AI Response
<|begin_of_response|>

{prediction}

<|end_of_response|>


# Evaluation

## Checklist

<|begin_of_checklist|>

{checklist}

<|end_of_checklist|>

Please use this checklist to guide your evaluation, but do \
not limit your assessment to the checklist.

## Rules

You should compare the above response based on your analysis\
 of the user queries and the conversation history.
You should first write down your analysis and the checklist \
that you used for the evaluation, and then provide your \
assessment according to the checklist.
The scores are in the range of 1~10, where 1 means the \
response is very poor and 10 means the response is perfect.
Here are more detailed criteria for the scores:

- Score 1~2: The response is very poor and does not make sense at all.
- Score 3~4: The response is poor and does help user solve the problem\
 in a meaningful way.
- Score 5~6: The response is fair but has some issues (e.g., factual \
errors, hallucinations, missing key information).
- Score 7~8: The response is good enough but could be improved in some ways.
- Score 9~10: The response is perfect and provides helpful information that\
 can help user solve the problem.

## Output Format
First, please output your analysis for the model response, and then summarize\
 your assessment to two aspects: "strengths" and "weaknesses"; Finally, please\
 write down your rating for the assessment.

Please provide your evaluation results in the following json format by filling\
 in the placeholders in []:
```
{
    "strengths": "[analysis for the strengths of the response]",
    "weaknesses": "[analysis for the weaknesses of the response]",
    "score": "[1~10]"
}
```"""

pair_prompt = """# Instruction

You are an expert evaluator. Your task is to evaluate the quality of the \
responses generated by two AI models.
We will provide you with the user query and a pair of AI-generated \
responses (Response A and Response B).
You should first read the user query and the conversation history \
carefully for analyzing the task, and then evaluate the quality of the \
responses based on and rules provided below.

# Conversation between User and AI

## History
<|begin_of_history|>

{history}

<|end_of_history|>

## Current User Query
<|begin_of_query|>

{user_query}

<|end_of_query|>

## Response A
<|begin_of_response_A|>

{prediction}

<|end_of_response_A|>

## Response B
<|begin_of_response_B|>

{prediction2}

<|end_of_response_B|>

# Evaluation

## Checklist

<|begin_of_checklist|>

{checklist}

<|end_of_checklist|>

Please use this checklist to guide your evaluation, but do not limit your \
assessment to the checklist.

## Rules

You should compare the above two responses based on your analysis of the \
user queries and the conversation history.
You should first write down your analysis and the checklist that you used \
for the evaluation, and then provide your assessment according to the \
checklist.
There are five choices to give your final assessment: ["A++", "A+", \
"A=B", "B+", "B++"], which correspond to the following meanings:

- `A++`: Response A is much better than Response B.
- `A+`: Response A is only slightly better than Response B.
- `A=B`: Response A and B are of the same quality. Please use this \
choice sparingly.
- `B+`: Response B is only slightly better than Response A.
- `B++`: Response B is much better than Response A.


## Output Format
First, please output your analysis for each model response, and \
then summarize your assessment to three aspects: "reason A=B", \
"reason A>B", and "reason B>A", and finally make your choice for \
the final assessment.

Please provide your evaluation results in the following json \
format by filling in the placeholders in []:
```
{
    "analysis of A": "[analysis of Response A]",
    "analysis of B": "[analysis of Response B]",
    "reason of A=B": "[where Response A and B perform equally well]",
    "reason of A>B": "[where Response A is better than Response B]",
    "reason of B>A": "[where Response B is better than Response A]",
    "choice": "[A++ or A+ or A=B or B+ or B++]",
}
```
"""


def parse_conversation(conversation):
    # parse conversation into chat dialogue
    role_dict = {'user': 'HUMAN', 'assistant': 'assistant'}
    chat_round = []
    history = ''
    if len(conversation) > 0:
        for x in conversation[:-1]:
            if x['role'] == 'user':
                history += 'USER: ' + x['content'] + '\n\n'
            elif x['role'] == 'assistant':
                history += 'ASSISTANT: ' + x['content'] + '\n\n'

            chat_round.append({
                'role': role_dict[x['role']],
                'content': x['content']
            })

    last_query = conversation[-1]['content']
    chat_round.append({
        'role': role_dict[conversation[-1]['role']],
        'content': conversation[-1]['content'],
    })
    chat_round.append({'role': 'assistant', 'content': ''})

    return chat_round, last_query, history


@LOAD_DATASET.register_module()
class WildBenchDataset(BaseDataset):

    def load(self, path: str, K=-1, eval_mode='pair', *args, **kwargs):
        path = get_data_path(path, local_mode=True)
        dataset = DatasetDict()
        raw_data = []
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                item = json.loads(line)
                chat_round, last_query, history = parse_conversation(
                    item['turn'])

                checklist_mardkdown = ''
                for checklist_item in item['checklist']:
                    checklist_mardkdown += f'- {checklist_item}\n'

                if eval_mode == 'single':
                    prompt = score_prompt
                elif eval_mode == 'pair':
                    prompt = pair_prompt
                else:
                    assert NotImplementedError(
                        f'Eval mode {eval_mode} not in single or pair.')

                prompt = prompt.replace('{history}', history)
                prompt = prompt.replace('{user_query}', last_query)
                prompt = prompt.replace('{checklist}', checklist_mardkdown)

                raw_data.append({
                    'dialogue': chat_round,
                    'history': history,
                    'prompt': prompt,
                    'judge': {
                        'other': None,
                        'primary_tag': item['primary_tag'],
                        'secondary_tag': item['secondary_tag'],
                        'question_id': item['session_id'],
                    },
                })
        dataset = Dataset.from_list(raw_data)
        return dataset


task_group_new = {
    'Information seeking': 'Information/Advice seeking',
    'Creative Writing': 'Creative Tasks',
    'Coding & Debugging': 'Coding & Debugging',
    'Reasoning': 'Planning & Reasoning',
    'Editing': 'Creative Tasks',
    'Math': 'Math & Data Analysis',
    'Planning': 'Planning & Reasoning',
    'Brainstorming': 'Creative Tasks',
    'Role playing': 'Creative Tasks',
    'Advice seeking': 'Information/Advice seeking',
    'Data Analysis': 'Math & Data Analysis',
    'Others': 'Creative Tasks',
}


def post_process_wildbench_pair(judgement: dict):
    judgement = judgement['prediction']
    pattern = r"\"choice\": \"(.*?)\""
    matched_result = re.findall(pattern, judgement)
    if matched_result:
        return matched_result[0]
    else:
        return None


def post_process_wildbench_single(judgement: dict):
    judgement = judgement['prediction']
    pattern = r"\"score\": \"(.*?)\""
    matched_result = re.findall(pattern, judgement)
    try:
        score = float(matched_result[0])
        return {'score': score}
    except (ValueError, IndexError) as e:
        return None

    # if matched_result:
    #     score = float(matched_result[0])
    # else:
    #     return None
    # return {'score': score}


@DICT_POSTPROCESSORS.register_module('wildbench')
def wildbench_postprocess(
    output: dict,
    output_path: str,
) -> dict:

    judged_answers, references = get_judgeanswer_and_reference(
        result=output,
        filename=output_path,
        post_process=post_process_wildbench_pair,
    )

    if 'base_models' in references[0]:
        base_models = references[0]['base_models']
    else:
        base_models = ['HaiKu', 'gpt4-turbo', 'llama-2-70b-chat-hf']

    if isinstance(base_models, str):
        base_models = [base_models]

    win_base_model = defaultdict(float)
    win_compare_model = defaultdict(float)
    categories = defaultdict(float)

    score_mapping = {'A++': 1, 'A+': 0.5, 'A=B': 0, 'B+': -0.5, 'B++': -1}
    for judged_answer, reference in zip(judged_answers, references):
        if judged_answer not in score_mapping:
            continue

        flag = 1 if reference['answer1'] in base_models else -1
        score_1 = score_mapping[judged_answer] * flag
        score_2 = -score_1

        tags = [reference['primary_tag']] + reference['secondary_tag']
        for tag in tags:
            win_base_model[task_group_new[tag]] += score_1
            win_compare_model[task_group_new[tag]] += score_2
            categories[task_group_new[tag]] += 1

    for capability in categories:
        win_base_model[capability] = (win_base_model[capability] /
                                      categories[capability] * 100)
        win_base_model[capability] = round(win_base_model[capability], 2)
        win_compare_model[capability] = (win_compare_model[capability] /
                                         categories[capability] * 100)
        win_compare_model[capability] = round(win_compare_model[capability], 2)

    # Calculating the mean of the values
    average = sum(win_compare_model.values()) / len(win_compare_model)

    # Adding the mean to the dictionary at the beginning
    win_compare_model['average'] = average

    results = win_compare_model
    results['details'] = output
    return results


@DICT_POSTPROCESSORS.register_module('wildbench_bradleyterry')
def wildbench_bradleyterry_postprocess(
    output: dict,
    output_path: str,
) -> dict:

    judged_answers, references = get_judgeanswer_and_reference(
        result=output,
        filename=output_path,
        post_process=post_process_wildbench_pair,
    )

    if 'prediction1' not in references[0]:
        raise ValueError(
            'prediction1 not in references. Set `keep_predictions=True` for LMEvaluator in dataset config and retry.'
        )

    if 'prediction2' not in references[0]:
        raise ValueError(
            'prediction2 not in references. Set `keep_predictions=True` for LMEvaluator in dataset config and retry.'
        )

    score_mapping = {
        'A++': 'model_a',
        'A+': 'model_a',
        'A=B': 'tie',
        'B+': 'model_b',
        'B++': 'model_b',
    }

    results = {}
    matches = []
    for judged_answer, reference in zip(judged_answers, references):
        cur_dict = {}

        if judged_answer in score_mapping:
            cur_dict['winner'] = score_mapping[judged_answer]
        else:
            # cur_dict["winner"] = (
            #     "tie"  # Count match as tie if judge answer cannot be parsed.
            # )

            # Skip if judge answer cannot be parsed
            print('Judge answer cannot be parsed. Skipping record...')
            continue

        cur_dict['primary_tag'] = reference['primary_tag']
        # Extract first tag from list and set as categorical level.
        # Can be used as categorical variable in Bradley-Terry model
        cur_dict['secondary_tag'] = (reference['secondary_tag'][0]
                                     if len(reference['secondary_tag']) > 0
                                     else 'Others')
        # Keep original secondary tag list for reference
        cur_dict['secondary_tags'] = reference['secondary_tag']
        cur_dict['model_a'] = reference['answer1']
        cur_dict['model_b'] = reference['answer2']
        cur_dict['prediction1'] = reference['prediction1']
        cur_dict['prediction2'] = reference['prediction2']

        matches.append(cur_dict)

    ### ---------- Add Style Metadata ---------- ###
    matches = get_element_counts(
        data=matches,
        column='prediction1',
        suffix='_a',
    )
    matches = get_element_counts(
        data=matches,
        column='prediction2',
        suffix='_b',
    )

    results['matches'] = matches
    # results["details"] = output

    return results
