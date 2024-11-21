from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import GSM8KDataset, gsm8k_postprocess, gsm8k_dataset_postprocess, Gsm8kEvaluator

gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')

gsm8k_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt="Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nAnswer:"),
                dict(role='BOT', prompt="There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.\n"),
                dict(role='HUMAN', prompt="Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\nAnswer:"),
                dict(role='BOT', prompt="There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.\n"),
                dict(role='HUMAN', prompt="Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nAnswer:"),
                dict(role='BOT', prompt="Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.\n"),
                dict(role='HUMAN', prompt="Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\nAnswer:"),
                dict(role='BOT', prompt="Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.\n"),
                dict(role='HUMAN', prompt="Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\nAnswer:"),
                dict(role='BOT', prompt="Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.\n"),
                dict(role='HUMAN', prompt="Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\nAnswer:"),
                dict(role='BOT', prompt="There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.\n"),
                dict(role='HUMAN', prompt="Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\nAnswer:"),
                dict(role='BOT', prompt="Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.\n"),
                dict(role='HUMAN', prompt="Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\nAnswer:"),
                dict(role='BOT', prompt="Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left.23 - 15 is 8. The answer is 8.\n"),
                dict(role='HUMAN', prompt="Question: {question}\nAnswer:"),
            ],
        )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512))

gsm8k_eval_cfg = dict(evaluator=dict(type=Gsm8kEvaluator),
                      pred_postprocessor=dict(type=gsm8k_postprocess),
                      dataset_postprocessor=dict(type=gsm8k_dataset_postprocess))

gsm8k_datasets = [
    dict(
        abbr='gsm8k',
        type=GSM8KDataset,
        path='opencompass/gsm8k',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg,
        eval_cfg=gsm8k_eval_cfg)
]
