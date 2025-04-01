from opencompass.models import OpenAISDK


api_meta_template = dict(
    round=[
            dict(role='HUMAN', api_role='HUMAN'),
            dict(role='BOT', api_role='BOT', generate=True),
    ],
)

models = [
    dict(abbr='gpt-4o-2024-11-20',
        type=OpenAISDK,
        path='gpt-4o-2024-11-20',
        key='ENV',  # The key will be obtained from $OPENAI_API_KEY, but you can write down your key here as well
        meta_template=api_meta_template,
        query_per_second=1,
        max_out_len=16384, max_seq_len=128000, batch_size=8),
]
