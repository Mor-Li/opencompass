from opencompass.models import OpenAISDK

api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
], )

models = [
    dict(
        abbr='o3-mini-2025-01-31',
        type=OpenAISDK,
        path='o3-mini-2025-01-31',
        key='ENV',  # The key will be obtained from $OPENAI_API_KEY, but you can write down your key here as well
        meta_template=api_meta_template,
        query_per_second=1,
        batch_size=1,
        temperature=1,
        max_completion_tokens=8192,
        reasoning_effort="medium",
        ), # you can change it for large reasoning inference cost, according to: https://platform.openai.com/docs/guides/reasoning
]
