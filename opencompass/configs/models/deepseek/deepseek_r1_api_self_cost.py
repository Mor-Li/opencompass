from opencompass.models import OpenAISDK

api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
], )

models = [
    dict(
        abbr='deepseek_r1_offcial',
        type=OpenAISDK,
        path='deepseek-reasoner',  # 模型名称
        key='sk-c397c14fa7b14c428191b5f8758f44f8',  # API 密钥
        openai_api_base='https://api.deepseek.com/v1',  # API 端点
        query_per_second=10,  # 每秒最大查询次数
        max_out_len=8192,
        max_seq_len=32768,
        batch_size=1,        # 批处理大小
        # system_prompt='',    # 系统提示词
        meta_template=api_meta_template,
        run_cfg=dict(num_gpus=0),  # API 模型不需要 GPU
        retry=10,              # 增加重试次数
        temperature=0,      # 设置温度参数
        verbose=True,  # 启用详细日志
    )
]
