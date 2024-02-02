from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig
# from lmdeploy.messages import ChatTemplateConfig


backend_config = TurbomindEngineConfig(rope_scaling_factor=2.0,
                                       session_len=210000,
                                       model_name="internlm2-chat-7b",
                                       tp=1)

internlm2_meta_template = dict(
    model_name="internlm2-chat-7b",
    round=[
        dict(role='HUMAN',
             begin='[UNUSED_TOKEN_146]user\n', end='[UNUSED_TOKEN_145]\n'),
        dict(role='BOT', begin='[UNUSED_TOKEN_146]assistant\n',
             end='[UNUSED_TOKEN_145]\n', generate=True),
    ],
)


pipe = pipeline(model_path='/mnt/petrelfs/share_data/zhangwenwei/models/llmit/exps/20231224/ampere_7B_3_0_0_FT_0_18rc32_32k_hf',
                backend_config=backend_config,
                # chat_template_config=internlm2_meta_template,
                model_name="internlm2-chat-7b"
                )

# prompt 可以替换为长文本的输入
prompt = '你好'

gen_config = GenerationConfig(top_p=0.8,
                              top_k=40,
                              temperature=0.8,
                              max_new_tokens=1024)
response = pipe(prompt, gen_config=gen_config)
print(response)
