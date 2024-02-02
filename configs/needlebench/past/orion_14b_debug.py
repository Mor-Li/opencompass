from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("OrionStarAI/Orion-14B-LongChat", trust_remote_code=True)
print(tokenizer.decode([2]))
