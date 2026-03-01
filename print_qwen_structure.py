import mlx.core as mx
from mlx_lm import load

model, tokenizer = load("mlx-community/Qwen2.5-1.5B-Instruct-bf16")
print(model)
