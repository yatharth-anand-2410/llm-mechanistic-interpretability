import mlx.core as mx
from mlx_lm import load

model_id = "mlx-community/Qwen2.5-1.5B-Instruct-bf16"
model, tokenizer = load(model_id)

prompts = [
    "Translate 'Hello' to Spanish: ",
    "English: Hello\nSpanish:",
    "How to say hello in Spanish?",
    "Hello in Spanish is"
]

for p in prompts:
    tokens = mx.array([tokenizer.encode(p)])
    logits = model(tokens)
    top_idx = mx.argmax(logits[0, -1, :]).item()
    print(f"Prompt: {repr(p)} -> Top Token: {repr(tokenizer.decode([top_idx]))}")

