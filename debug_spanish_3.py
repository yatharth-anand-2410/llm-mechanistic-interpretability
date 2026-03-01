import mlx.core as mx
from mlx_lm import load

model_id = "mlx-community/Qwen2.5-1.5B-Instruct-bf16"
model, tokenizer = load(model_id)

prompt = "English: Hello\nSpanish: Hola\nEnglish: Goodbye\nSpanish:"
tokens = mx.array([tokenizer.encode(prompt)])
logits = model(tokens)

for i in range(5):
    top_idx = mx.argmax(logits[0, -1, :]).item()
    prob = mx.softmax(logits[0, -1, :])[top_idx].item()
    print(f"Top {i+1}: {repr(tokenizer.decode([top_idx]))} ({prob:.2%})")
    logits[0, -1, top_idx] = -float('inf')

