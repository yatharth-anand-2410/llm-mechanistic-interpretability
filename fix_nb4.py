import json

with open("01_layerwise_logit_lens.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code" and "plot_logit_lens_trajectory" in "".join(cell["source"]) and "translation" in "".join(cell["source"]):
        cell["source"] = [
            "# Tracking a translation prompt (Few-shot formatting helps the model isolate the exact token)\n",
            "prompt = \"English: Goodbye\\nSpanish: Adios\\nEnglish: Hello\\nSpanish:\"\n",
            "plot_logit_lens_trajectory(prompt, \" Hol\", model, tokenizer)\n"
        ]
        
    elif cell["cell_type"] == "markdown" and "Hello" in "".join(cell["source"]):
        cell["source"] = [
            "## 4. Syntax vs. Factual Knowledge\n",
            "Do language models resolve grammar earlier than they resolve facts? Let's trace a translation prompt to see when the network decides to switch from English to Spanish."
        ]

with open("01_layerwise_logit_lens.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
