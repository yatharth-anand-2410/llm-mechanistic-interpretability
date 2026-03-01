import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Notebook 1: Layerwise Logit Lens Exploration\n",
                "\n",
                "**Goal:** Use the instrumentation built in Notebook 0 to analyze how the model forms its predictions across the depth of the network. We will watch the probability of the correct answer emerge layer by layer."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import mlx.core as mx\n",
                "from mlx_lm import load\n",
                "import matplotlib.pyplot as plt\n",
                "import numpy as np"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Load Model & Infrastructure\n",
                "We re-import the core utilities we built in Notebook 0."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "model_id = \"mlx-community/Qwen2.5-1.5B-Instruct-bf16\"\n",
                "model, tokenizer = load(model_id)\n",
                "\n",
                "def extract_hidden_states(model, prompt):\n",
                "    tokens = mx.array([tokenizer.encode(prompt)])\n",
                "    h = model.model.embed_tokens(tokens)\n",
                "    hidden_states = {}\n",
                "    \n",
                "    from mlx_lm.models.qwen2 import create_attention_mask\n",
                "    for i, layer in enumerate(model.model.layers):\n",
                "        mask = create_attention_mask(h, None)\n",
                "        h = layer(h, mask, None)\n",
                "        hidden_states[i] = h\n",
                "        \n",
                "    return tokens, hidden_states\n",
                "\n",
                "def apply_logit_lens(hidden_state, model):\n",
                "    h_normed = model.model.norm(hidden_state)\n",
                "    if hasattr(model, 'lm_head'):\n",
                "        return model.lm_head(h_normed)\n",
                "    elif hasattr(model.model, 'embed_tokens'):\n",
                "        if hasattr(model.model.embed_tokens, 'as_linear'):\n",
                "            return model.model.embed_tokens.as_linear(h_normed)\n",
                "        else:\n",
                "            return h_normed @ model.model.embed_tokens.weight.T\n",
                "\n",
                "def get_token_probability(logits, target_token_id):\n",
                "    # Get probabilities for the last token in the sequence\n",
                "    probs = mx.softmax(logits[0, -1, :])\n",
                "    return probs[target_token_id].item()\n",
                "\n",
                "def get_top_prediction(logits, tokenizer):\n",
                "    last_logits = logits[0, -1, :]\n",
                "    top_idx = mx.argmax(last_logits).item()\n",
                "    prob = mx.softmax(last_logits)[top_idx].item()\n",
                "    return tokenizer.decode([top_idx]), prob"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Reading the Model's Mind (Layer by Layer)\n",
                "Let's trace a factual prompt through all 28 layers to see what the model expects the next token to be at every stage of the computation."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "prompt = \"The capital of France is\"\n",
                "print(f\"Prompt: '{prompt}'\\n\")\n",
                "\n",
                "tokens, hidden_states = extract_hidden_states(model, prompt)\n",
                "\n",
                "print(f\"{'Layer':<8} | {'Top Prediction':<15} | {'Confidence'}\")\n",
                "print(\"-\" * 45)\n",
                "\n",
                "for layer_idx in range(len(hidden_states)):\n",
                "    logits = apply_logit_lens(hidden_states[layer_idx], model)\n",
                "    top_word, prob = get_top_prediction(logits, tokenizer)\n",
                "    print(f\"Layer {layer_idx:<2} | {repr(top_word):<15} | {prob:.2%}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "**Observation:** Notice how in the early layers, the model is mostly focused on grammar (predicting words like 'a', 'the', 'not'). The actual factual knowledge of 'Paris' doesn't get injected into the residual stream until the middle/late layers!"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Visualizing Knowledge Emergence\n",
                "To make this more rigorous, let's track the exact probability of our target token (' Paris') across all layers and plot it."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def plot_logit_lens_trajectory(prompt, target_word, model, tokenizer):\n",
                "    # Encode to find the exact token ID we want to track\n",
                "    target_token_id = tokenizer.encode(target_word)[0]\n",
                "    \n",
                "    tokens, hidden_states = extract_hidden_states(model, prompt)\n",
                "    \n",
                "    probabilities = []\n",
                "    for layer_idx in range(len(hidden_states)):\n",
                "        logits = apply_logit_lens(hidden_states[layer_idx], model)\n",
                "        prob = get_token_probability(logits, target_token_id)\n",
                "        probabilities.append(prob)\n",
                "        \n",
                "    plt.figure(figsize=(10, 5))\n",
                "    plt.plot(range(len(probabilities)), probabilities, marker='o', linewidth=2, color='#2c3e50')\n",
                "    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)\n",
                "    \n",
                "    plt.title(f\"Logit Lens Trajectory\\nPrompt: '{prompt}'\\nTarget: '{target_word}'\")\n",
                "    plt.xlabel(\"Transformer Layer\")\n",
                "    plt.ylabel(\"Probability\")\n",
                "    plt.grid(True, alpha=0.3)\n",
                "    plt.xticks(range(0, len(probabilities), 2))\n",
                "    plt.ylim(0, 1.05)\n",
                "    plt.show()\n",
                "\n",
                "# Let's test it on our factual prompt\n",
                "plot_logit_lens_trajectory(\"The capital of France is\", \" Paris\", model, tokenizer)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Syntax vs. Factual Knowledge\n",
                "Do language models resolve grammar earlier than they resolve facts? Let's trace a translation prompt to see when the network decides to switch from English to Spanish."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Tracking a translation prompt\n",
                "plot_logit_lens_trajectory(\"The Spanish translation for 'Hello' is\", \" Hola\", model, tokenizer)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "Compare the two graphs. You often find that syntax/format-related tokens emerge earlier and more smoothly, while pure factual lookups exhibit a sharper 'step-function' spike at a specific middle layer where the knowledge is stored."
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("01_layerwise_logit_lens.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)
