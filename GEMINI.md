# Gemini Context: Mechanistic Interpretability Research

## Project Overview
This project explores the internal "thought process" of **Llama 3 8B** (4-bit quantized) using MLX on Apple Silicon. We focus on identifying functional redundancy and the "critical path" of computation for different cognitive tasks.

## Current Model
- **Model ID:** `mlx-community/Meta-Llama-3-8B-Instruct-4bit`
- **Architecture:** 32 Transformer Layers, 4096-dimensional Residual Stream.

## Key Research Findings (March 2026)

### 1. Factual Knowledge Emergence
- **Step-Function Spike:** Factual recall (e.g., "The capital of France is") is not a gradual process. The model remains at ~0% confidence for " Paris" until **Layer 19**, where probability spikes instantly.
- **Arithmetic vs. Fact:** Simple patterns (e.g., "2+2=") emerge much earlier and more smoothly (starting around **Layer 8**).

### 2. The "Critical Path" Discovery
We identified that not all layers are equal. The model has specific architectural bottlenecks:
- **Bottlenecks:** Layers **0, 1, and 30** are non-redundant. Removing them causes total output collapse or "logit smearing" (loss of precision).
- **Redundancy:** 29 out of 32 layers are individually removable for simple tasks.

### 3. The "Middle Void" Theory
Our most significant discovery: For shallow tasks (facts and simple arithmetic), **60% of the model is mathematically optional.**
- **Finding:** We can remove a contiguous window of **19 layers (Layers 3 through 21)** and the model still predicts correctly.
- **Interpretation:** The middle layers act as a "mathematical conveyor belt." If the prompt is encoded correctly in the first 3 layers, the vector can bypass the middle and still trigger the retrieval heads in the final layers.

## Repository Structure
- **`00_model_instrumentation.ipynb`**: Infrastructure for hidden state extraction and attention monkey-patching.
- **`01_layerwise_logit_lens.ipynb`**: The primary experiment notebook. Contains the buildup from simple facts to the "Middle Void" ablation tests.
- **`02_residual_stream_decomposition.ipynb`**: Mathematical analysis using Cosine Similarity to prove decision points.
- **Visuals**: `trajectory_facts.png`, `cosine_similarity.png`, and `attention_heatmap.png` contain the latest Llama 3 evidence.

## Future Directions
- **Reasoning Limits:** Investigate why "multi-hop" prompts (like the "Sunday" logic prompt) fail when the Middle Void is applied.
- **Head-Level Ablation:** Moving from removing entire layers to removing specific Attention vs. MLP heads within the Critical Path.
- **Cross-Model Validation:** Comparing these Llama 3 "Void" results against Llama 3.1 or 3.2.
- **Sparse Autoencoders (SAEs):** Investigate training small SAEs on the residual stream of the Middle Void (Layer 3) to see if we can identify "pre-retrieval" features.
- **Path Patching:** Trace the activation of factual knowledge from the early layers, through the Middle Void, to the final logit sharpening.

## Advanced Research Tracks (Pending)

### 1. The "Ablated Logit Lens" (Cross-Project)
- **Goal:** Apply the Logit Lens to a model where the "Middle Void" (Layers 3-21) has been weight-shifted by 500k.
- **Question:** Does the model's internal "thought" scramble in the middle and then "self-correct" as it hits the clean final layers (30-31)?
- **Insight:** Visualize the Error Correction capabilities of the Transformer architecture in real-time.

