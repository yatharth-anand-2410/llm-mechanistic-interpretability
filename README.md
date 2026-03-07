# Mechanistic Interpretability of LLMs

**Goal:** Look inside the "black box" of the LLM by analyzing its internal activations during the forward pass. This repository contains experiments on `mlx-community/Meta-Llama-3-8B-Instruct-4bit`.

---

## Phase 1: Model Instrumentation (Notebook 0)

To perform mechanistic interpretability on an Apple Silicon Mac using MLX, we built custom infrastructure to extract the internal thoughts of the model.

### 1. Extracting Hidden States
Instead of relying on `output_hidden_states=True`, we manually iterate through the 32 Transformer layers of Llama 3 8B.
```python
# Pass through transformer block
h = layer(h, mask, None)
hidden_states[i] = h
```
**CRITICAL FINDING:** For sequences longer than 1 token, you *must* pass the `causal_mask` explicitly into each layer. If omitted, tokens will illegally attend to future tokens during the manual extraction, causing a total mathematical divergence from the ground truth forward pass.

### 2. The Logit Lens
The Logit Lens is a mathematical trick used to decode the "intermediate thoughts" of the model.

**The Math:**
In Llama 3, the embedding matrix ($W_E$) turns tokens into 4096-dimensional vectors. At the end of the network, the unembedding head (`lm_head`) turns the final vector back into a probability distribution.
Normally: `Input -> W_E -> Layer 1 -> ... -> Layer 32 -> lm_head -> Output`

The Logit Lens takes a shortcut:
`Input -> W_E -> Layer 1 -> lm_head -> Output`

This works because of the **Residual Stream**. Every layer adds its output on top of a single continuous "conveyor belt" vector (`x_new = x_old + layer_output`). Therefore, vectors at Layer 16 are already in the same mathematical "language" as the final layer, allowing us to peak at the model's progress.

---

## Phase 2: Knowledge Emergence & Structural Limits (Notebook 1)

This phase focuses on **intervention**. By removing layers and measuring the "breaking point" of different tasks, we map the model's functional topology.

### 1. Task-Specific Emergence Trajectories
We found that the network does not treat all tasks with the same priority. High-level facts are retrieved at different depths.

| Prompt Type | Example | Emergence Layer | Profile Type |
| :--- | :--- | :--- | :--- |
| **Factual Recall** | `The capital of France is` | **Layer 19** | Step-Function (Sharp) |
| **Arithmetic** | `2+2=` | **Layer 23** | Oscillatory / Syntactic |
| **Relational Logic**| `The opposite of hot is...`| **Layer 25** | Distributed / Gradual |

![Factual vs Arithmetic Trajectory](trajectory_facts.png)
*Figure 1: Comparison of emergence points. Note the sharp spike for facts vs. the later stability for arithmetic.*

---

### 2. The "Critical Path" (Redundancy Analysis)
We tested the model's resilience by removing one layer at a time. The results identified a narrow **Critical Path** surrounded by massive **Functional Redundancy**.

| Layer Range | Role | Status | Consequence of Removal |
| :--- | :--- | :--- | :--- |
| **0 - 1** | **Context Initialization** | **CRITICAL** | Total output collapse (Garbage tokens) |
| **2 - 29** | **Iterative Refinement** | **REDUNDANT** | Near-zero impact on final accuracy |
| **30** | **Logit Sharpening** | **CRITICAL** | Prediction "smears" (e.g., ' Paris' becomes ' the') |
| **31** | **Final Output** | **CRITICAL** | No prediction possible |

---

### 3. The "Middle Void" Theory
Our most significant discovery: For shallow tasks, **60% of the model is mathematically optional.**

We discovered a contiguous window from **Layer 3 to Layer 21** that can be entirely deleted while the model still correctly predicts "Paris" with high confidence.

| Task | Total Layers | Max Removable Window | % Optional |
| :--- | :--- | :--- | :--- |
| **Factual Recall** | 32 | **15 Layers** (5-19) | 46.8% |
| **Arithmetic** | 32 | **19 Layers** (3-21) | 59.3% |
| **Relational Logic**| 32 | **10 Layers** (20-29) | 31.2% |

**Mechanistic Conclusion:**
The "Middle Void" proves that the middle layers act as a **mathematical conveyor belt**. If the input vector is already "correct enough" by Layer 2, it can coast through the void and still trigger the correct factual lookup heads at Layer 22.

---

## Phase 3: Mathematical Decomposition & Fragility Mapping (Notebook 2)

Phase 3 bridges interpretability with **Weight Surgery** (Phase 1) and **Decoding Strategies** (Phase 2) to find the model's structural bottlenecks.

### 1. Dual-Axis Correlation Analysis
By overlaying **Target Probability** with **Cosine Similarity (Transformation Rate)**, we proved that factual spikes correlate with specific mathematical "Decision Points."

| Metric | perimeter (L1-3) | middle (L10-28) | final (L31) |
| :--- | :--- | :--- | :--- |
| **Cosine Similarity** | **Low (~0.6)** | **High (>0.9)** | **Very Low (~0.45)** |
| **Primary Workload** | Context Encoding | Incremental Refinement | Logit Sharpening |

### 2. The "Fragility Sandwich" Theory
We performed a **Triple-Prompt Functional Scan** to find the exact bit-level noise tolerance (Integer Tipping Point) for every layer.

| Layer Index | Factual Shift Limit | Reasoning Shift Limit | Arithmetic Shift Limit |
| :--- | :--- | :--- | :--- |
| **Layer 1 (Logic)** | 1,328,126 | 840,210 | 1,105,400 |
| **Layer 15 (Void)** | >10,000,000 | >10,000,000 | >10,000,000 |
| **Layer 32 (LM_Head)** | 1,000,005 | 1,000,005 | 1,000,005 |

**Key Breakthroughs:**
1.  **Internal logic is more fragile than the interface:** Layers 1 and 31 are often more sensitive to noise than the `lm_head` itself.
2.  **Reasoning Tension:** Relational reasoning tasks place the weights under higher "mathematical tension," resulting in lower tipping points in early-middle layers compared to factual lookups.
3.  **Recursive Collapse:** Weight surgery on the `lm_head` does not destroy knowledge in a single pass; it triggers a **Recursive Failure** as garbage tokens are fed back into the model during autoregressive generation.

### 3. Mechanistic "Self-Healing"
We analyzed how **Contrastive Search** rescues corrupted models. By checking the $L_2$ Norm of embeddings, Contrastive Search acts as a **Structural Filter**, leveraging the healthy internal residual stream to override the noisy, corrupted outputs of the damaged `lm_head`.

---

## Visual Evidence
*   **`trajectory_facts.png`**: Visual proof of the Layer 19 knowledge spike.
*   **`cosine_similarity.png`**: The mathematical signature of structural transformation.
*   **`fragility_map.png`**: Log-scale proof of the "Fragility Sandwich" (Low limit at perimeters, High limit in middle).
