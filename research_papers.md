# 📚 Research Papers: Mechanistic Interpretability

These papers provide the framework for looking inside the "black box" of LLMs, identifying internal circuits, and decomposing complex representations.

### Core Papers

*   **[In-context Learning and Induction Heads](https://arxiv.org/abs/2209.11895)** (Olsson et al., 2022)
    *   **Relevance:** The foundation for identifying the physical "induction circuits" (Induction Heads) that allow models to copy-paste information from context.
    *   **Key Concept:** Two-layer attention heads that form "induction" behaviors.

*   **[Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small](https://arxiv.org/abs/2211.00593)** (Wang et al., 2022)
    *   **Relevance:** The definitive methodology for mapping complex logic circuits, specifically the **IOI Circuit**, and the use of **Path Patching** as a tracing tool.
    *   **Key Concept:** Structural mapping of multi-head ensembles performing logical tasks.

*   **[Towards Monosemanticity: Decomposing Language Models With Sparse Autoencoders](https://transformer-circuits.pub/2023/monosemantic-features/index.html)** (Bricken et al., 2023)
    *   **Relevance:** The core research on **Sparse Autoencoders (SAEs)**, used in this project's "Middle Void" experiments to disentangle polysemantic features in the residual stream.
    *   **Key Concept:** Overcomplete autoencoders as a way to find "monosemantic neurons."

*   **[Logit Lens](https://www.lesswrong.com/posts/AcKRB8wLmcNfeyWfn/interpreting-gpt-the-logit-lens)** (nostalgebraist, 2020)
    *   **Relevance:** A foundational technique for visualizing the "intermediate thoughts" of a transformer across layers.
    *   **Key Concept:** Decoding hidden states into vocabulary space using the pre-existing unembedding head (`lm_head`).

---
*Generated for the Mechanistic Interpretability context.*
