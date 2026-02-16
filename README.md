# Project: Bulgarian → English Neural Machine Translation (generative LM)

## Overview

This repository contains a course project implementing a generative language model for neural machine translation (Bulgarian → English). The model is trained as a conditional generative language model: each training sequence is the source sentence (Bulgarian), followed by a special separator token, followed by the target sentence (English). At inference time the model receives a Bulgarian sentence + separator and generates the English translation as continuation.

The intended audience is familiar with Transformer-style architectures and common deep learning terminology.

## Model architecture

The model is based on the Transformer design from Attention Is All You Need, but implemented using only the decoder stack because the task is formulated as conditional generation (generate continuation), not sequence-to-sequence encoding-only inference.

Best model configuration used in experiments:

* **Input pipeline**

  * Token-level segmentation (word-level tokens; see Tokenization section).
  * Token embedding + positional embedding.
  * Dropout after embeddings for regularization.

* **Transformer decoder stack**

  * 8 identical decoder blocks.
  * Each block: multi-head self-attention → add & norm (residual + LayerNorm) → feed-forward network (FFN) → add & norm.
  * Multi-head attention implementation optimizations:

    1. Single linear projection to compute Q/K/V concatenated, then split into three matrices. This reduces overhead by replacing three small matmuls with one larger matmul (better utilization on modern BLAS/cuBLAS).
    2. Use of `torch.nn.functional.scaled_dot_product_attention` for the attention kernel to leverage the optimized attention implementation (FlashAttention-style speedups) available in modern runtimes and GPUs. PyTorch's API is used for this function.
  * Nonlinearity in the FFN: **GELU** (as used in GPT-2 family), rather than ReLU from the original Transformer paper.

* **Output**

  * Final linear projection to vocabulary logits.
  * Weight tying between the input token embeddings and the final linear projection was used in several experiments: tying reduces total parameters significantly while leaving perplexity and BLEU roughly unchanged in our tests.

## Tokenization

* Primary tokenization: **word-level tokens** (vocabulary built from words with frequency thresholds; several thresholds were tried).
* Byte Pair Encoding (BPE / subword) was tested (tokenizer sizes: 25k, 40k, 51.2k tokens), but **word-level tokenization produced better BLEU scores** in our experiments (BPE models were ~10 BLEU points worse on the evaluation set).

## Hyperparameters (model configuration)

* Embedding dimension: **256**
* Context length (max sequence length): **256**
* Number of attention heads: **8**
* Dimension per head: **32** (so total model width = 8 × 32 = 256)
* Number of decoder layers: **8**
* Dropout probability: **0.1**

## Training details

* Loss: cross-entropy (standard next-token language-modeling objective conditioned on source+separator).
* Optimizer: **AdamW**, `weight_decay=0.01`.
* Gradient clipping: `torch.nn.utils.clip_grad_norm_` with `max_norm=3`.
* Batch size: **16**.

### Learning rate schedule

A cosine-decay schedule with warmup was used:

* **Warmup**: first **3,000 steps** — linear warmup up to the peak learning rate.
* **Peak LR**: `2e-3`.
* After warmup, **cosine decay** over the next **200,000 steps** down to a minimum learning rate of `1e-4`.
* After reaching `1e-4` the LR is kept constant for the remainder of training.

(The schedule is the common warmup + cosine decay pattern used in transformer training literature.)

## Results — experiments and comparisons

Multiple tokenization strategies and vocabulary sizes were evaluated. The table below summarizes the main comparisons. Columns: model description | perplexity | BLEU | approximate number of parameters.

| Model / tokenization (notes)                                                                     | Perplexity |      BLEU | Parameters |
| ------------------------------------------------------------------------------------------------ | ---------: | --------: | ---------: |
| Word-level tokens (keep words with >3 occurrences); **weight-tied** embeddings/output projection |  **11.19** | **40.61** |    **18M** |
| Word-level tokens (keep words with >5 occurrences); weight-tied                                  |      10.84 |     40.49 |        16M |
| Word-level tokens (keep words with >3 occurrences); **no weight tying**                          |      11.93 |     40.04 |        31M |
| Word-level tokens (keep words with >10 occurrences)                                              |      10.35 |     39.33 |        13M |
| Word-level tokens (keep words with >2 occurrences)                                               |      12.01 |     39.21 |        36M |
| BPE tokenization — 50k tokens (no weight tying)                                                  |      10.26 |     29.08 |        33M |
| BPE tokenization — 50k tokens, weight tied                                                       |      10.92 |     28.02 |        19M |
| BPE tokenization — 40k tokens                                                                    |      11.61 |     26.04 |        27M |
| BPE tokenization — 25k tokens (embedding dim = 512)                                              |      11.81 |     22.03 |        38M |

**Key observations**

* Word-level tokenization consistently outperformed BPE on BLEU for this dataset / setup.
* Weight tying (sharing input embeddings and output projection weights) reduced model size significantly with negligible impact on perplexity and BLEU in our experiments — a useful tradeoff for parameter budget.
* Perplexity and BLEU do not always move together: some configurations had slightly better perplexity but worse BLEU; BLEU was prioritized for model selection because translation quality (BLEU) is the primary task metric.

## Notes and future work

* The current decoder-only framing (source+separator → target continuation) makes the model compatible with standard autoregressive generation APIs and simplifies sampling/beam search at inference.
* The relative underperformance of BPE in these experiments may be dataset-dependent (e.g., morphology and token frequency patterns in Bulgarian). Future work: try hybrid schemes (e.g., morpheme-aware tokenization) and larger (or multilingual) training corpora.
* Additional architecture experiments that could be explored: larger model widths/depths, different FFN sizes, alternative attention kernels, or incorporating encoder information explicitly (encoder–decoder Transformer) for comparison.

## References

* The Transformer architecture: Attention Is All You Need.
* Optimized attention kernels (FlashAttention-style): implemented via the attention primitives available in modern runtimes / libraries.
* Training code uses `torch.nn.functional.scaled_dot_product_attention` and other utilities from PyTorch.