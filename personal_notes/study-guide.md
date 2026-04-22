# LLM Inference Serving & Systems Optimization — Study Guide

A curated reading list for a Python backend engineer ramping into GPU/ML
systems. Sources skew toward canonical papers, production engineering
blogs, and recent (2023–2026) material.

---

## 1. Foundations (transformer internals, attention math)

- **Attention Is All You Need** — Vaswani et al., NeurIPS 2017. Still the
  canonical transformer reference; read it once to anchor notation
  (Q/K/V, multi-head, positional encodings) before touching serving papers.
  https://arxiv.org/abs/1706.03762

- **The Illustrated Transformer** / **The Annotated Transformer** — Jay
  Alammar (2018) & Sasha Rush (Harvard NLP, updated 2022). Pair the two:
  Alammar for intuition, the Annotated version for line-by-line PyTorch.
  - https://jalammar.github.io/illustrated-transformer/
  - https://nlp.seas.harvard.edu/annotated-transformer/

- **Stanford CS336: Language Modeling from Scratch** (Spring 2025, Percy
  Liang & Tatsu Hashimoto). Builds tokenizer, model, kernels, and training
  stack from zero; lectures on YouTube plus public assignments. The single
  best curriculum-style entry point as of 2026.
  https://cs336.stanford.edu/

---

## 2. KV cache & memory management

- **Efficient Memory Management for LLM Serving with PagedAttention** —
  Kwon et al., SOSP 2023 (the vLLM paper). The foundational paper for
  block-based KV cache. Read before vLLM internals.
  https://arxiv.org/abs/2309.06180

- **Fast and Expressive LLM Inference with RadixAttention and SGLang** —
  Zheng et al., NeurIPS 2024 + LMSYS blog (Jan 2024). Canonical reference
  for automatic prefix caching via a radix tree; pairs with the vLLM paper.
  - https://arxiv.org/abs/2312.07104
  - https://www.lmsys.org/blog/2024-01-17-sglang/

- **Optimizing AI Inference at Character.AI (Parts I & II)** —
  Character.AI engineering blog (Jun 2024 / Aug 2025). Rare production
  write-up on >20× KV cache reduction (MQA, cross-layer sharing, int8
  attention). Closest thing to a "real-world war story" for KV management.
  - https://blog.character.ai/optimizing-ai-inference-at-character-ai/
  - https://blog.character.ai/optimizing-ai-inference-at-character-ai-part-deux-2/

---

## 3. Scheduling (continuous batching, chunked prefill, disaggregation)

- **Orca: A Distributed Serving System for Transformer-Based Generative
  Models** — Yu et al., OSDI 2022. The origin of iteration-level
  (continuous) batching and selective batching. Still the reference.
  https://www.usenix.org/conference/osdi22/presentation/yu

- **How continuous batching enables 23x throughput in LLM inference** —
  Cade Daniel et al., Anyscale (Jun 2023). Accessible engineer-oriented
  explainer with benchmarks; the standard link to send teammates.
  https://www.anyscale.com/blog/continuous-batching-llm-inference

- **Taming Throughput-Latency Tradeoff in LLM Inference with
  Sarathi-Serve** — Agrawal et al., OSDI 2024. Canonical chunked-prefill +
  stall-free scheduling paper. Supersedes the 2023 SARATHI preprint.
  https://www.usenix.org/conference/osdi24/presentation/agrawal

- **DistServe: Disaggregating Prefill and Decoding for Goodput-Optimized
  LLM Serving** — Zhong et al., OSDI 2024. The reference paper for
  prefill/decode disaggregation; read alongside Microsoft's concurrent
  **Splitwise** (ISCA 2024) for the heterogeneous-hardware angle.
  - https://arxiv.org/abs/2401.09670

---

## 4. Parallelism (TP / PP / EP / SP)

- **Megatron-LM: Training Multi-Billion Parameter Language Models Using
  Model Parallelism** — Shoeybi et al., 2019, and the 2021 follow-up
  **Efficient Large-Scale LM Training on GPU Clusters (PTD-P)** — Narayanan
  et al., SC '21. Together these define tensor parallelism and how to
  combine TP/PP/DP. Older but still canonical.
  https://arxiv.org/abs/1909.08053

- **How to Scale Your Model** — Google DeepMind / JAX team (2024, updated
  2025). Free online book that walks through TP, PP, sequence parallelism,
  expert parallelism with arithmetic intensity / roofline reasoning. The
  best modern replacement for reading the Megatron papers cold.
  https://jax-ml.github.io/scaling-book/

---

## 5. Compute optimizations (kernels, quantization, speculative decoding)

- **FlashAttention-3: Fast and Accurate Attention with Asynchrony and
  Low-Precision** — Shah, Bikshandi, Dao et al., NeurIPS 2024. Current
  canonical attention kernel on Hopper; cite FA-2 only for pre-H100 work.
  *Flag: FlashAttention v1 (2022) and v2 (2023) are now superseded for H100+.*
  https://arxiv.org/abs/2407.08608

- **Making Deep Learning Go Brrrr from First Principles** — Horace He
  (2022). Compute vs. memory-bandwidth vs. overhead mental model; the
  shortest path to reasoning about GPU perf. Still the right first read.
  https://horace.io/brrr_intro.html

- **Large Transformer Model Inference Optimization** — Lilian Weng, Lil'Log
  (Jan 2023). Compact survey of quantization, distillation, sparsity, and
  speculative decoding — high signal per word.
  https://lilianweng.github.io/posts/2023-01-10-inference-optimization/

- **Fast Inference from Transformers via Speculative Decoding** — Leviathan
  et al., ICML 2023 (concurrent with Chen et al., DeepMind). The two
  foundational speculative decoding papers. For the state of the art, read
  the **Comprehensive Survey of Speculative Decoding** (Xia et al., ACL
  Findings 2024).
  - https://arxiv.org/abs/2211.17192
  - https://aclanthology.org/2024.findings-acl.456.pdf

- **gpt-fast** — PyTorch team / Horace He (2023). ~1000 lines of PyTorch
  implementing int8/int4 quant, speculative decoding, and TP. Best "read
  the code" companion to the papers above.
  https://github.com/pytorch-labs/gpt-fast

---

## 6. Serving systems & production

- **vLLM documentation & source** — https://docs.vllm.ai/ and
  https://github.com/vllm-project/vllm. De facto reference implementation;
  the scheduler and block manager are worth reading directly.

- **NVIDIA TensorRT-LLM Performance Tuning Guide** —
  https://nvidia.github.io/TensorRT-LLM/ and the **TensorRT-LLM Performance
  Benchmarking** technical blog (2025). Essential for FP8/NVFP4, in-flight
  batching, and disaggregated serving on NVIDIA hardware.
  https://developer.nvidia.com/blog/llm-inference-benchmarking-performance-tuning-with-tensorrt-llm

- **LLM Inference Serving: Survey of Recent Advances and Opportunities** —
  Li et al., 2024 (updated 2025). Taxonomy of ~50 serving systems/papers
  from ASPLOS/MLSys/OSDI; useful as a map when you hit unfamiliar terms.
  https://arxiv.org/html/2407.12391

- **What is inference engineering? A deep dive** — Gergely Orosz,
  *The Pragmatic Engineer* (2024). Practitioner-oriented overview of the
  role, stack, and team shapes — good for framing production concerns end
  to end.
  https://newsletter.pragmaticengineer.com/p/what-is-inference-engineering

---

## 7. Books & curriculum-style resources

- **AI Engineering: Building Applications with Foundation Models** — Chip
  Huyen, O'Reilly (Jan 2025). Strong production framing: evaluation,
  observability, cost, latency budgets. Complements the serving papers
  with the "what do I measure?" layer.
  https://www.oreilly.com/library/view/ai-engineering/9781098166298/

- **Designing Machine Learning Systems** — Chip Huyen, O'Reilly (2022).
  Older but still the best single book for ML-in-production mental models
  (data, pipelines, monitoring, CI/CD). *Flag: pre-LLM — skip the modeling
  chapters, keep the systems chapters.*
  https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/

- **Build a Large Language Model (From Scratch)** — Sebastian Raschka,
  Manning (2024). PyTorch implementation from tokenizer to training loop;
  useful if CS336 is too dense.
  https://www.manning.com/books/build-a-large-language-model-from-scratch

- **LLMs in Production** — Brousseau, Sharp & Baylor, Manning (2025).
  Practitioner-focused: GPU management, model compilation, adaptive
  batching, k8s deployment patterns. The closest thing to an "LLM SRE
  handbook" as of 2026.
  https://www.manning.com/books/llms-in-production

- **GPU MODE** lecture series (YouTube + Discord, 2024–2026, curated by
  Mark Saroufim & Andreas Koepf). Free weekly talks on CUDA, Triton,
  FlashAttention internals, quantization; the de facto community
  classroom for GPU kernel work.
  https://github.com/gpu-mode/lectures

---

## Suggested reading order for a Python backend engineer

1. Vaswani → Alammar → He's "Brrrr" post → Weng's inference-optimization
   post (~1 week, establishes vocabulary).
2. Orca → vLLM/PagedAttention → Anyscale continuous-batching post →
   Sarathi-Serve → DistServe (~2 weeks, the scheduling core).
3. FlashAttention-3 + gpt-fast code read → speculative decoding survey →
   "How to Scale Your Model" book (~2 weeks, compute & parallelism).
4. vLLM source dive + TensorRT-LLM tuning guide + Character.AI posts
   (~ongoing, production reality).
5. Use Huyen's *AI Engineering* and Manning's *LLMs in Production* as
   reference texts alongside, not linearly.

**Flagged-as-dated:** FlashAttention v1/v2 (use FA-3), original SARATHI
preprint (use Sarathi-Serve OSDI '24), Huyen's *Designing ML Systems*
modeling chapters (pre-LLM).
