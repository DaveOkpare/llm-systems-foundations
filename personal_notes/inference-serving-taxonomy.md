# Taxonomy of the LLM Serving Layer (Inference Systems Optimization)

Organized around the problem each category solves. These are orthogonal to
sampling strategies (greedy, temperature, top-k, top-p) — they don't change
*what* token gets picked, only *how efficiently* the model runs.

---

## 1. Memory / Cache Layer

**Problem:** KV cache dominates GPU memory (often 30–80%) and grows linearly
with sequence length × batch size. Running out = OOM crashes or tiny batches.

- **PagedAttention** (vLLM) — treats KV cache like virtual memory. Splits it
  into fixed-size "pages" so requests of different lengths don't waste memory
  to fragmentation. Enabled 2–4× throughput jumps.
- **Prefix caching / RadixAttention** (SGLang) — stores KV of shared prompts
  in a radix tree; new requests match prefixes and skip recomputation. Huge
  win for chat apps with long system prompts.
- **KV cache quantization** — store KV in int8/int4/fp8 instead of fp16.
  ~2–4× memory savings with minor quality loss.
- **KV cache offloading** — spill cold cache to CPU RAM or NVMe (e.g.,
  DeepSpeed-FastGen, FlexGen). Trades latency for capacity.
- **Cache eviction / compression** — H2O, StreamingLLM, SnapKV drop or
  compress "unimportant" tokens' KV.

---

## 2. Scheduling

**Problem:** Requests arrive asynchronously with wildly different prompt and
output lengths. Naive batching wastes GPU cycles.

- **Continuous (in-flight) batching** — Orca's invention. Instead of waiting
  for a whole batch to finish, finished sequences drop out mid-batch and new
  ones slot in. Core technique in every modern engine.
- **Chunked prefill** — break long prefills into chunks and interleave with
  decode steps, preventing long prompts from starving ongoing generations.
- **Prefill-decode disaggregation** — DistServe, Mooncake, Splitwise.
  Separate GPU pools for the two phases; transfer KV between them.
- **SLO-aware scheduling** — prioritize requests by deadline (TTFT, TPOT
  targets) rather than FIFO.
- **Speculative scheduling** — predict output length to pack batches better.

---

## 3. Parallelism

**Problem:** Models don't fit on one GPU, or fit but run too slowly.

- **Tensor parallelism (TP)** — split individual matmuls across GPUs
  (Megatron-style). Low latency, high communication. Used within a node
  (NVLink).
- **Pipeline parallelism (PP)** — split layers across GPUs. Lower bandwidth
  need, but adds pipeline bubbles. Good across nodes.
- **Expert parallelism (EP)** — for MoE models, shard experts across GPUs.
  DeepSeek-V3 and Mixtral rely heavily on this.
- **Sequence / context parallelism** — split long sequences across GPUs
  (Ring Attention, Ulysses). Essential for 1M+ context.
- **Data parallelism (DP)** — replicate the model, split requests. Trivial
  at inference; mostly a training concept.

These compose: a frontier MoE might run TP×EP within a node, PP across nodes.

---

## 4. Compute-Level Optimizations

**Problem:** Even with memory/scheduling fixed, the raw math can be faster.

- **Speculative decoding** — a small "draft" model proposes tokens; the big
  model verifies them in parallel. 2–3× decode speedup for free. Variants:
  Medusa, EAGLE, lookahead decoding, self-speculative.
- **FlashAttention / FlashAttention-2/3** — fused attention kernel that
  avoids materializing the N×N attention matrix. Standard now.
- **Quantization** — INT8, FP8, INT4 (GPTQ, AWQ), even INT2. Smaller weights
  → faster memory-bound decode. FP8 is the current sweet spot on H100/H200.
- **Kernel fusion** — combine ops (e.g., RMSNorm + matmul) to reduce memory
  round trips. Handled by compilers like TensorRT, torch.compile.
- **Custom kernels** — hand-written CUDA/Triton for specific shapes
  (decoding attention, MoE routing).

---

## 5. Routing / Orchestration

**Problem:** One engine instance isn't enough; you need a fleet.

- **Load balancing** — distribute requests across replicas. Prefix-aware
  routing (send requests with the same system prompt to the same replica to
  hit the cache) is a big win.
- **Model routing / cascades** — route easy queries to small models, hard
  ones to big ones (FrugalGPT-style).
- **Multi-tenancy** — LoRA hot-swapping (S-LoRA, Punica) serves many
  fine-tuned models from one base.
- **Autoscaling** — scale replicas based on queue depth, TTFT SLO, or token
  throughput.

---

## How it fits together in practice

A modern production stack (e.g., vLLM + a gateway):

```
Request → Router (prefix-aware LB)
       → Engine replica
         ├─ Scheduler (continuous batching, chunked prefill)
         ├─ Memory (PagedAttention + prefix cache)
         ├─ Parallelism (TP across 8 GPUs)
         └─ Kernels (FlashAttention, FP8 matmul)
       → Streaming tokens back
```

---

## Anchor papers per layer

- **PagedAttention** — *Efficient Memory Management for LLM Serving*
  (vLLM, 2023)
- **Continuous batching** — *Orca: A Distributed Serving System* (OSDI 2022)
- **Disaggregation** — *DistServe* (OSDI 2024), *Mooncake* (2024)
- **Speculative decoding** — *Fast Inference from Transformers via
  Speculative Decoding* (Leviathan et al., 2023)
- **RadixAttention** — *SGLang* (2024)
- **FlashAttention** — Dao et al. (2022, 2023, 2024)
