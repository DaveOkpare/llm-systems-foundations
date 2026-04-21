# LLM Inference Serving — Buildable Projects & OSS Contribution Targets

Companion to `study-guide.md`. Two sections: projects to **build yourself**
(ordered foundations → full serving stack) and **open-source projects** to
read/PR against (activity verified April 2026). All assume Python/PyTorch
as the primary language; CUDA/Triton appears only in the latter half of
the build list.

---

## 1. Build-your-own projects (ordered by difficulty)

### 1. KV cache from scratch on a small GPT — *weekend*
Take nanoGPT or `microgpt` (Karpathy's 200-line Feb 2026 release) and add
a per-layer, per-head KV cache, then toggle it on/off to measure the O(n)
vs O(n²) decode speedup. **You'll learn** the mechanics of prefill vs
decode, tensor shapes `[batch, heads, seq, head_dim]`, and why caching is
table-stakes. Reference: Sebastian Raschka's
[coding-the-kv-cache](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms)
walkthrough.

### 2. Sampling zoo + logit processors — *weekend*
Implement greedy, temperature, top-k, top-p, min-p, repetition penalty,
and a logit-bias hook as composable `LogitsProcessor` classes. Add a
streaming generator that yields tokens. **You'll learn** the inference
loop plumbing every engine re-implements. Reference against HuggingFace
`transformers/generation/logits_process.py` and vLLM's `SamplingParams`.

### 3. Naive continuous batcher on a single GPU — *1 week*
Write a scheduler that accepts requests on an `asyncio.Queue`, pads them
into a batch tensor, does one decode step, evicts EOS'd sequences, and
admits new ones — the Orca pattern. Measure tokens/sec vs static batching
at varying request arrival rates. **You'll learn** why continuous
batching is the single biggest throughput win. Reference: Anyscale's
[continuous batching post](https://www.anyscale.com/blog/continuous-batching-llm-inference)
and vLLM's `LLMEngine.step()`.

### 4. PagedAttention in pure PyTorch — *1 week*
Build a block allocator (e.g. 16 tokens/block), a block table mapping
logical → physical blocks per sequence, and a gather-based attention
kernel that reads non-contiguous blocks. Skip CUDA; use `torch.gather` /
FlexAttention. Demonstrate zero fragmentation and copy-on-write prefix
sharing. **Compare against**
[nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) (~1,200 LOC).

### 5. Prefix / radix cache for shared system prompts — *1 week*
Extend project 4 with a trie (radix tree) keyed by token-id sequences so
that many requests sharing a system prompt re-use the same physical KV
blocks. Benchmark hit rate on a chat dataset. **You'll learn**
RadixAttention — the key idea behind SGLang. Reference: SGLang's
`RadixCache` in `python/sglang/srt/mem_cache/`.

### 6. Speculative decoding driver — *1 week*
Pair a small draft model (e.g. SmolLM2-135M) with a target
(Llama-3-8B or Qwen3-1.7B). Implement multi-step drafting, parallel
verification in one target forward pass, and the Leviathan et al.
rejection-sampling acceptance rule. Measure mean acceptance length and
end-to-end speedup. Reference:
[PyTorch's guide](https://pytorch.org/blog/hitchhikers-guide-speculative-decoding/)
and gpt-fast's `generate.py`.

### 7. Chunked prefill scheduler (Sarathi-Serve) — *1 week*
On top of project 3, split long prompts into fixed-size chunks
(e.g. 512 tokens) and interleave prefill chunks with decode steps in the
same batch. Plot TTFT vs throughput tradeoffs vs pure prefill-priority
and pure decode-priority. **You'll learn** the stall-free scheduling
regime used by every modern engine. Reference: vLLM's V1 scheduler and
SGLang's `schedule_batch.py`.

### 8. KV quantization — *weekend / 1 week*
Quantize the V cache to int8 (per-token) and K to int8 or FP8, with
dequant-on-gather. Measure memory saved, perplexity delta on WikiText,
and throughput change. Extend to groupwise int4 for an ambitious version.
Reference: vLLM's `kv_cache_dtype="fp8"` path and KIVI paper impls.

### 9. OpenAI-compatible HTTP server with streaming — *1 week*
Wrap your engine in FastAPI with `/v1/chat/completions` (SSE streaming),
`/v1/completions`, token usage accounting, and cancellation on client
disconnect. Add Prometheus metrics (TTFT, inter-token latency, queue
depth, running/waiting requests). **You'll learn** the production
surface area — this is 70% of what "serving engine" means day-to-day.
Reference: vLLM's `api_server.py`.

### 10. Tensor parallelism on 2 GPUs — *1 week to 1 month*
Shard a Llama-style model's attention (column-split QKV, row-split O) and
MLP (column-split up/gate, row-split down) across 2 GPUs using
`torch.distributed` all-reduces. Get it working without any parallelism
library, then compare to `torch.distributed.tensor` / DTensor.
**You'll learn** Megatron-style TP from the inside. Reference:
[gpt-fast](https://github.com/pytorch-labs/gpt-fast)'s `tp.py` and
[torchtitan](https://github.com/pytorch/torchtitan).

### 11. Disaggregated prefill/decode proof of concept — *1 month*
Run two engine workers — one prefill-only, one decode-only — and ship
the KV cache between them over NCCL or a `zmq` socket. Measure TTFT vs
colocated. Follow the DistServe paper's split. Reference: vLLM's
disaggregated serving mode and SGLang's PD disaggregation.

### 12. FlashAttention-style Triton kernel — *1 month, ambitious*
Port a tiled, online-softmax attention kernel to Triton for the decode
path (q_len=1). Get within 2–3× of the official kernel and integrate it
into project 4. This is the first project that truly requires
kernel-level thinking. Reference: Triton's `06-fused-attention.py`
tutorial and the [FlashAttention](https://github.com/Dao-AILab/flash-attention) repo.

### 13. MoE router and expert parallelism — *1 month, ambitious*
Implement a top-k gate, token dispatch/combine, and a 2-GPU
expert-parallel split for a Mixtral-style model. Measure load imbalance
and add a capacity factor. Reference: gpt-fast's Mixtral branch and
torchtitan's DeepEP integration.

### 14. Kubernetes LeaderWorkerSet deployment — *1 week*
Take your engine from project 9, containerize it, and deploy with an
LWS (multi-host TP), a Service, an HPA on queue depth, and a Gateway
with a request router that does prefix-affinity routing. Reference:
[llm-d](https://github.com/llm-d/llm-d) and `vllm-project/production-stack`.

### 15. End-to-end benchmark harness — *weekend, bolt onto anything above*
Drive your engine with the `ShareGPT` trace at a Poisson arrival rate,
compute TTFT p50/p99, ITL p50/p99, goodput at an SLO, and
throughput-at-SLO curves. Reference:
[vllm-project/guidellm](https://github.com/vllm-project/guidellm) and
SGLang's `bench_serving.py`.

---

## 2. Open-source projects to read and/or contribute to

Activity verified April 2026. Star ranges approximate.

### vLLM — https://github.com/vllm-project/vllm — ~50k stars
**Activity:** extremely active — 448 commits / 197 contributors in one
recent release; PyTorch Foundation project since 2025; Q1 and Q2 2026
roadmaps published. **PRs newcomers land:** new model architectures
(`model_executor/models/`), sampling params, logits processors, test
coverage, metrics/Prometheus, API-compat quirks for OpenAI endpoints,
docs. **Easy wins:** pick a trending Hugging Face model vLLM doesn't
support and port it using an existing model as template; add a missing
sampler config field the OpenAI client passes through. Watch the `v1`
label and SIG-Core meetings.

### SGLang — https://github.com/sgl-project/sglang — ~15k stars
**Activity:** very active, 2026 Q1 roadmap in issue #12780, Model
Gateway v0.3.0 in Feb 2026. **PRs newcomers land:** frontend language
primitives, model ports, tokenizer edge cases, RadixCache improvements,
tests for speculative decoding. LMSYS sponsors long-term contributors
via a coding-agent program (email sglang@lmsys.org with your best PRs).
**Easy wins:** add a missing chat template; fix a `schedule_batch` bug
reproducible from a failing test; extend the benchmark harness.

### llama.cpp — https://github.com/ggml-org/llama.cpp — ~70k stars
**Activity:** daily releases (b8838 on April 18 2026). Primarily C/C++,
but the Python conversion scripts (`convert_hf_to_gguf.py`) are a
Python-friendly on-ramp. **PRs newcomers land:** HF-to-GGUF conversion
for new architectures, server OpenAI-compat endpoints, grammar/JSON
schema fixes, Python bindings. **Easy wins:** add GGUF conversion for a
new small model; fix a sampler parameter not threaded through the server.

### gpt-fast — https://github.com/pytorch-labs/gpt-fast
**Activity:** low volume by design — it's a **reference implementation**
(~1000 LOC, int4/int8 quant, speculative decoding, TP, Mixtral), not a
framework. Read it end-to-end; it's the best Python+PyTorch
demonstration of compile+quant+spec on a GPU. Use it as a **study
target**, not a primary contribution target.

### torchtitan — https://github.com/pytorch/torchtitan — ~4k stars
**Activity:** very active (last update April 20 2026); training-focused
but the parallelism primitives (FSDP2, TP, PP, EP, Context Parallelism)
are the same ones serving engines need. **PRs newcomers land:** new
parallelism recipes, model configs, CI/tests, docs, dataloaders.
**Easy wins:** add a training config for a newly released open model;
extend a parallelism combo to a model that doesn't have it yet.

### nano-vLLM — https://github.com/GeeeekExplorer/nano-vllm
**Activity:** small but alive; forks are plentiful. Use as the
**canonical reference** for build-projects 3–5 above. Forks accept PRs
readily and are an easier place to get review feedback than vLLM itself.

### tinygrad — https://github.com/tinygrad/tinygrad — ~27k stars
**Activity:** very active; daily commits, recent 2026 work on FSDP,
fused QKV backward, CDNA SQTT timing. Runs a **paid bounty program**
(tracked on a public Google sheet) — a real option to get paid to learn
GPU/compiler internals. **PRs newcomers land:** ops, backends, model
ports, bug fixes tied to bounties. **Easy wins:** pick the smallest open
bounty; the team explicitly uses bounties as a hiring funnel.

### nanochat — https://github.com/karpathy/nanochat
**Activity:** released Oct 2025, updated through Feb 2026; Karpathy's
successor to nanoGPT, covering tokenization → pretrain → finetune → eval
→ inference → chat UI in hackable code. PRs are rare (Karpathy prefers
forks) but it's the best **read-through** for fundamentals. Also check
`karpathy/microgpt` (Feb 12 2026, 200 lines).

### llm-d — https://github.com/llm-d/llm-d
**Activity:** new-ish, active; CNCF-adjacent effort for distributed
vLLM on K8s with KV-cache-aware routing (Red Hat-led, Oct 2025 onward).
**PRs newcomers land:** Gateway plugins, routing policies, Helm charts,
benchmarks, Python client tooling. Sweet spot for a Python backend
engineer who wants the **K8s + inference** intersection.

### TensorRT-LLM — https://github.com/NVIDIA/TensorRT-LLM
**Activity:** active (PyTorch 2.9.1 / Triton 3.5.1 base image updates,
N-gram spec decoding, disaggregated serving). Contribution bar is higher
(CLA, CUDA-heavy), but the Python API layer and model definitions are
approachable. **PRs newcomers land:** model definitions, Python examples,
tests, docs. Good to **read** even if you don't PR.

### Skip / caveat
**HuggingFace TGI** (huggingface/text-generation-inference) was
**archived on March 21 2026** — historical reference only. **LitGPT**
(Lightning-AI/litgpt) is maintained but lower-velocity (v0.5.12 Dec
2025); good to read for clean model implementations, not the
fastest-moving PR target.

---

## Suggested path

1. Build-projects **1–4** in parallel with reading nano-vLLM end-to-end.
2. Build-projects **5–9** alongside a first vLLM "new model" or SGLang
   tokenizer PR.
3. Build-projects **10–14** while reading gpt-fast and torchtitan, then
   pick up a tinygrad bounty or an llm-d routing PR to cement production
   skills.
