# Roadmap — One Engineer, One Lab, One Year

A single top-down plan derived from the other notes in this directory.
The other files are **reference material**; this is the path.

**Target role:** Research Engineer / Member of Technical Staff at a
frontier AI lab — primarily the Anthropic *Research Engineer, Agents*
(Agentic Systems team) JD. Equivalent openings exist at OpenAI,
Google DeepMind, Meta FAIR, AI2, LangChain, Nous Research, Sierra.

**Starting assumption:** Senior Python backend engineer. **New to**
LLM internals, inference systems, distributed / GPU systems, k8s,
post-training, agents. Assumes strong SWE fundamentals; everything
else is on the learning path.

**Guiding principle:** *one engineer cannot learn everything at once.*
Pick the narrowest path that lands the job. Every side quest is
deferred.

---

## 0. The top-down choice — pick a specialization, plus a shared base

Frontier labs hire against *specializations*, not generalists — but in
**2026, inference fluency is table stakes regardless of lane.** No one
at a lab can ignore the decoding loop, KV cache, or serving economics.

Three lanes, shared foundations:

| Lane | Role shape | Reference JD |
|------|------------|--------------|
| **A. Agents / Post-training** | Harnesses, evals, RL on LMs. | Anthropic Research Engineer, Agents |
| **B. Inference systems** | Scheduler, KV cache, kernels, multi-host serving, k8s. | Anthropic/OpenAI Inference Engineer; NVIDIA; Together; Fireworks |
| **C. Training infra / pre-training** | Parallelism, mega-cluster reliability. | OpenAI Training Engineer |

**This roadmap targets Lane A** (agents) — but weaves in enough of
Lane B that the engineer can hold a conversation about serving in a
Lane-A interview, *and* switch to a Lane-B offer if one lands first.
Lane C is deferred to the appendix.

---

## 1. The ladder — what the job actually looks like

```
             ┌──────────────────────────────────────────────────┐
             │    Level 5  —  Frontier agent research           │
             │  (Novel harness, novel eval, novel training)     │
             └──────────────────────────────────────────────────┘
                               ▲
             ┌──────────────────────────────────────────────────┐
             │    Level 4  —  Train models inside your harness  │
             │  (SFT → DPO → GRPO → RLVR on real tasks)         │
             └──────────────────────────────────────────────────┘
                               ▲
             ┌──────────────────────────────────────────────────┐
             │    Level 3  —  Products on top of your harness   │
             │  (Pi→OpenClaw, DeepAgents→OpenSWE, SDK→Code)     │
             └──────────────────────────────────────────────────┘
                               ▲
             ┌──────────────────────────────────────────────────┐
             │    Level 2  —  Agent harness & eval discipline   │
             │  (Loop, tools, memory, context, benchmarks)      │
             └──────────────────────────────────────────────────┘
                               ▲
             ┌──────────────────────────────────────────────────┐
             │    Level 1  —  LLM + inference foundations       │
             │  (Transformer, sampling, KV cache, batching, k8s)│
             └──────────────────────────────────────────────────┘
```

You learn **top-down** (understand Level 5 before grinding Level 1)
but you build **bottom-up** (can't ship Level 4 without Level 1–3).

### The Level-3 framing (the user's insight, load-bearing)

Harnesses don't matter in the abstract; they matter because of what
you ship on them.

| Harness (runtime / library) | Product(s) built on it |
|------------------------------|------------------------|
| **Pi / pi-mono** (4 tools, minimal) | OpenClaw · Pi Coding Agent |
| **deepagents** (LangGraph middleware) | OpenSWE · Deep Research clones · enterprise search agents |
| **Claude Agent SDK** | Claude Code · Claude Cowork · SDK Demos |
| **Hermes Agent** | Personal self-improving agent with platform adapters |

**The discipline for the portfolio:** build **one** minimal harness,
then build **one** real product on top. That's the demonstration a
hiring manager actually wants — that you can both design the runtime
and use it for something specific.

---

## 2. The year in six phases

~40 weeks part-time (10–15 hrs/week) or ~20 weeks full-time. Each
phase ends with a *shippable portfolio artifact*, not a checkbox.

### Phase 1 (weeks 1–4) — LLM foundations: stop guessing at internals

**Read (narrow list):**

1. Vaswani — *Attention Is All You Need*. Once, for notation.
2. Jay Alammar — *The Illustrated Transformer*.
3. Sebastian Raschka — *Build an LLM from Scratch*, chs 1–4 (or
   nanoGPT if you prefer terse code).
4. Horace He — *Making Deep Learning Go Brrrr from First Principles*.
   Compute vs memory-bandwidth vs overhead. The only mental model you
   need for reasoning about cost later.
5. Lilian Weng — *LLM Inference Optimization* (Jan 2023).

**Build — `lm-from-scratch` (weekend):**
~200-line PyTorch transformer that trains on TinyStories and
generates. Add a KV cache; measure the speedup. The one "from-scratch
LM" project you need.

**Optional bolt-on — `sampling-dashboard` (weekend):**
Gradio/Streamlit UI with live sliders for `temperature`, `top-k`,
`top-p`, `min-p`, `repetition_penalty`, `logit_bias`. Render the
probability distribution + entropy at each decoding step. Pays off
twice: in Phase 5, rollout temperature directly controls GRPO
reward variance; in the capstone, logit bias + structured sampling
save you from hallucinated tool arguments.

**Gate to Phase 2:** explain without notes: `argmax` vs `torch.max`,
sampling strategies, KV cache contents, tensor shape
`[batch, heads, seq, head_dim]`, prefill vs decode.

---

### Phase 2 (weeks 5–8) — Inference & serving: the 2026 table stakes

**Why this is mandatory now:** every 2026 frontier-lab engineer, Lane
A included, must understand scheduling, KV cache, and the k8s
surface an inference stack runs on. Lane-A interviews *will* ask how
you'd instrument latency of an agentic rollout; "I don't know how
serving works" is disqualifying.

**Read (the 30%-of-Lane-B-material you actually need):**

1. *Orca* (OSDI 2022) — continuous batching, the foundational
   scheduling idea.
2. Anyscale — *How continuous batching enables 23× throughput in LLM
   inference*. Engineer-friendly explainer.
3. *PagedAttention* / vLLM paper (SOSP 2023). KV as virtual memory.
4. *SGLang + RadixAttention* (2024). Prefix caching.
5. *Sarathi-Serve* (OSDI 2024) — chunked prefill.
6. *DistServe* (OSDI 2024) — prefill/decode disaggregation.
7. `inference-serving-taxonomy.md` in this repo. Pin it open.
8. For k8s: `k8s-ramp.md` in this repo — the minimal path to
   understand LWS deployments, HPAs, and prefix-affinity routing.
   Don't try to master k8s; learn it through an LLM lens.

**Build — `mini-serve` in two stages (~3 weeks total).**
Minimal first, then extend. Mirrors the rest of the roadmap's
approach and the genre of small readable reimplementations
(nano-vLLM → Mini-SGLang → production SGLang).

**Stage A — the nano-vLLM level (~1 week):**
- a continuous batcher with an `asyncio.Queue` of requests;
- a block-based KV cache (paged);
- an OpenAI-compatible `/v1/chat/completions` FastAPI endpoint with
  SSE streaming, TTFT / inter-token-latency metrics, cancellation;
- a `Dockerfile` and a minimal Helm chart deploying it with an HPA
  on queue depth.

Reference: **nano-vLLM** (~1,200 LOC). End of Stage A, you have a
working paged-attention engine.

**Stage B — extend toward Mini-SGLang (~2 weeks):**
- split the monolith into **four processes** (`api_server` /
  `tokenizer` / `scheduler` / `detokenizer`) talking over ZeroMQ;
- add a **radix-tree prefix cache** (RadixAttention);
- add **chunked prefill** (Sarathi-Serve-style stall-free batching);
- add **CUDA graphs** for the decode path.

Reference: **Mini-SGLang** (~5k LOC Python, production-faithful).
Diff-audit Stage B against it when you finish. End of Stage B, you
have the modern-engine mental model: scheduler as separate actor,
prefix reuse, chunked prefill, graph capture.

**Why this engine pays off twice:** `mini-serve` is also your rollout
engine in Phase 5. Throughput here directly caps GRPO training
throughput — an honest Lane-A reason to care about serving perf.

**Deliberately skipped:** CUDA/Triton kernel writing, FlashAttention
internals, tensor/pipeline parallelism, MoE, speculative decoding
kernels. Those are Lane B's deep stack. You can cite them; you don't
need to build them.

**Gate to Phase 3:** you can draw the request path through `mini-
serve` on a whiteboard, state the bottleneck at each layer, and
explain what k8s objects your service depends on.

---

### Phase 3 (weeks 9–20) — One harness + one product on top

**The core Lane-A phase.** Two projects, tightly coupled. The
discipline is Pi→OpenClaw: a minimal runtime, then a real product
that uses it.

**Read, in this order:**

1. Anthropic — *Harness design for long-running application
   development*. Primary source from the team hiring for this role.
2. Armin Ronacher — *Pi: The Minimal Agent Within OpenClaw*. Clearest
   design-philosophy statement in the field.
3. LangChain — *The Anatomy of an Agent Harness*
   (`Agent = Model + Harness`) + *Deep Agents* + *Context Management
   for Deep Agents* + *Choosing the Right Multi-Agent Architecture*.
4. *Dive into Claude Code* (arXiv:2604.14228) — design space. Skim,
   then deep-dive 5-layer compaction + 4-gate permission cascade.
5. Zain Hasan — *Inside Claude Code: An Architecture Deep Dive*.
6. Epsilla — *12 Reusable Agentic Harness Design Patterns from Claude
   Code*.
7. Hermes Agent architecture doc (Nous Research).

**Study one codebase end-to-end:** **langchain-ai/deepagents**
(Python, readable in a day). All 7 middleware modules. Every other
harness maps onto these primitives; this is your scaffold.

**Build P1 — `harness` (4 weeks):**
One repo. A minimal, opinionated harness that includes:
- a single `loop()` that can serve CLI, HTTP, and SDK surfaces;
- 4 base tools (read/write/edit/bash);
- 4-gate permission cascade (rules → tool `check_permissions` → mode
  → user prompt);
- filesystem-as-memory with `AGENTS.md` loader;
- 5-stage compaction pipeline (budget → snip → microcompact →
  collapse → autocompact);
- subagents with isolated context + sidechain transcripts;
- 8 lifecycle hooks (PreToolUse, PostToolUse, PreModelCall, …);
- MCP client + one MCP server for one of your tools.

Target: <1,500 lines Python, reads like a textbook, published as a
library with an `ARCHITECTURE.md` and a one-line `install_my_harness`.
Diff-audit against deepagents.

**Build P2 — one real product on top of `harness` (4 weeks):**
Pick **one**. These mirror the shipped examples above:

| Model | Your version | Scope |
|-------|--------------|-------|
| OpenSWE on deepagents | **`coding-agent`** — SWE-agent-style ACI on your harness, evaluated on 20 SWE-bench-Verified tasks. | Small, measurable. |
| Claude Code on SDK | **`terminal-coder`** — terminal CLI with your harness, project-scoped `AGENTS.md`, permission prompts. | Product polish. |
| Deep Research clone | **`research-agent`** — planner + parallel subagents + synthesizer on a narrow domain (arXiv papers in one sub-field). | Multi-agent pattern. |
| Cleric-style SRE | **`on-call-agent`** — agent watching one GitHub repo + Slack channel, proposes PRs, with skill accretion. | Ambient pattern. |

Recommendation: **`coding-agent`** — most legible to hiring managers,
direct line to SWE-bench numbers in interviews.

**Artifacts from Phase 3:** two repos — the harness library and the
product — each with a ≤2,500-word blog post explaining architecture
and one measured number.

**Gate to Phase 4:** you can fill the cross-harness comparison table
(Claude Code / Hermes / Pi / deepagents across ~10 primitives) from
memory, and you've used your own harness to build something that
works end-to-end.

---

### Phase 4 (weeks 21–24) — Eval discipline

**This is the single most under-built skill in the field.** Evals are
what separate research engineers from app devs in interviews.

**Read:**

1. Hamel Husain — *LLM Evals FAQ*. Your eval bible; one sitting.
2. LangChain — *Agent Evaluation Readiness Checklist*.
3. LangChain — *How We Build Evals for Deep Agents*.
4. UIUC Kang Lab — *Agentic Benchmark Checklist* (arXiv:2507.02825).
5. Berkeley RDI — *How We Broke Top AI Agent Benchmarks*. 7
   vulnerability categories.
6. LangChain — *You Don't Know What Your Agent Will Do Until It's in
   Production*.

**Build P3 — `eval-suite` on top of P2 (3 weeks):**

Apply the **full Hamel workflow** to your product from Phase 3:

1. Collect 100 real traces from `coding-agent` (or chosen P2).
2. Open-code 50 — failure taxonomy.
3. Quantify: transition matrix of last-success → first-failure.
4. Build one validated LLM-as-judge (100 hand-labels, TPR/TNR > 80%
   on held-out).
5. Run the RDI adversarial probes against your own eval. Fix what
   breaks.
6. Pass the UIUC Agentic Benchmark Checklist on your task/outcome
   validity.

**Artifact:** a public blog post titled "We ran [Anthropic/Berkeley's
eval] on our own agent — here's what we learned." Include the failure
taxonomy, the judge TPR/TNR, and the adversarial probe table.

This artifact is the one that actually gets you interviews.

---

### Phase 5 (weeks 25–36) — Post-training: models trained inside your harness

Now touch model weights. Small, focused, the path to Level 4.

**Read, in this order:**

1. Sebastian Raschka — *Build a Reasoning Model (From Scratch)*,
   paired with Cameron R. Wolfe — *Demystifying Reasoning Models*.
   Do the book's exercises; use Wolfe for concept reinforcement.
2. HF smol-course (SFT + DPO units). Run the notebooks.
3. Nathan Lambert — *RLHF Book*. Textbook.
4. DPO paper (Rafailov et al., 2023).
5. DeepSeek-R1 paper.
6. Tulu 3 report (AI2).
7. Cameron R. Wolfe — *RL Scaling Laws for LLMs* + *GRPO++: Tricks
   for Making RL Actually Work*. **Required before P5.** Vanilla
   GRPO stalls without the variants (GSPO / DAPO / Dr. GRPO / TIS /
   CISPO); this is where you learn which to reach for.
8. Apple — *RL for Long-Horizon Interactive LLM Agents*
   (arXiv:2502.01600). LOOP; cleanest multi-turn agent RL paper.
9. Agent-RLVR (arXiv:2506.11425) + DeepSWE.

**Defer:** PPO (DPO subsumes what you need), HAAF, multi-agent RL
surveys, everything else.

**Build P4 — SFT + DPO on a small model (2 weeks).**
SmolLM3 or Qwen3-0.6B. TRL. Write the loops yourself once; diff
against TRL after. (Small-model choice is deliberate: under tight
compute, learning efficiency saturates with size — you get more
update steps at 0.6B than at 8B. See the Qwen 2.5 scaling paper
covered in Wolfe's *RL Scaling Laws*.)

**Build P5 — GRPO with a code-execution verifier (4 weeks).**
Same loop pattern as DPO project; reward = unit-test pass fraction.
Sandbox (seccomp or Docker). Train on a small problem set; target
meaningful pass@1 lift.

Three gotchas to plan for up front:
- **TIS (Truncated Importance Sampling).** If your rollout engine
  (vLLM / `mini-serve`) and training engine (HF) produce different
  token probabilities, gradients silently degrade. TIS corrects it.
- **Rollout count `G` is not fixed.** Start `G=8`; scale up on
  harder prompts. Optimal allocation is difficulty-dependent.
- **Pick one GRPO variant, not vanilla.** Dr. GRPO (bias-free) or
  DAPO (asymmetric clip + dynamic sampling) are good defaults.

**Build P6 — Capstone (~4 weeks):**

Take your **`harness`** (P1) + your **`eval-suite`** (P3) + your
GRPO loop (P5). Use them to train a small reasoning agent. The
dedicated note `language-learning-agent-project.md` contains the
full plan: target Kalamang (MTOB), wire Agent Skills (dictionary,
morphology, grammar, examples) into your harness, reward = chrF +
back-translation. This is the single highest-leverage artifact in
the whole roadmap.

Why this capstone: it ties together *every level of the ladder*. It
maps 1:1 onto the Anthropic JD:
- *"Large-scale RL on language models"* — P5 + P6.
- *"Novel harness design"* — P1.
- *"Rigorous quantitative benchmarks"* — P3 + MTOB.
- *"Memory, context engineering, agent-to-agent communication"* —
  Agent Skills as durable memory on your harness.

Budget-dependent: one A100/H100 for 3–5 days is enough.

---

### Phase 6 (weeks 37–40) — Polish, write, ship, apply

You don't need more projects. You need legibility.

**Write:**
- Blog post per project (P1, P2, P3, P5, P6). <2,500 words, one graph,
  one code snippet.
- An arXiv-style writeup for P6 if the numbers warrant, even if
  unpublished. Title/abstract/method/results/ablations.
- GitHub READMEs: each repo states the one-sentence research claim up
  top.

**Read (interview-adjacent, light):**
- Chip Huyen — *AI Engineering*, evals + observability chapters.
- LangChain — *Agent Engineering: A New Discipline*.
- Your target lab's 3 most recent papers + 2 most recent engineering
  blog posts. Reference them in cover letters.

**Contribute (high-leverage if you can):**
One merged PR to **TRL**, **deepagents**, or **Inspect AI** in the
area you built. Even small — shows you can land code in someone
else's repo.

**Interview prep:**
- 5 live whiteboards with a peer: "design an eval for X," "add memory
  to Y," "explain GRPO."
- A one-page narrative: the year, in sequence, each project one line.

**Apply:**
- 5–10 labs, batched. Cover letter: who you are, what you built
  (one artifact per lab matched to the JD), why them.

---

## 3. Priorities — what to cut when time runs short

Cut order (first cut first):

1. **First cut:** Phase 6 contributions. A blog post beats an
   unmerged PR.
2. **Second cut:** `mini-serve`'s k8s layer. Keep the engine; skip
   the Helm chart. You can still speak to it.
3. **Third cut:** Phase 5's SFT/DPO project (P4) *if and only if*
   you can start from an open chat-tuned model. GRPO is load-bearing.
4. **Never cut:**
   - Phase 3's P1 (harness) + P2 (product on top). Shows design range.
   - Phase 4 P3 (evals). This is what separates research engineers.
   - Phase 5 P6 (capstone). The single artifact that sells the whole
     year.

Slack? Double down here (in order):
- A second Phase-3 product on the same harness. If your harness can
  host two different products, that's a library, not a demo.
- Read two more harness codebases end-to-end (claude-agent-sdk-python
  + Hermes Agent); publish a 1,500-word architecture comparison.
- A stretched version of P6 — add continual learning via skill-file
  accretion, or a second language (Manchu).

---

## 4. The canonical reading list (~40 items)

Everything in the other notes is *reference*. This list is the
curriculum.

**LLM + inference foundations (10):**
1. Vaswani — Attention Is All You Need
2. Alammar — Illustrated Transformer
3. Raschka — Build an LLM from Scratch (chs 1–4)
4. Horace He — Making DL Go Brrrr
5. Weng — LLM Inference Optimization
6. Orca (OSDI 2022) — continuous batching
7. Anyscale — Continuous Batching 23× post
8. PagedAttention / vLLM (SOSP 2023)
9. SGLang + RadixAttention (2024)
10. Sarathi-Serve (OSDI 2024) + DistServe (OSDI 2024)

**Harness & agents (10):**
11. Anthropic — Harness design for long-running apps
12. Ronacher — Pi: The Minimal Agent
13. LangChain — The Anatomy of an Agent Harness
14. LangChain — Deep Agents + Context Management for Deep Agents
15. LangChain — Choosing the Right Multi-Agent Architecture
16. arXiv:2604.14228 — Dive into Claude Code
17. Zain Hasan — Inside Claude Code
18. Epsilla — 12 Reusable Harness Patterns
19. Hermes Agent architecture doc
20. Lee Han Chung — Taxonomy of RL Environments for LLM Agents

**Evals (5):**
21. Hamel Husain — LLM Evals FAQ
22. LangChain — Agent Evaluation Readiness Checklist
23. LangChain — How We Build Evals for Deep Agents
24. Berkeley RDI — How We Broke Top AI Agent Benchmarks
25. UIUC Kang Lab — Agentic Benchmark Checklist paper

**Post-training (10):**
26. Raschka — Build a Reasoning Model from Scratch
27. HF smol-course (SFT + DPO units)
28. Lambert — RLHF Book
29. InstructGPT paper
30. DPO paper (Rafailov et al.)
31. DeepSeek-R1 paper
32. Tulu 3 report
33. Apple LOOP (arXiv:2502.01600)
34. Agent-RLVR (arXiv:2506.11425)
35. DeepSWE writeup

**Production & target-lab context (5):**
36. LangChain — You Don't Know What Your Agent Will Do Until Prod
37. Phil Schmid — How Kimi/Cursor/Chroma Train Agentic Models with RL
38. LangChain — Agent Engineering: A New Discipline
39. Chip Huyen — AI Engineering (evals + observability chapters)
40. Your target lab's 3 most recent papers + eng blog posts

---

## 5. The projects list (6 items)

In order, with effort budgets. Do them all.

1. **`lm-from-scratch`** — 200-line transformer + KV cache. (weekend)
2. **`mini-serve`** — continuous batch + paged KV + radix cache +
   OpenAI-compat FastAPI + k8s deploy. (3 weeks)
3. **`harness`** — minimal agent runtime (loop, 4 tools, permission
   cascade, FS-as-memory, compaction, subagents, MCP, hooks). (4 weeks)
4. **`product-on-harness`** — one real product built on `harness`
   (recommend `coding-agent` on SWE-bench subset). (4 weeks)
5. **`eval-suite`** — Hamel workflow + validated LLM-as-judge +
   RDI adversarial probes on your product. (3 weeks)
6. **Capstone — `language-learning-agent`** — SFT + GRPO + code-exec
   verifier loop → MTOB/Kalamang with Agent Skills on your harness.
   (8 weeks, combines P4 SFT/DPO + P5 GRPO + P6 capstone from the
   earlier plan)

**"Done" for any project:**
1. You can explain every file without opening it.
2. You can rebuild the core primitive from scratch in a day.
3. You can name 3 design decisions that differ from the reference
   implementation, and why.

If not all three: redo it.

---

## 6. What this roadmap explicitly does NOT cover

Deliberately cut to keep the path tractable:

- **Kernels, CUDA, Triton, FlashAttention internals.** Cite; don't
  build. Lane-B deep stack.
- **Tensor/pipeline/expert parallelism.** You'll meet Megatron's
  ideas reading other papers; don't build TP on 8 GPUs unless Lane B.
- **Mega-cluster training infra, checkpointing at scale, fault
  tolerance.** Lane C.
- **Alignment research** beyond what's needed to get RLHF/RLVR to
  work.
- **Multi-agent** beyond what LangChain's posts give you in Phase 3.

Catch yourself >1 day in any of these → you are drifting.

---

## 7. Appendix — switching lanes

- **Lane B (inference systems):** keep Phases 1–2; replace Phase 3's
  harness with deeper serving work from `buildable-projects.md`
  items 6–14 (speculative decoding, chunked prefill scheduler, KV
  quant, TP on 2 GPUs, disaggregation, Triton kernel). Keep Phase 4
  (evals still matter). Drop Phase 5. Target: vLLM/SGLang
  contributions, Anthropic/OpenAI inference engineering.
- **Lane C (training infra):** read *How to Scale Your Model* +
  Megatron + torchtitan code; build a multi-node TP/FSDP run + a
  checkpointing system; capstone = reproduce a public pretraining
  recipe at 1B scale. Target: OpenAI/Anthropic/DeepMind training.

---

## 8. One-sentence summary

**Build one minimal harness, build one real product on top of it,
wrap it in rigorous evals, then post-train a small reasoning model
inside your own harness with RL. Publish the receipts.**
