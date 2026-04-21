# Roadmap — One Engineer, One Lab, One Year

A single top-down plan derived from the other notes in this directory.
The other files are **reference material**; this is the path.

**Target role:** Research Engineer / Member of Technical Staff at a
frontier AI lab — concretely, the Anthropic "Research Engineer,
Agents" (Agentic Systems team) JD. Equivalent openings exist at
OpenAI, Google DeepMind, Meta FAIR, AI2, LangChain, Nous Research.

**Starting assumption:** Senior Python backend engineer. Comfortable
with distributed systems. New to LLM internals, training, and
agent-specific work.

**Guiding principle:** *one engineer cannot learn everything at once.*
Pick the narrowest path that lands the job. Everything outside that
path is optional.

---

## 0. The top-down choice — pick a specialization first

Frontier labs hire against *specializations*, not generalists. Before
you start reading, pick one of these three lanes. They share ~30%
common foundation; the top layer diverges.

| Lane | Role shape | Reference JD | Primary lab targets |
|------|------------|--------------|--------------------|
| **A. Agents / Post-training** | Build harnesses, design evals, train reasoning/agent models with RL. | Anthropic Research Engineer, Agents | Anthropic, OpenAI (Agents), DeepMind (agents), LangChain, Nous, Sierra |
| **B. Inference systems** | Scheduler, KV cache, kernels, multi-host serving. | Anthropic/OpenAI Inference Engineer | Anthropic, OpenAI, NVIDIA, Together, Fireworks |
| **C. Training infra / pre-training** | Parallelism, mega-cluster reliability, checkpointing. | OpenAI Training Engineer | OpenAI, Anthropic, DeepMind, xAI |

The user in this repo has been building toward **Lane A** (agents).
This roadmap is written for Lane A. The appendix has a pointer to
switch to B or C without starting over.

---

## 1. The top-down map — what a Lane-A engineer actually does

At a lab, an agents engineer's job is some mix of the following. Your
year-long plan must produce evidence that you can do *each* of them.

```
             ┌───────────────────────────────────────────────┐
             │    Level 4  —  Frontier agent research        │
             │  (Novel harness, novel eval, novel training)  │
             └───────────────────────────────────────────────┘
                              ▲
             ┌───────────────────────────────────────────────┐
             │       Level 3  —  Training loops on LMs       │
             │  (SFT → DPO → GRPO → RLVR on real tasks)      │
             └───────────────────────────────────────────────┘
                              ▲
             ┌───────────────────────────────────────────────┐
             │    Level 2  —  Agent harness + evals          │
             │  (Loop, tools, memory, context, benchmarks)   │
             └───────────────────────────────────────────────┘
                              ▲
             ┌───────────────────────────────────────────────┐
             │    Level 1  —  LLM foundations & internals    │
             │  (Transformer, sampling, PyTorch, KV basics)  │
             └───────────────────────────────────────────────┘
```

You learn **top-down** (understand what L4 looks like before grinding
L1) but you build **bottom-up** (can't ship L3 without L1/L2
primitives).

---

## 2. The year in four phases

~40 weeks part-time (10–15 hrs/week) or ~20 weeks full-time. Each
phase ends with a *shippable portfolio artifact*, not a checkbox of
readings.

### Phase 1 (weeks 1–4) — Foundations: stop guessing at LLM internals

**Read (pick, don't grind):**

1. Vaswani — *Attention Is All You Need* — once, for notation.
2. Jay Alammar — *The Illustrated Transformer*.
3. Sebastian Raschka — *Build a Large Language Model (From Scratch)*,
   chs 1–4. Or nanoGPT if you prefer terse code.
4. Horace He — *Making Deep Learning Go Brrrr from First Principles*.
   Compute vs. memory-bandwidth vs. overhead — the only mental model
   you need for serving and training cost.
5. Lilian Weng — *Large Transformer Model Inference Optimization*
   (Jan 2023). High signal; read once.

**Skip / defer:** CS336 full course, FlashAttention papers, all of
`study-guide.md` beyond items 1–5. They're for Lane B.

**Build — `lm-from-scratch` (weekend project):**
A ~200-line PyTorch transformer that trains on TinyStories and
generates. Add a KV cache and measure the speedup. This is the only
"from-scratch LM" code you need. Reference: karpathy/nanoGPT.

**Artifact:** GitHub repo with a README explaining prefill vs. decode,
KV cache math, and your measured numbers.

**Gate to Phase 2:** you can explain, without notes, what `argmax` vs.
`torch.max` returns, what "sampling" means, what a KV cache stores,
and what `[batch, heads, seq, head_dim]` means.

---

### Phase 2 (weeks 5–14) — Agent harness & evals: the Lane-A core

This is the heart of Lane A. The JD is literally *"novel harness
design, custom evals, memory and context engineering."* Ship these
and you have 70% of what you need.

**Read, in this order:**

1. Anthropic — *Harness design for long-running application
   development*. Primary source from the team hiring for this role.
2. Armin Ronacher — *Pi: The Minimal Agent Within OpenClaw*. Clearest
   design-philosophy statement in the field; anchors minimalism.
3. LangChain — *The Anatomy of an Agent Harness*
   (`Agent = Model + Harness`) + *Deep Agents* + *Context Management
   for Deep Agents* + *Choosing the Right Multi-Agent Architecture*.
4. *Dive into Claude Code* (arXiv:2604.14228) — the design-space
   paper. Skim, then deep-dive the sections on the 5-layer compaction
   and 4-gate permission cascade.
5. Zain Hasan — *Inside Claude Code: An Architecture Deep Dive*. The
   most engineer-friendly walkthrough of Claude Code.
6. Epsilla — *12 Reusable Agentic Harness Design Patterns from Claude
   Code*.
7. Hamel Husain — *LLM Evals FAQ*. Your eval bible. Read in one sitting.
8. UIUC Kang Lab — *Agentic Benchmark Checklist* + Berkeley RDI —
   *How We Broke Top AI Agent Benchmarks*. If you skip these, your
   benchmarks will be exploitable.
9. From Storage to Experience — *Memory Mechanisms Survey* (2026),
   skim chapter on context engineering.

**Study one codebase end-to-end:**
**langchain-ai/deepagents** (Python, readable in a day). Read all 7
middleware modules: TodoList · Memory · Skills · Filesystem ·
SubAgent · Summarization · PatchToolCalls. This is your mental
model scaffold; every other harness maps onto these primitives.

**Build — four projects, ~8 weeks total:**

- **P1. Minimal Pi-style harness (1 week).** Loop + 4 tools
  (read/write/edit/bash). <500 lines. Single file if you can.
  Reference: openclaw/pi-mono.
- **P2. Permission cascade + filesystem-as-memory + compaction
  (2 weeks).** Bolt onto P1: 4-gate permission (rules → tool check →
  mode → prompt), `AGENTS.md` loader, 5-stage compaction pipeline.
  Reference: Claude Code design space paper.
- **P3. Subagents + MCP + lifecycle hooks (2 weeks).** Add
  `spawn_subagent()` with isolated context + sidechain transcripts.
  Expose 3 of your tools as MCP servers. 8 lifecycle hooks.
  Reference: deepagents' SubAgent + PatchToolCalls middleware.
- **P4. Eval pipeline on P3 (3 weeks).** Run the **full Hamel
  workflow** on your own agent: 100 traces → open code → axial code
  → quantify → validated LLM-as-judge with TPR/TNR >80%. Then
  adversarially probe your own eval using the 7 RDI vulnerability
  categories.

**Artifacts:** one repo per project. For P4, publish the failure
taxonomy, the TPR/TNR numbers, and the adversarial probe results.
This is the artifact that actually gets you interviews.

**Gate to Phase 3:** you can fill in the cross-harness comparison
table (Claude Code / Hermes / OpenClaw-Pi / deepagents across ~10
primitives) from memory.

---

### Phase 3 (weeks 15–26) — Post-training: SFT, DPO, GRPO, RLVR

Now the model-touching layer. Lane A needs this to *train* agents, not
just orchestrate them.

**Read, in this order:**

1. Sebastian Raschka — *Build a Reasoning Model (From Scratch)*.
   Laptop-friendly. Inference-time scaling → RL training →
   distillation. Do the exercises.
2. HF smol-course (SFT + DPO units). Actually run the notebooks.
3. Nathan Lambert — *RLHF Book* (free PDF). The textbook.
4. DPO paper (Rafailov et al., 2023). One read.
5. DeepSeek-R1 paper. The GRPO reference.
6. Tulu 3 report (AI2). The closest thing to a production recipe
   with data + code + weights all open.
7. Apple — *RL for Long-Horizon Interactive LLM Agents*
   (arXiv:2502.01600). LOOP; the cleanest multi-turn agent RL paper.
8. Agent-RLVR (arXiv:2506.11425) + DeepSWE. Where verifiable rewards
   meet SWE-bench.

**Skip for now:** PPO (DPO subsumes most of what you need); HAAF;
most multi-agent surveys; everything else in `post-training-and-
agents.md` §9–§16 that isn't DeepSeek-R1 or LOOP. You'll come back
to it later.

**Build — three projects, ~12 weeks:**

- **P5. SFT + DPO on a small model (2 weeks).** SmolLM3 or
  Qwen3-0.6B. Use TRL. Write the loops yourself the first time, then
  diff against TRL. Target: a chat-tuned model that holds a
  conversation.
- **P6. Tiny GRPO with a synthetic verifier (2 weeks).** Task:
  "output must parse as JSON matching this schema." G=8, β=0.04, no
  critic. Run on a single GPU. Track KL, reward, group variance.
  Reference: `trl/trainer/grpo_trainer.py`.
- **P7. Replace verifier with code execution (4 weeks).** Same loop
  as P6; reward = fraction of unit tests passing on model-emitted
  code. Sandbox properly (seccomp or Docker). Train on a small
  problem set. Target: meaningful pass@1 lift over the base model.
  This is the actual RLVR skill.

**Capstone project for the phase (P8, 4 weeks) — post-train your own
agent:**
Pick the language-learning agent project (the dedicated note
`language-learning-agent-project.md`): target Kalamang (MTOB), wire
your harness from Phase 2 into a GRPO loop with chrF + back-
translation reward. Agent Skills folders for dictionary, morphology,
grammar, examples.

This capstone is the single most important artifact in the whole
roadmap. It maps 1:1 onto the Anthropic JD:
- *"Large-scale RL on language models"* — ✅ you did it.
- *"Novel harness design"* — ✅ your Phase-2 harness.
- *"Rigorous quantitative benchmarks"* — ✅ MTOB + your ablations.
- *"Memory, context engineering, agent-to-agent communication"* —
  ✅ Agent Skills as durable memory.

**Gate to Phase 4:** you have trained a model with RL whose reward
came from tool execution, and you can explain every hyperparameter.

---

### Phase 4 (weeks 27–40) — Polish, write, ship, interview

You don't need more projects. You need to make the ones you have
*legible*.

**Write:**
- One **blog post** per project P2, P4, P7, P8. Each <2,500 words,
  with a graph and a code snippet. Publish on your own domain.
- One **arXiv-style writeup** for P8 if the numbers are interesting,
  even if unpublished. Title, abstract, method, results, ablations.
- Update your **GitHub** bios and READMEs so each repo states the
  one-sentence research claim.

**Read (interview-adjacent):**
- Chip Huyen — *AI Engineering* (relevant chapters: evals,
  observability, cost). Skim.
- *Agent Engineering: A New Discipline* (LangChain).
- *Agent Evaluation Readiness Checklist* (LangChain).
- A couple of LangChain customer case studies (Cleric SRE, Kensho
  financial data, Exa) — you'll be asked production-grounding
  questions.

**Contribute (optional, high-leverage if you can):**
- One merged PR to **TRL**, **deepagents**, or **Inspect AI** in the
  area you built in. Even a small one — it shows you can land code in
  someone else's codebase.

**Interview prep:**
- Do 5 live post-training / agent-design whiteboards with a peer.
  Pattern: "design an eval for X," "how would you add memory to Y,"
  "explain GRPO on a whiteboard."
- For every lab you apply to, read that lab's most recent 3 papers
  and their most recent engineering blog posts. Reference them in
  your cover letter.

**Apply:**
- 5–10 labs, batched. Cover letter has 3 sentences: who you are, what
  you built (one artifact per lab, matched to their JD), why them.
- If rejected early, request async feedback; use it to refine the
  portfolio.

---

## 3. Priorities — what to cut when time gets short

If you lose half your time, cut in this order:

1. **First cut:** the inference/serving stack. Lane B material. You
   can cite it in interviews without building it. Skip `buildable-
   projects.md` items 7–15.
2. **Second cut:** Phase 4 contributions. A blog post is worth more
   than an unmerged PR.
3. **Third cut:** Phase 3's SFT project (P5) *if and only if* you
   have a chat-tuned open model you can build on. DPO+GRPO are the
   load-bearing pieces.
4. **Never cut:** Phase 2 P4 (evals). This is what separates
   research engineers from app devs in interviews.

What to *double down* on if you have slack:

- Another **capstone-scale** project beyond P8. The single strongest
  signal to a hiring manager is two projects with real results, not
  one.
- Reading **two more agent codebases** end-to-end. After deepagents,
  add claude-agent-sdk-python and Hermes Agent. Write a 1,500-word
  architectural comparison. Post it publicly. It will get attention.

---

## 4. The canonical reading list (40 items max)

Everything else in the other notes is *reference*. If you only read
these, you're prepared.

**Foundations (5):**
1. Vaswani — Attention Is All You Need
2. Alammar — Illustrated Transformer
3. Raschka — Build an LLM from Scratch (chs 1–4)
4. Horace He — Making DL Go Brrrr
5. Lilian Weng — LLM Inference Optimization survey

**Harness & agents (10):**
6. Anthropic — Harness design for long-running apps
7. Ronacher — Pi: The Minimal Agent
8. LangChain — The Anatomy of an Agent Harness
9. LangChain — Deep Agents
10. LangChain — Context Management for Deep Agents
11. LangChain — Choosing the Right Multi-Agent Architecture
12. arXiv:2604.14228 — Dive into Claude Code (design space)
13. Zain Hasan — Inside Claude Code
14. Epsilla — 12 Reusable Harness Patterns
15. Hermes Agent architecture doc (Nous Research)

**Evals (5):**
16. Hamel Husain — LLM Evals FAQ
17. LangChain — Agent Evaluation Readiness Checklist
18. LangChain — How We Build Evals for Deep Agents
19. Berkeley RDI — How We Broke Top AI Agent Benchmarks
20. UIUC Kang Lab — Agentic Benchmark Checklist paper

**Post-training (10):**
21. Raschka — Build a Reasoning Model from Scratch (book)
22. HF smol-course (SFT + DPO units)
23. Lambert — RLHF Book
24. InstructGPT paper
25. DPO paper (Rafailov et al.)
26. DeepSeek-R1 paper
27. Tulu 3 report
28. Apple LOOP (arXiv:2502.01600)
29. Agent-RLVR (arXiv:2506.11425)
30. DeepSWE writeup (Together AI)

**Memory, context, continual (5):**
31. Karpathy note — Context Engineering (via Weaviate/LC summaries)
32. From Storage to Experience (memory survey, 2026)
33. A-Mem paper (arXiv:2502.12110)
34. LangChain — Continual Learning for AI Agents
35. Lifelong Learning of LLM Agents roadmap (TPAMI)

**Production context (5):**
36. LangChain — You Don't Know What Your Agent Will Do Until Prod
37. Phil Schmid — How Kimi/Cursor/Chroma Train Agentic Models with RL
38. Lee Han Chung — A Taxonomy of RL Environments for LLM Agents
39. Chip Huyen — AI Engineering (eval + observability chapters)
40. Your target lab's 3 most recent papers + engineering blog posts

---

## 5. The projects list (8 items)

In order, with effort budgets. Do them all.

1. **P1 — Minimal Pi-style harness** (1 wk)
2. **P2 — Permissions + FS-as-memory + 5-stage compaction** (2 wk)
3. **P3 — Subagents + MCP + hooks** (2 wk)
4. **P4 — Full Hamel-workflow eval + RDI adversarial probes** (3 wk)
5. **P5 — SFT + DPO on a small model** (2 wk)
6. **P6 — GRPO with a JSON verifier** (2 wk)
7. **P7 — GRPO with a code-execution verifier** (4 wk)
8. **P8 — Language-learning agent (capstone)** (4 wk)

**What "done" means for any project:**
1. You can explain every file you wrote without opening it.
2. You can rebuild the core primitive from scratch in a day.
3. You can name 3 design decisions you made that differ from the
   reference implementation, and why.
If not all three: redo it.

---

## 6. What this roadmap explicitly does NOT cover

Deliberately cut to make the path tractable. Revisit later if needed.

- **K8s / inference serving** (the content of `study-guide.md`,
  `buildable-projects.md`, `inference-serving-*.md`, `k8s-ramp.md`).
  That's Lane B. Cite it in interviews; don't build it.
- **Large-scale pre-training / training infra** (parallelism, mega-
  cluster reliability). That's Lane C.
- **Alignment research** beyond what's necessary to get RLHF/RLVR to
  work. Fascinating; not on the critical path.
- **Multi-agent beyond the basics.** You'll cover it in Phase 2 via
  LangChain's posts. Deeper work comes after you land the job.
- **Kernels, CUDA, Triton, FlashAttention internals.** Lane B.

If you catch yourself spending >1 day in any of the above, you are
drifting from the plan.

---

## 7. Appendix — If you pick Lane B or Lane C

**Lane B (inference systems):** the plan is already written — follow
`study-guide.md` as the reading list and `buildable-projects.md`
items 1–14 as the projects. Keep Phase 1 and the foundations section
here; swap Phases 2–4 for that plan. Target JDs: vLLM/SGLang
contributors, Anthropic/OpenAI inference engineers, NVIDIA.

**Lane C (training infra):** read *How to Scale Your Model* (Google
DeepMind/JAX, free online book) + Megatron paper + nanoGPT at scale +
torchtitan code. Build: a multi-node TP/FSDP run on 2–4 GPUs, a
checkpointing system, a fault-tolerant training loop. Capstone: a
reproduction of a public pretraining recipe at 1B scale. Target JDs:
OpenAI/Anthropic/DeepMind training infra.

---

## 8. One-sentence summary

**Build a minimal agent harness → wrap it in rigorous evals → post-
train a small reasoning model inside your own harness with RL →
publish the receipts.** Everything else is optional.
