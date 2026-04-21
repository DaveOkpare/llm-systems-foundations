# Post-Training & Agent Training — Study Guide

A curated reading list for a Python backend engineer ramping into LLM
post-training and agent training. Companion to `study-guide.md` (which
covers the inference/serving side). Sources skew toward recent
(2024–2026) surveys, canonical papers, and practitioner guides.

Context for the reader: post-training is where a raw pretrained model
becomes something useful (chat, reasoning, agent). It sits between
pre-training and the inference/serving layer. This note assumes
familiarity with PyTorch and basic transformer internals.

---

## 0. Mental model

### The LLM lifecycle

1. **Pre-training** — next-token prediction on huge corpora.
2. **Post-training** — SFT, preference optimization, RLVR, distillation.
3. **Inference / decoding** — sampling strategies (greedy, temperature,
   top-k, top-p).
4. **Serving** — KV-cache, prefix sharing, prefill/decode disaggregation,
   batching, quantization. (Covered in `study-guide.md`.)

### Taxonomy of post-training

- **SFT** — train on `(prompt, ideal response)` pairs with next-token
  cross-entropy. Includes instruction tuning, chat tuning, domain SFT,
  and rejection-sampling SFT.
- **Preference optimization (RLHF family)** — train on `(prompt, chosen,
  rejected)` triples. PPO → DPO (dominant) → IPO / KTO / ORPO / SimPO /
  SLiC. Also RLAIF / Constitutional AI.
- **RL with verifiable rewards (RLVR)** — skip human preferences; use an
  automatic verifier (tests, math checker, compiler) as the reward.
  GRPO (DeepSeek), RLOO, REINFORCE++, DAPO, VAPO. Powers o1, R1, Qwen
  reasoning models.
- **Distillation** — response, logit, and reasoning-trace distillation.
- **Alignment / safety tuning** — red-teaming + refusal SFT, harmlessness
  preference data, Constitutional AI loops.
- **Continued / mid-training** (boundary case) — continued pretraining,
  long-context extension (YaRN, RoPE scaling), tool-use tuning.

Typical modern pipeline (Llama 3, Qwen3, DeepSeek-V3):
`pretrain → (continued PT) → SFT → rejection-sampling SFT → DPO/RLHF →
RLVR (for reasoning) → safety tuning`.

### Key distinction — sampling vs. serving vs. training

- **Sampling strategies** (inference-time): greedy, temperature, top-k,
  top-p, beam. `argmax` gives the *index* (token id) into the vocab;
  `torch.max` gives the *value* (logit) — only the index decodes.
- **Serving optimizations** (systems): KV-cache, prefix sharing,
  prefill/decode disaggregation — orthogonal to sampling.
- **Post-training**: changes the model weights; different loop from both
  of the above.

---

## 1. Post-training — start here (hands-on, Python-first)

- **Build a Reasoning Model (From Scratch)** — Sebastian Raschka,
  Manning (2025/26). Pure PyTorch; laptop-friendly. Inference-time
  scaling → RL training → distillation.
  - https://www.manning.com/books/build-a-reasoning-model-from-scratch
  - Code: https://github.com/rasbt/reasoning-from-scratch

- **Hugging Face smol-course** — Free; runs locally on SmolLM3. SFT,
  DPO, preference alignment using TRL. Best way to actually ship a
  post-trained model without a cluster.
  - https://huggingface.co/learn/smol-course/unit0/1

- **Hugging Face Alignment Handbook** — Recipes (YAML + scripts) used to
  reproduce Zephyr/Tulu. Reading the configs teaches the pipeline better
  than most blog posts.
  - https://github.com/huggingface/alignment-handbook

- **The Smol Training Playbook** (HF Space, 2026) — Practitioner guide
  tying SFT + DPO + eval into one workflow.
  - https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook

- **TRL (Transformers Reinforcement Learning)** — The go-to library;
  implements SFT, DPO, PPO, GRPO.
  - https://huggingface.co/docs/trl/index

## 2. Post-training — reference textbook

- **RLHF Book** — Nathan Lambert, `rlhfbook.com`. Free PDF. De facto
  textbook for RLHF/DPO/GRPO. Readable; math-light enough for engineers
  but math-honest.
  - https://rlhfbook.com/book.pdf

## 3. Post-training — surveys (read after the vocab lands)

- **LLM Post-Training: A Deep Dive into Reasoning LLMs** (arXiv:2502.21321).
  Strong taxonomy of SFT, RLHF, DPO, RLAIF, GRPO in one place.
  - https://arxiv.org/pdf/2502.21321
- **A Survey of RL for Large Reasoning Models** (arXiv:2509.08827).
  Focused on the post-o1/R1 reasoning wave. Best single paper for *where
  the field is now*.
  - https://arxiv.org/abs/2509.08827
- **SFT vs RL: Comprehensive Survey on LLM Post-training** (OpenReview,
  March 2026). Direct comparison of tradeoffs; very current.
  - https://openreview.net/forum?id=eSkxiOsS37
- **Toward Large Reasoning Models** (Cell Patterns, 2025). Peer-reviewed.
  - https://www.cell.com/patterns/fulltext/S2666-3899(25)00218-1

## 4. Post-training — foundational papers

- **InstructGPT** — Ouyang et al., 2022. Original RLHF recipe.
- **DPO: Your Language Model Is Secretly a Reward Model** — Rafailov
  et al., 2023. Why most labs dropped PPO.
- **DeepSeek-R1** — 2025. GRPO + RLVR in practice; kicked off the open
  reasoning-model era.
- **Tulu 3** — AI2, 2024/25. Fully open recipe (data + code + weights)
  for modern post-training. Best production-shaped reference.

## 5. Post-training — living resources

- **Sebastian Raschka's Substack** — monthly "state of" syntheses.
  https://magazine.sebastianraschka.com/
- **mbzuai-oryx/Awesome-LLM-Post-training** (GitHub) — curated paper list.
- **TsinghuaC3I/Awesome-RL-for-LRMs** (GitHub) — curated paper list.

### Suggested post-training order

Raschka's book → smol-course (SFT + DPO units) → RLHF Book → one survey
(start with arXiv:2509.08827) → DeepSeek-R1 → Tulu 3 report.

---

## 6. Agents — mental model first

Agent training is mostly **environment engineering**. The model is the
small part; most work is rewards, rollouts, verifiers, and curricula.

- **A Taxonomy of RL Environments for LLM Agents** — Lee Han Chung,
  March 2026. Best single post to understand environments, rewards,
  rollouts, and verifiers. Read before the papers.
  - https://leehanchung.github.io/blogs/2026/03/21/rl-environments-for-llm-agents/
- **How Kimi, Cursor, and Chroma Train Agentic Models with RL** — Phil
  Schmid, 2026. Production case studies.
  - https://www.philschmid.de/kimi-composer-context
- **Unsloth: RL environments and how to build them** — Hands-on guide
  to writing the environment, not just the model side.
  - https://unsloth.ai/blog/rl-environments

## 7. Agents — long-horizon training

- **A Survey on the Optimization of LLM-based Agents** (arXiv:2503.12434,
  2026). SFT/RL/reward-design taxonomy for agent tasks.
  - https://arxiv.org/html/2503.12434v2
- **LLM Agents: A Comprehensive Survey** (Preprints, Dec 2025). Covers
  long-horizon planning, memory, safety, scalability.
  - https://www.preprints.org/manuscript/202512.2119
- **RL for Long-Horizon Interactive LLM Agents** (arXiv:2502.01600,
  Apple). Introduces LOOP (lean PPO variant); 32B agent beats o1 on
  AppWorld by 9pp. Cleanest example of end-to-end long-horizon agent RL.
  - https://arxiv.org/abs/2502.01600
- **AgentGym-RL** (arXiv:2509.08755, ICLR 2026). Open framework for
  multi-turn agent RL. Reproduce locally to internalize the training loop.
  - https://arxiv.org/abs/2509.08755
  - Code: https://github.com/WooooDyy/AgentGym-RL

## 8. Agents — tool use & verifiable rewards

- **Agent-RLVR** (arXiv:2506.11425). RLVR for SWE-bench: Pass@1 9.4%
  → 22.4% with 817 training envs. Shows how to turn sparse "did tests
  pass" into a usable signal.
  - https://arxiv.org/html/2506.11425
- **DeepSWE** (Together AI). Fully open SOTA coding agent trained with
  pure RL on Qwen3-32B. Weights + code + write-up.
  - https://www.together.ai/blog/deepswe
- **SkyRL-Agent** (arXiv:2511.16108). Efficient multi-turn RL; SA-SWE-32B
  hits 39.4% on SWE-Bench Verified.
  - https://arxiv.org/html/2511.16108v1
- **Eigen AI: Reliable Post-Training for Interactive Tool-Using Agents**
  (Feb 2026). Self-evolving data + verifier design on τ²-bench.
  - https://www.eigenai.com/blog/2026-02-23-reliable-post-training-interactive-tool-agents
- **opendilab/awesome-RLVR** (GitHub) — continually updated list.

## 9. Agents — continual / lifelong learning

- **Lifelong Learning of LLM-based Agents: A Roadmap** (TPAMI, 2026).
  Closest thing to a textbook in this subfield.
  - https://pubmed.ncbi.nlm.nih.gov/41489969/
  - List: https://github.com/qianlima-lab/awesome-lifelong-llm-agent
- **From Storage to Experience: A Survey on the Evolution of LLM Agent
  Memory Mechanisms** (OpenReview, 2026). Three-stage framework
  (Storage → Reflection → Experience).
  - https://openreview.net/forum?id=l9Ly41xxPb
- **Memory for Autonomous LLM Agents** (arXiv:2603.07670). Formalizes
  memory as a write–manage–read loop; five mechanism families.
  - https://arxiv.org/html/2603.07670v1
- **A-Mem: Agentic Memory for LLM Agents** (arXiv:2502.12110).
  Practical, implementable memory system.
  - https://arxiv.org/pdf/2502.12110
- **MemoryBench** (arXiv:2510.17281). First benchmark specifically for
  continual learning in LLM systems.
  - https://arxiv.org/html/2510.17281v4

## 10. Agents — test-time / self-improving

- **Self-Improving LLM Agents at Test-Time** (arXiv:2510.07841).
  Identify hard samples → augment → test-time fine-tune. +5.48% avg with
  68× less training data.
  - https://arxiv.org/abs/2510.07841
- **Learning on the Job: Test-Time Curricula for Targeted RL**
  (arXiv:2510.04786). TTC-RL picks its own curriculum; 1.8× pass@1 on
  AIME25, 2.1× on CodeElo.
  - https://arxiv.org/html/2510.04786v1
- **EvoTest** (arXiv:2510.13220). Evolutionary test-time learning.
  - https://arxiv.org/html/2510.13220
- **HF Blog: AI Trends 2026 — Test-Time Reasoning & Reflective Agents**
  — synthesis of where this subfield is headed.
  - https://huggingface.co/blog/aufklarer/ai-trends-2026-test-time-reasoning-reflective-agen

## 11. Agents — production orchestration (not training, but adjacent)

- **LangChain State of Agent Engineering 2026** — what teams actually
  ship. Grounds which post-training choices matter.
  - https://www.langchain.com/state-of-agent-engineering
- **Agentic AI Engineering with LangChain & LangGraph** — Norman, 2026.
- **AI Agents and Applications** — Roberto Infante, Manning. Python-first;
  LangGraph, tool-use, multi-agent workflows.
  - https://www.manning.com/books/ai-agents-and-applications

---

## Suggested order for agents

Environments taxonomy post → Unsloth env-building guide → Apple LOOP
paper (arXiv:2502.01600) → AgentGym-RL (reproduce locally) → Agent-RLVR
+ DeepSWE → memory survey (Storage→Reflection→Experience) → A-Mem
paper → test-time learning papers → LangGraph book for orchestration.

## Practical notes

- Pick **one** open agent-RL framework (AgentGym-RL or SkyRL-Agent) and
  run a real training job, even tiny. Agent post-training is ~70%
  environment/infra engineering — reading won't substitute.
- Verifier quality is the ceiling. Auto-generated envs with weak reward
  functions will teach the wrong behaviors at scale.
- For solo work, prefer DPO/GRPO over PPO: no critic network, simpler
  infra, competitive results.
- RLVR beats RLHF wherever you can write a verifier (code, math, SQL,
  structured output). Reach for preference data only when you can't.

---

## 12. Targeted prep — Anthropic "Research Engineer, Agents" (Agentic Systems team)

Topics pulled directly from the JD that this note didn't yet cover in
depth. The role focuses on: large-scale RL on LMs, novel **harness
design**, multi-agent systems, memory / context engineering,
agent-to-agent communication, and rigorous agentic evals/benchmarks.
Applied domains: coding agents, research automation, customer support,
network security.

### 12.1 Agent harness design

A **harness** is the software that wraps the model: tool orchestration,
sub-agents, filesystem access, human approvals, prompts, lifecycle,
context management. "2025 was agents. 2026 is harnesses." The model is
commodity; the harness is the product.

- **Anthropic: Harness design for long-running application development**
  — primary source from the team hiring for this role. Covers context
  anxiety in Sonnet 4.5, why compaction alone wasn't enough, and why
  context resets became essential.
  - https://www.anthropic.com/engineering/harness-design-long-running-apps
- **Dive into Claude Code: The Design Space of Today's and Future AI
  Agent Systems** (arXiv:2604.14228). Research-paper treatment of the
  Claude Code harness.
  - https://arxiv.org/html/2604.14228v1
- **12 Reusable Agentic Harness Design Patterns from Claude Code**
  (Epsilla, April 2026). Four categories: memory/context, workflow/
  orchestration, tools/permissions, automation.
  - https://www.epsilla.com/blogs/2026-04-18-deep-dive-12-reusable-agentic-harness-design-patte
- **Phil Schmid: The importance of Agent Harness in 2026**.
  - https://www.philschmid.de/agent-harness-2026
- **OpenHarness** (HKUDS, GitHub) — open-source agent harness reference
  implementation.
  - https://github.com/HKUDS/OpenHarness

### 12.2 Multi-agent systems & agent-to-agent communication

- **Multi-Agent Collaboration Mechanisms: A Survey of LLMs**
  (arXiv:2501.06322). Taxonomy of actors, types (cooperation/
  competition/coopetition), structures (centralized/decentralized/
  layered/nested), strategies, and coordination protocols.
  - https://arxiv.org/abs/2501.06322
- **A Communication-Centric Survey of LLM-Based Multi-Agent Systems**
  (arXiv:2502.14321). Three parts: communication paradigms, structures,
  content. The right frame for "agent-to-agent communication."
  - https://arxiv.org/pdf/2502.14321
- **Springer: LLM-based multi-agent systems — workflow, infrastructure,
  and challenges**.
  - https://link.springer.com/article/10.1007/s44336-024-00009-2
- **WMAC 2026 (AAAI Bridge Program)** — "Advancing LLM-Based Multi-Agent
  Collaboration." Newest venue; workshop papers are bleeding-edge.
  - https://multiagents.org/2026/
- **LLM-Based Human-Agent Collaboration Systems Survey**
  (arXiv:2505.00753).
  - https://arxiv.org/html/2505.00753v4
- **taichengguo/LLM_MultiAgents_Survey_Papers** (GitHub) — curated list.
- Standard frameworks to read code of: **AutoGen**, **CrewAI**,
  **LangGraph multi-agent**, **Swarm** (OpenAI).

### 12.3 Context engineering & context compression

Karpathy's term: deliberately architect what the model sees, how much,
in what order. Treat the context window as a scarce resource.

- **Weaviate: Context Engineering — LLM Memory and Retrieval for AI
  Agents**. Clean primer.
  - https://weaviate.io/blog/context-engineering
- **Automatic Context Compression in LLM Agents** (Plaban Nayak, March
  2026). Why agents need to forget, and how.
  - https://medium.com/the-ai-forum/automatic-context-compression-in-llm-agents-why-agents-need-to-forget-and-how-to-help-them-do-it-43bff14c341d
- **Mastra: Observational Memory** (ZenML LLMOps DB). 5–40× compression
  while preserving temporal awareness.
  - https://www.zenml.io/llmops-database/observational-memory-human-inspired-context-compression-for-agent-systems
- **Active Context Compression: Autonomous Memory Management in LLM
  Agents** (arXiv:2601.07190). Focus architecture — agents decide when
  to consolidate vs. prune.
  - https://arxiv.org/abs/2601.07190
- **ACON: Optimizing Context Compression** (OpenReview).
  - https://openreview.net/pdf?id=7JbSwX6bNL
- **Agent Memory Systems in 2026: What Actually Matters**.
  - https://blog.bymar.co/posts/agent-memory-systems-2026/
- **The LLM context problem in 2026** (LogRocket). Memory, relevance,
  scale strategies; also covers "context rot."
  - https://blog.logrocket.com/llm-context-problem/
- **Memory for AI Agents: A New Paradigm of Context Engineering** (The
  New Stack).
  - https://thenewstack.io/memory-for-ai-agents-a-new-paradigm-of-context-engineering/

(Cross-reference section 9 — continual/lifelong memory — for the
training-side treatment of memory.)

### 12.4 Evals & benchmarks for agentic tasks

The JD calls out "rigorous quantitative benchmarks for large scale
agentic tasks" and "model-based evaluation techniques at scale."

- **philschmid/ai-agent-benchmark-compendium** (GitHub) — 50+ agent
  benchmarks across Function Calling & Tool Use, General Assistant &
  Reasoning, Coding & SWE, Computer Interaction. Best single starting
  point.
  - https://github.com/philschmid/ai-agent-benchmark-compendium
- **SWE-bench + SWE-bench Verified** — repo-level issue resolution.
  - https://www.swebench.com/ ·
    https://openai.com/index/introducing-swe-bench-verified/
- **τ-bench / τ²-bench** — tool-agent-user interaction in real-world
  domains (airline, retail, telecom).
- **OSWorld** — 369 desktop computing tasks in a full Ubuntu VM.
- **WebArena**, **GAIA**, **Terminal-Bench**, **AppWorld** — standard
  long-horizon agent benchmarks.
- **How We Broke Top AI Agent Benchmarks** (Berkeley RDI, 2026) —
  required reading on *why evals are harder than you think*. Agents
  exploit evaluation mechanisms to score high without solving tasks.
  - https://rdi.berkeley.edu/blog/trustworthy-benchmarks-cont/
- **OccuBench** (arXiv:2604.10866) — evaluating agents on real-world
  professional tasks via language-environment simulation.
  - https://arxiv.org/html/2604.10866
- **MemoryBench** (section 9, arXiv:2510.17281) — continual learning
  eval.
- **2025–2026 AI Computer-Use Benchmarks guide** (o-mega.ai).
  - https://o-mega.ai/articles/the-2025-2026-guide-to-ai-computer-use-benchmarks-and-top-ai-agents

### 12.5 Large-scale RL on LMs (depth for the JD's core skill)

Beyond sections 3–7, go deep on:

- **GRPO** (DeepSeek-R1 paper, 2025) — critic-free, group-relative
  policy optimization. Default for RLVR-scale training.
- **DAPO** (ByteDance, 2025) — decoupled clip + dynamic sampling on
  top of GRPO. Strong on long-CoT RL.
- **VAPO**, **REINFORCE++**, **RLOO** — stability/efficiency variants.
- **Process Reward Models (PRMs)** — reward intermediate reasoning
  steps; relevant for long-horizon agents.
- **verl** (Volcengine Reinforcement Learning) and **OpenRLHF** — the
  production-grade training frameworks. Read their schedulers.
- **OpenReward / Open Reward Standard (ORS)** (2026) — MCP-extending
  protocol for RL episodes, rewards, curricula. Relevant to harness ↔
  training-infra boundary.

### 12.6 Suggested prep sequence for this role

1. Anthropic harness-design post → Claude Code paper (arXiv:2604.14228)
   → 12-patterns deep dive.
2. Multi-agent communication survey (arXiv:2502.14321) + collaboration
   mechanisms survey (arXiv:2501.06322).
3. Weaviate context-engineering primer → Active Context Compression
   paper → Mastra case study.
4. Agent benchmarks compendium — pick 3 to run locally (SWE-bench
   Verified, τ²-bench, OSWorld). Then read the Berkeley "How We Broke"
   post.
5. DeepSeek-R1 + DAPO papers → reproduce a tiny GRPO run on verl or
   TRL → wire it into AgentGym-RL.
6. Build a toy harness + toy benchmark + tiny RL loop end-to-end. This
   is what the JD actually tests for.

### 12.7 Principles worth internalizing

- The model is commodity. The harness, eval, and environment are the
  product.
- Separate **generation** from **evaluation** — agents will confidently
  praise mediocre work otherwise (Anthropic's own guidance).
- Context is a scarce resource. Design for forgetting, not just
  remembering.
- Most agent benchmarks are exploitable. Assume your reward function is
  wrong until proven otherwise.
- Multi-agent ≠ better. Overhead of coordination often exceeds gains
  unless the structure (centralized/hierarchical/peer) matches the task.
