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

Topics pulled from the JD: large-scale RL on LMs, novel **harness
design**, multi-agent systems, memory / context engineering,
agent-to-agent communication, and rigorous agentic evals/benchmarks.
Applied domains: coding agents, research automation, customer support,
network security.

### 12.1 Agent harness design (from primary sources)

A **harness** is the software that wraps the model: tool orchestration,
sub-agents, filesystem access, human approvals, prompts, lifecycle,
context management. In the Claude Code codebase, only ~1.6% is
AI-logic; 98.4% is operational infrastructure. The model is commodity;
the harness is the product.

#### Two persistent failure modes (Anthropic's own framing)

1. **Context window degradation** — models "lose coherence on lengthy
   tasks as the context window fills." Observed in Sonnet 4.5 as
   *context anxiety*: agents prematurely conclude work as they approach
   perceived token limits.
2. **Self-evaluation bias** — when asked to assess their own output,
   agents "respond by confidently praising the work — even when quality
   is obviously mediocre." Worst for subjective tasks lacking binary
   verification.

#### Anthropic's three-agent pattern

- **Planner** — turns 1–4-sentence prompts into full specs; stays at
  direction level.
- **Generator** — implements against the spec; self-evals but still
  biased toward optimism.
- **Evaluator** — independent, uses Playwright MCP to interact with the
  live app, grades against concrete criteria.

Measured on a "retro game maker" task with Opus 4.5: solo agent 20 min
/ \$9 / broken; full harness 6 hr / \$200 / complete & playable. 20× cost
for dramatically higher quality.

#### Context management: compaction vs. resets

- **Compaction** — summarize earlier conversation in place. Preserves
  continuity but "doesn't give the agent a clean slate, which means
  context anxiety can still persist."
- **Context resets** — clear the window; hand off via structured files.
  Psychological relief + orchestration complexity + token overhead +
  latency.
- With Opus 4.6, resets became unnecessary for many tasks; SDK-level
  compaction still helps.

#### Sprint contracts

Before each sprint, generator and evaluator negotiate a contract —
agreeing on what "done" looks like *before* any code is written. Bridges
the spec-to-test gap.

#### Claude Code design space (arXiv:2604.14228) — 13 design principles

Mapping values (human authority, safety, reliability, capability
amplification, contextual adaptability) to implementation choices:

- Deny-first escalation · Graduated trust spectrum · Defense in depth ·
  Externalized policy · Context as scarce resource · Append-only durable
  state · Minimal scaffolding · Values over rules · Composable
  extensibility · Reversibility-weighted assessment · Transparent
  file-based memory · Isolated subagents · Graceful recovery.

#### Seven-component architecture

```
User → Interfaces → Agent Loop → Permission System → Tools → Execution
                         ↓
                    State & Persistence
```

All surfaces (CLI, headless, SDK, IDE) converge on one `queryLoop()`
async generator. Five subsystems: Surface · Core · Safety/Action ·
State · Backend.

#### Seven-layer safety architecture

Any layer can block: tool pre-filtering → deny-first rules → permission
modes → ML classifier → shell sandbox → no-permission-restore on
resume → hook interception.

#### Permission modes (graduated autonomy)

`plan → default → acceptEdits → auto → dontAsk → bypassPermissions` plus
internal `bubble` for subagent escalation.

#### Five-stage compaction pipeline

Lighter stages first, every model call:

1. **Budget reduction** — per-message caps on tool results.
2. **Snip** — temporal trimming of old history.
3. **Microcompact** — fine-grained compression, optionally cache-aware.
4. **Context collapse** — read-time projection (no mutation).
5. **Auto-compact** — full semantic summary via model call (last resort).

#### Four extensibility mechanisms (at different context costs)

MCP servers · Plugins · Skills · Hooks (27 event types; 5 safety-
critical, 22 orchestration).

#### Twelve reusable harness patterns (from Claude Code, Epsilla 2026)

**Memory & Context**
1. **Persistent Instruction File** (e.g. CLAUDE.md) — auto-injected,
   version-controlled.
2. **Scoped Context Assembly** — org/project/subdirectory layering;
   lazy-load by CWD. Monorepo-friendly.
3. **Tiered Memory** — lean index in context, details loaded on demand,
   logs on disk.
4. **Dream Consolidation** — background dedup/prune during idle.
5. **Progressive Context Compaction** — tiered compression; full
   fidelity on recent, aggressive on old.

**Workflow & Orchestration**
6. **Explore-Plan-Act Loop** — read-only → planning → execution with
   escalating permissions.
7. **Context-Isolated Subagents** — Researcher/Planner/Executor, each
   sandboxed, returns summary only.
8. **Fork-Join Parallelism** — shard across sub-agents in isolated
   workspaces; merge.

**Tools & Permissions**
9. **Progressive Tool Expansion** — minimalist default set; mount more
   on demand. Prevents decision paralysis.
10. **Command Risk Classification** — risk-router middleware; low-risk
    bypasses gating, high-risk requires approval.
11. **Single-Purpose Tool Design** — specific, deterministic tools over
    generic shell wrappers.

**Automation**
12. **Deterministic Lifecycle Hooks** — for non-negotiable ops
    (formatting, validation) bound to events, run by middleware.

#### Critical failure modes (from the Claude Code paper)

- **Approval fatigue** — users approve ~93% of permission prompts;
  interactive confirmation alone is insufficient.
- **Context overflow** — no single compaction strategy covers all
  pressure; five-layer pipeline needed.
- **Silent failures** — weak observability when agents diverge.
- **Capability paradox** — 27% task amplification vs. documented 17%
  lower comprehension in AI-assisted conditions.
- **Layered safety degradation** — commands with >50 subcommands fall
  back to generic approval for UI reasons.

#### Open directions

Observability/eval gap · cross-session persistence · harness boundary
evolution · horizon scaling (session → multi-month) · governance at
scale · long-term human capability preservation.

#### Primary sources

- **Anthropic: Harness design for long-running application
  development**. https://www.anthropic.com/engineering/harness-design-long-running-apps
- **Dive into Claude Code: The Design Space of Today's and Future AI
  Agent Systems** (arXiv:2604.14228).
  https://arxiv.org/html/2604.14228v1
- **12 Reusable Agentic Harness Design Patterns from Claude Code**
  (Epsilla, April 2026).
  https://www.epsilla.com/blogs/2026-04-18-deep-dive-12-reusable-agentic-harness-design-patte
- Phil Schmid — https://www.philschmid.de/agent-harness-2026
- OpenHarness (HKUDS) — https://github.com/HKUDS/OpenHarness

### 12.1a Great agent harnesses — study the production code

The single most useful thing a prospective agent engineer can do: read
the source of the best shipping harnesses. Each embodies different
choices along the same design axes. Read ≥2 to triangulate.

#### Claude Code (Anthropic)

- **Architecture:** Terminal-native client; single `queryLoop()` async
  generator across all surfaces (CLI, headless, SDK, IDE).
- **Split:** ~1.6% AI logic / 98.4% operational infrastructure.
- **Safety:** Seven-layer deny-first stack + 7 permission modes + ML
  classifier + shell sandbox + 27 hook event types.
- **Context:** Five-stage compaction pipeline (budget → snip →
  microcompact → context collapse → auto-compact).
- **State:** Append-only JSONL transcripts; CLAUDE.md hierarchy;
  sidechain files for subagent isolation.
- **Extensibility:** Four mechanisms at different context costs — MCP,
  Plugins, Skills, Hooks.
- **Failure-mode tell:** 93% approval-prompt accept rate → confirmation
  alone is not a safety mechanism.
- **Read:** arXiv:2604.14228 for the full design-space writeup;
  Anthropic's harness-design blog post; Epsilla's 12-patterns deep
  dive.

#### OpenClaw (open-source, ~354k GitHub stars in early 2026)

- **Architecture:** Long-running Node.js service. Message-platform-first
  OS treating conversations as first-class entities. BYOK (bring your
  own key) for any model vendor.
- **Core pieces:** **Gateway** (always-on control plane — sessions,
  channel routing, tool dispatch, events) + pluggable **Agent Harness**
  plugins (per-turn loop logic, tool timing, internal state).
- **Loop:** Continuous ReAct loop (not one-shot response) for long
  workflows.
- **Extensibility:** Plugin marketplace; sandboxed tool execution;
  multi-channel messaging integrations.
- **Weakness:** Heavy configuration; rapid token consumption with many
  tools mounted.
- **Read:** OpenClaw docs — https://deepwiki.com/openclaw/docs/5.4-agent-harness-plugins ·
  Zylon practical guide —
  https://www.zylon.ai/resources/blog/what-is-openclaw-a-practical-guide-to-the-agent-harness-behind-the-hype ·
  Reference architecture (Feb 2026, Opus 4.6) —
  https://robotpaper.ai/reference-architecture-openclaw-early-feb-2026-edition-opus-4-6/

#### Hermes Agent (Nous Research, released Feb 2026; ~95.6k stars by
April)

- **Architecture:** Memory-first, self-improvement loops at the center.
  First productization of the five "Harness Engineering" components —
  **instructions / constraints / feedback / memory / orchestration**.
- **Learning loop:** Creates reusable **skills** from experience,
  refines them in use, persists knowledge, builds a deepening model of
  the user across sessions.
- **Memory:** Three-layer memory system.
- **Runtimes:** Five execution backends — local, Docker, SSH,
  Singularity, Modal — with container hardening and namespace
  isolation.
- **Footprint:** v0.10.0 ships with 118 bundled skills, six messaging
  integrations, and a closed experience → skill loop.
- **Weakness:** "Powerful but raw"; docs work-in-progress.
- **Read:** https://hermes-agent.nousresearch.com/docs/ ·
  https://github.com/nousresearch/hermes-agent

#### deepagents (LangChain, open-source)

The deliberate open-source reproduction of the *Deep Research / Manus /
Claude Code* pattern as a library. Written in Python on LangGraph.

- **Thesis:** Four things make agents "deep" instead of "shallow":
  1. a planning tool, 2. sub-agents, 3. a file system, 4. a detailed
  prompt. deepagents ships all four as a harness you instantiate via
  `create_deep_agent()`.
- **Runtime:** Returns a `CompiledStateGraph` (LangGraph). Built-in
  `Checkpointer` + `BaseStore` for persistence; `recursion_limit=1000`
  for deep tool chains.
- **Seven-layer middleware stack** (ordered):
  1. **TodoListMiddleware** — `write_todos` planning.
  2. **MemoryMiddleware** — loads context from `AGENTS.md` files into
     system prompt.
  3. **SkillsMiddleware** — dynamic loading of custom Python scripts
     as tools.
  4. **FilesystemMiddleware** — `ls`, `read_file`, `write_file`,
     `edit_file`, `glob`, `grep`.
  5. **SubAgentMiddleware** — spawn specialized sub-agents with
     isolated context windows.
  6. **SummarizationMiddleware** — auto-compaction at 85% token
     threshold; offloads to `/conversation_history/{thread_id}.md`.
  7. **PatchToolCallsMiddleware** — fixes LLM malformations in tool
     arguments.
- **Four backend implementations** for filesystem routing:
  `StateBackend` (ephemeral) · `StoreBackend` (persistent across
  sessions) · `FilesystemBackend` (disk) · `CompositeBackend`
  (e.g. `/memories/` → Store, `/` → disk).
- **Why to read it:** Python-first, small enough to read end-to-end in
  one sitting, and maps cleanly to the same concepts as Claude Code's
  C++/TS codebase. Best single artifact for a Python backend engineer
  trying to *build intuition* for harness design.
- **Read:** https://github.com/langchain-ai/deepagents ·
  https://blog.langchain.com/deep-agents/ ·
  https://deepwiki.com/langchain-ai/deepagents/1.3-architecture-overview

#### Comparative axes

| Axis | Claude Code | OpenClaw | Hermes Agent | deepagents |
|------|-------------|----------|--------------|------------|
| Surface | Terminal + IDE | Messaging + Gateway | Personal agent across messengers | Library (embed) |
| State | Append-only JSONL | Thread-based plugins | 3-layer memory + skills | Pluggable backends |
| Context mgmt | 5-stage pipeline | Plugin-based | Memory-first | Auto-compact at 85% |
| Subagents | Bubble mode + sidechain | Plugin-defined | Skill-scoped | SubAgentMiddleware |
| Safety | 7-layer deny-first | Sandbox per tool | Namespace isolation | Inherited from LangGraph |
| Tools | MCP + Plugins + Skills + Hooks | Plugin marketplace | 118 bundled skills | Middleware |
| License | Proprietary | MIT | MIT | MIT |
| Best for | Pro dev, codebase work | Personal OS, BYOK | Long-horizon autonomy, research | Building your own harness |

#### Reading strategy

1. Start with **deepagents** — read all seven middleware modules plus
   `create_deep_agent`. ~1 day. Clearest mental model.
2. Then **Claude Code paper** (arXiv:2604.14228) and Anthropic's
   harness-design post — you'll recognize every pattern from
   deepagents, just at production scale.
3. Skim **OpenClaw** Gateway + harness plugin docs to see a
   messaging-first take on the same primitives.
4. Read **Hermes Agent** source for the learning-loop / skills
   architecture — the pattern most missing from the others.

The patterns are the same across all four: planning tool, isolated
subagents, compaction, filesystem-as-memory, lifecycle hooks, tool
gating. Studying them in parallel teaches the *design space*, not any
one framework.

#### Comparison sources

- MCPlato harness comparison 2026 —
  https://mcplato.com/en/blog/ai-agent-harness-comparison-2026/
- All Things Open on OpenClaw —
  https://allthingsopen.org/articles/openclaw-viral-open-source-ai-agent-architecture
- DataCamp deepagents tutorial —
  https://www.datacamp.com/tutorial/deep-agents
- DataCamp Hermes tutorial —
  https://www.datacamp.com/tutorial/hermes-agent

### 12.2 Multi-agent systems & agent-to-agent communication

A communication-centric view (arXiv:2502.14321) splits the space into
**paradigms**, **structures**, and **content**.

#### Structures

- **Centralized / star** — hub agent coordinates spokes.
- **Decentralized / peer-to-peer** — agents negotiate directly.
- **Layered / hierarchical** — manager/subordinate chains.
- **Nested** — agents containing sub-systems.

Pick the structure for the task. Overhead of coordination often
exceeds gains if mismatched.

#### Collaboration mechanisms (arXiv:2501.06322)

Characterize by: **actors** · **types** (cooperation / competition /
coopetition) · **structure** · **strategies** (role-based, model-based)
· **coordination protocols**.

#### Communication content modalities

Debate (agents argue positions to sharpen reasoning) · simulation
(roleplay social/economic contexts) · tool-mediated coordination ·
hierarchical reporting.

#### Open problems

Scalability of communication · coherence across large teams · dialogue
efficiency · evaluation benchmarks for MAS · security (prompt injection
between agents).

#### Frameworks to read code of

AutoGen · CrewAI · LangGraph multi-agent · OpenAI Swarm · MetaGPT ·
ChatDev · AutoAgents.

#### Sources

- arXiv:2501.06322 — Multi-Agent Collaboration Mechanisms.
  https://arxiv.org/abs/2501.06322
- arXiv:2502.14321 — Communication-Centric Survey.
  https://arxiv.org/pdf/2502.14321
- WMAC 2026 (AAAI Bridge) — https://multiagents.org/2026/
- taichengguo/LLM_MultiAgents_Survey_Papers (GitHub) — curated list.

### 12.3 Context engineering & context compression

Karpathy's framing: "deliberately architect what the model sees, how
much, and in what order." Treat the context window as a scarce
resource.

#### Memory architecture layers

- **Short-term** — recent turns, tool outputs, retrieved docs (in
  context window).
- **Long-term** — vector DB: episodic, semantic, procedural.
- **Working** — task-specific transient info (dates, IDs).

"The worst memory system is the one that faithfully stores everything."
Filter on save, prune on schedule, summarize recurring material.

#### Chunking trade-off

Small chunks = precision, weak context. Large chunks = rich context,
noisy embeddings. "Getting it wrong forces the model to fall back on
hallucination."

#### Four context failure modes

- **Context poisoning** — hallucinated info compounds through reuse.
- **Context distraction** — excess history overwhelms fresh reasoning.
- **Context confusion** — irrelevant tools/docs misdirect the agent.
- **Context clash** — contradictory info creates paralysis.

#### Active Context Compression (arXiv:2601.07190)

Focus architecture, biologically inspired: agent autonomously decides
when to consolidate learnings into persistent **Knowledge** blocks and
when to prune raw history.

#### Observational Memory (Mastra)

Human-inspired compression, 5–40× while preserving temporal awareness
and contextual relevance.

#### Sources

- Weaviate — https://weaviate.io/blog/context-engineering
- Active Context Compression (arXiv:2601.07190) —
  https://arxiv.org/abs/2601.07190
- ACON (OpenReview) — https://openreview.net/pdf?id=7JbSwX6bNL
- Mastra Observational Memory —
  https://www.zenml.io/llmops-database/observational-memory-human-inspired-context-compression-for-agent-systems
- Automatic Context Compression (Medium, Mar 2026) —
  https://medium.com/the-ai-forum/automatic-context-compression-in-llm-agents-why-agents-need-to-forget-and-how-to-help-them-do-it-43bff14c341d
- The LLM context problem in 2026 (LogRocket) —
  https://blog.logrocket.com/llm-context-problem/

Cross-ref section 9 for the training-side treatment of memory.

### 12.4 Large-scale RL on LMs (depth for the JD)

Beyond sections 3–7:

- **GRPO** (DeepSeek-R1, 2025) — critic-free group-relative PO. Default
  for RLVR training.
- **DAPO** (ByteDance, 2025) — decoupled clip + dynamic sampling on top
  of GRPO. Strong on long-CoT.
- **VAPO, REINFORCE++, RLOO** — stability/efficiency variants.
- **Process Reward Models (PRMs)** — per-step rewards for long-horizon.
- **verl** (Volcengine) + **OpenRLHF** — production-grade frameworks;
  read the schedulers and rollout engines.
- **OpenReward / Open Reward Standard (ORS)** (2026) — MCP-extending
  protocol for RL episodes, rewards, curricula. Relevant at the harness
  ↔ training-infra boundary.

---

## 13. Designing good evals and benchmarks

Two layers: **product evals** (methodology for your own agent in your
own environment) and **benchmark design** (scientifically rigorous
tasks intended for the community).

### 13.1 Product evals — the Hamel Husain / Shreya Shankar workflow

Taught to 700+ engineers/PMs. Ordering matters.

**Step 1 — Error analysis (open coding).** Manually review 50–100
traces. Write open-ended notes, no pre-categorizing. Adapted from
qualitative research. Use a single domain expert as "benevolent
dictator." **Do not outsource this** — it's where product intuition
develops.

**Step 2 — Axial coding.** Group observations into a failure taxonomy:
prompt problems, tool design, model limits, tool failures, data gaps,
etc.

**Step 3 — Quantify.** Count failures per category. Visualize as a
transition matrix: last successful state → first failure point.

**Step 4 — Iterate until saturation.** Stop when no new categories
emerge.

**Step 5 — Build evaluators only for persistent failure modes.** Not
for failures you *imagine*.

**Step 6 — Validate LLM-as-judge.** Measure TPR/TNR against a held-out
human-labeled set. Apply correction factors if baseline TPR/TNR skews.

Key rule: error analysis is 60–80% of eval effort.

#### Evaluator tiers (cost-benefit)

- **Tier 1 (cheap):** regex, assertions, schema validation, reference
  comparisons.
- **Tier 2 (medium):** lightweight classifiers, structured code checks.
- **Tier 3 (expensive):** LLM-as-Judge — 100+ labeled examples + weekly
  maintenance. "Only build for problems you'll iterate on repeatedly."

#### Binary > Likert

"Binary evaluations force clearer thinking and more consistent
labeling. Likert scales introduce significant challenges: the
difference between adjacent points (like 3 vs 4) is subjective and
inconsistent across annotators." Force decisions; don't hide
uncertainty in middle values.

#### LLM-as-judge construction sequence

1. Identify failure modes via error analysis.
2. Label 100+ examples by domain expert.
3. Iteratively refine judge prompt on misalignment cases.
4. "Critique shadowing" — expert critiques judge predictions to surface
   systematic issues.
5. Measure on held-out validation set before prod.
6. Apply correction factors.

#### Multi-turn / agentic specifics

- Annotate **first upstream failure** only; downstream cascades from
  root causes.
- Use N-1 testing with real conversation prefixes when possible.
- Two-phase eval: (a) black-box end-to-end success, (b) step-level
  diagnostics (tool choice, param extraction, error handling, context
  retention).

#### CI vs. production

- **CI:** small curated datasets (100+), regression tests for past bugs,
  favor deterministic checks.
- **Production:** async evaluation on live traffic samples, reference-
  free LLM-as-judge on dashboards.
- **Guardrails ≠ evaluators.** Guardrails are sync, fast, deterministic,
  block output. Evaluators are async, nuanced, feed improvement loops.

#### Anti-patterns (Hamel)

- **Eval-driven development** — "LLMs have infinite surface area for
  potential failures. You can't anticipate what will break."
- **Generic metrics** — ROUGE, BERTScore, cosine similarity are "not
  useful for evaluating LLM outputs in most AI applications."
- **Outsourcing error analysis** — kills the feedback loop.
- **Automating prompt engineering too early** — "you risk never fully
  understanding your own requirements."
- **Obsessing over model selection** — "don't think of switching models
  as the main axis of improvement without evidence."
- **High eval pass rates** — "if you're passing 100%, you're likely not
  challenging your system enough. 70% may be more meaningful."

### 13.2 Benchmark design — the agentic benchmarks checklist (ABC)

For benchmarks you publish (not internal evals), rigor is a different
beast. From UIUC Kang Lab (arXiv:2507.02825, OpenReview 2026).

**Two validity concepts:**

- **Task Validity** — "a task should be solvable if and only if the
  agent possesses the target capability."
- **Outcome Validity** — "the evaluation method should indicate
  correctly whether the task has been solved."

Applied to 10 popular benchmarks, ABC revealed **outcome validity flaws
up to 40%**. On CVE-Bench specifically, ABC reduced performance
overestimation by **33%**.

**Required checks:**

1. **Outcome validity checks** — does the pass/fail actually reflect
   solving?
2. **Task validity checks** — does the task require the capability you
   claim to measure?
3. **Benchmark reporting checks** — full methodology + known validity
   gaps disclosed.

**Recommended practices:**

- Use **process-based** metrics alongside outcome-based.
- Benchmark LLM-as-a-judge reproducibly (same judge, same prompt, same
  seeds).
- Use **frozen websites** for navigation/reading tasks — live sites
  change and poison reproducibility.
- **Complete isolation** between agent execution and evaluation infra.
- Keep ground truth out of paths the agent can reach.

### 13.3 How benchmarks get broken (Berkeley RDI, 2026)

An automated scanning agent broke eight top benchmarks:

| Benchmark | Score achieved | Attack |
|-----------|---------------|--------|
| Terminal-Bench (89) | 100% | Binary wrapper trojans |
| SWE-bench Verified (500) | 100% | pytest hook injection |
| SWE-bench Pro (731) | 100% | Container parser overwrite |
| WebArena (812) | ~100% | `file://` URL steals answers |
| FieldWorkArena (890) | 100% | Zero capability validation |
| CAR-bench | 100% | LLM judge manipulation |
| GAIA (165) | ~98% | Public answer lookup |
| OSWorld (369) | 73% | Gold file direct download |

#### Seven recurring vulnerability categories

1. **Missing isolation** — agent and evaluator share environments.
2. **Exposed answers** — references in configs or public repos.
3. **Unsafe `eval()`** — direct evaluation of untrusted agent output.
4. **Unsanitized LLM judges** — prompt injection via agent content.
5. **Weak matching** — substring + aggressive normalization defeats
   precision.
6. **Broken evaluation logic** — components skipped or validators
   ignoring correctness.
7. **Trusting compromised output** — trusting artifacts the agent can
   tamper with.

#### Mitigations ("Agent-Eval Checklist")

- Isolate agent execution from eval infrastructure (separate processes,
  filesystems, networks).
- Answers in paths the agent cannot reach.
- Structured parsers, never `eval()`.
- Sanitized, delimited LLM judge inputs.
- Adversarial probes: null agent, injection agent, state-tampering
  agent.
- Avoid substring matching on short strings.
- Ground truth **never** published alongside leaderboard.

> "Don't trust the number. Trust the methodology."

### 13.4 Frameworks worth knowing

- **HAAF** (Holographic Agent Assessment Framework) — distribution-
  aware sampling across task types, tool interfaces, interaction
  dynamics, social contexts, risk levels.
  https://arxiv.org/abs/2603.14987
- **Agent Evaluation Readiness Checklist** (LangChain).
  https://blog.langchain.com/agent-evaluation-readiness-checklist/
- **Galileo Agent Eval Framework** — metrics/rubrics/benchmarks.
  https://galileo.ai/blog/agent-evaluation-framework-metrics-rubrics-benchmarks
- **philschmid/ai-agent-benchmark-compendium** — 50+ benchmarks
  organized by Function Calling, General Assistant, Coding/SWE,
  Computer Interaction. https://github.com/philschmid/ai-agent-benchmark-compendium

### 13.5 A starter eval-design playbook

For your own agent work, this is the default plan:

1. Ship something rough → collect 100 real traces.
2. Open-code 50+ traces; build a failure taxonomy.
3. Quantify: transition matrix of last-success → first-failure.
4. Top 3 categories → build Tier 1/2 evaluators.
5. Top subjective category → build one LLM-as-judge with 100 labels,
   validate TPR/TNR.
6. Wire into CI with a curated regression set.
7. Sample live traffic async for prod eval.
8. Re-do error analysis quarterly — failure distribution shifts.
9. If publishing a benchmark: add ABC review + Berkeley-RDI adversarial
   probes before release.

### 13.6 Key eval design sources

- **Hamel Husain: LLM Evals FAQ** (Jan 2026) —
  https://hamel.dev/blog/posts/evals-faq/
- **Hamel Husain: Your AI Product Needs Evals** —
  https://hamel.dev/blog/posts/evals/
- **Pragmatic Engineer: Pragmatic guide to LLM evals for devs** —
  https://newsletter.pragmaticengineer.com/p/evals
- **AI Evals For Engineers & PMs** (Maven course, Husain & Shankar) —
  https://maven.com/parlance-labs/evals
- **Establishing Best Practices for Building Rigorous Agentic
  Benchmarks** (arXiv:2507.02825) —
  https://arxiv.org/abs/2507.02825
- **Agentic Benchmark Checklist site** —
  https://uiuc-kang-lab.github.io/agentic-benchmarks/
- **How We Broke Top AI Agent Benchmarks** (Berkeley RDI) —
  https://rdi.berkeley.edu/blog/trustworthy-benchmarks-cont/
- **Beyond Benchmark Islands** (arXiv:2603.14987) —
  https://arxiv.org/abs/2603.14987
- **LLM Evaluation: Frameworks, Metrics, Best Practices (2026)** —
  https://futureagi.substack.com/p/llm-evaluation-frameworks-metrics

### 13.7 Suggested prep sequence for evals/benchmarks

1. Hamel evals-FAQ (one sitting) → start open-coding your own traces
   the same week.
2. Berkeley RDI "How We Broke" → internalize the seven vulnerability
   categories.
3. ABC paper + checklist site → apply to one existing benchmark and
   score its validity.
4. Build: pick one of your agents → run the full Hamel workflow →
   publish the failure taxonomy for the team.
5. Then read HAAF + LangChain readiness checklist for the
   production-monitoring layer.

---

## 14. LangChain blog — curated reading list

The LangChain blog is one of the highest-signal sources in 2025–2026:
it reads as a practitioner journal from a team that has shipped more
agent code than almost anyone else. Articles below are ordered to plug
into each chapter of this note. Many re-state ideas we've already
covered — that's the point. Reading the same pattern through a
different lens accelerates internalization.

### 14.1 Harness engineering (plugs into §12.1 / §12.1a)

- **The Anatomy of an Agent Harness** — the canonical LC definition.
  `Agent = Model + Harness`. "If you're not the model, you're the
  harness." Lists harness components: system prompts · tools/skills/
  MCPs · bundled infra (filesystem, sandbox, browser) · orchestration
  (subagents, handoffs, routing) · deterministic hooks/middleware
  (compaction, continuation, lint).
  https://www.langchain.com/blog/the-anatomy-of-an-agent-harness
- **Agent Engineering: A New Discipline** — frames agent engineering
  as distinct from ML engineering and traditional SWE.
  https://blog.langchain.com/agent-engineering-a-new-discipline/
- **Agent Frameworks, Runtimes, and Harnesses — oh my!** — vocabulary
  disambiguation; read before the other harness posts.
  https://blog.langchain.com/agent-frameworks-runtimes-and-harnesses-oh-my/
- **Improving Deep Agents with harness engineering** — a coding agent
  went from **Top 30 → Top 5 on Terminal Bench 2.0 by only changing
  the harness**, not the model. Proof that harness > model-picking.
  https://blog.langchain.com/improving-deep-agents-with-harness-engineering/
- **Better Harness: A Recipe for Harness Hill-Climbing with Evals** —
  data sourcing → experiment design → optimization → review.
  Autonomous loop: diagnose → experiment → validate. Human review
  catches token-wasting overfit that metrics miss. Concrete wins on
  Sonnet 4.6 + GLM-5: "use reasonable defaults," "don't re-ask for
  supplied details," bounded-exploration instructions.
  https://blog.langchain.com/better-harness-a-recipe-for-harness-hill-climbing-with-evals/
- **How Middleware Lets You Customize Your Agent Harness** — middleware
  pattern in deepagents as the main customization surface.
  https://blog.langchain.com/how-middleware-lets-you-customize-your-agent-harness/
- **Your harness, your memory** — memory is a harness concern; closed
  harnesses mean yielding control of your agent's memory to a third
  party.
  https://blog.langchain.com/your-harness-your-memory/
- **Deep Agents** — original post announcing the open-source
  reproduction of the Deep Research / Manus / Claude Code pattern.
  https://blog.langchain.com/deep-agents/
- **Deep Agents Deploy: an open alternative to Claude Managed Agents**
  — deployment layer for deepagents-built systems.
  https://blog.langchain.com/deep-agents-deploy-an-open-alternative-to-claude-managed-agents/

### 14.2 Context engineering (plugs into §12.3)

- **The rise of "context engineering"** — the term's popularization;
  framing post.
  https://blog.langchain.com/the-rise-of-context-engineering/
- **Context Engineering** — deeper LC treatment: "filling the context
  window with just the right information at each step of an agent's
  trajectory."
  https://blog.langchain.com/context-engineering-for-agents/
- **Context Management for Deep Agents** — three compression
  techniques deepagents ships: offloading large tool results,
  offloading large tool inputs, summarization at threshold.
  https://blog.langchain.com/context-management-for-deepagents/
- **How agents can use filesystems for context engineering** —
  filesystem as external memory; the canonical pattern.
  https://blog.langchain.com/how-agents-can-use-filesystems-for-context-engineering/
- **Autonomous context compression** — tools that let the *model*
  compress its own context window.
  https://blog.langchain.com/autonomous-context-compression/

### 14.3 Multi-agent systems (plugs into §12.2)

- **How and when to build multi-agent systems** — decision framework.
  Most useful first read; many problems don't need multi-agent.
  https://blog.langchain.com/how-and-when-to-build-multi-agent-systems/
- **Choosing the Right Multi-Agent Architecture** — four patterns:
  **subagents · skills · handoffs · routers**. Task-coordination vs.
  state-management trade-offs.
  https://www.blog.langchain.com/choosing-the-right-multi-agent-architecture/
- **Benchmarking Multi-Agent Architectures** — empirical comparison of
  the four patterns.
  https://blog.langchain.com/benchmarking-multi-agent-architectures/
- **Building Multi-Agent Applications with Deep Agents** — subagents
  as the fix for context bloat (filled context windows).
  https://www.blog.langchain.com/building-multi-agent-applications-with-deep-agents/
- **LangGraph: Multi-Agent Workflows** — the graph-based coordination
  primitive.
  https://blog.langchain.com/langgraph-multi-agent-workflows/

### 14.4 Memory & continual learning (plugs into §9, §12.3)

- **Memory for agents** — LC taxonomy of agent memory types.
  https://blog.langchain.com/memory-for-agents/
- **LangMem SDK for agent long-term memory** — Feb 2025 launch.
  Extraction from conversations, prompt-update optimization, behavior/
  fact/event memory. Storage-agnostic; LangGraph-native.
  https://blog.langchain.com/langmem-sdk-launch/
- **Launching Long-Term Memory Support in LangGraph** — namespaced
  key-value store with semantic search baked into the runtime.
  https://blog.langchain.com/launching-long-term-memory-support-in-langgraph/
- **Continual learning for AI agents** — explicitly covers SFT + RL
  (GRPO) as update mechanisms for agents. Direct bridge between
  sections 2–7 (training) and 9 (memory).
  https://blog.langchain.com/continual-learning-for-ai-agents/

### 14.5 Evals & benchmarks (plugs into §13)

- **Agent Evaluation Readiness Checklist** — practical checklist
  covering error analysis · dataset construction · grader design ·
  offline & online evals · production readiness.
  https://blog.langchain.com/agent-evaluation-readiness-checklist/
- **How we build evals for Deep Agents** — methodology: decide which
  behaviors matter → curate targeted evals → self-documenting evals
  with docstrings tagged by category (`tool_use`, etc.) → review
  traces → update coverage. "More evals ≠ better agents."
  https://blog.langchain.com/how-we-build-evals-for-deep-agents/
- **Evaluating Deep Agents: Our Learnings** — what LC learned from
  running the eval playbook on their own agents.
  https://blog.langchain.com/evaluating-deep-agents-our-learnings/
- **Evaluating Deep Agents CLI on Terminal Bench 2.0** — applied case
  study on a public benchmark.
  https://blog.langchain.com/evaluating-deepagents-cli-on-terminal-bench-2-0/
- **Evaluating Skills** — clearly-defined tasks as regression-catching
  benchmarks.
  https://blog.langchain.com/evaluating-skills/
- **Benchmarking Agent Tool Use** — four tool-use test environments
  for comparing LLM + prompting strategies.
  https://blog.langchain.com/benchmarking-agent-tool-use/
- **Agent Observability Powers Agent Evaluation** — traces are the
  substrate for eval and improvement. Three levels: single-step (run),
  full-turn (trace), multi-turn (thread). Start with trace-level.
  https://blog.langchain.com/agent-observability-powers-agent-evaluation/
- **You don't know what your agent will do until it's in production**
  — "inputs are infinite, behavior is non-deterministic, quality
  lives in the conversations themselves." Case for prod monitoring.
  https://blog.langchain.com/you-dont-know-what-your-agent-will-do-until-its-in-production/
- **Human judgment in the agent improvement loop** — where/how to
  keep humans in the labeling and review loop as you scale evals.
  https://blog.langchain.com/human-judgment-in-the-agent-improvement-loop/
- **On Agent Frameworks and Agent Observability** — why traces are
  critical regardless of framework.
  https://blog.langchain.com/on-agent-frameworks-and-agent-observability/

### 14.6 UX, deployment, production (adjacent to §11)

- **Introducing ambient agents** — agents that respond to ambient
  signals and only demand attention on high-signal events.
  https://blog.langchain.com/introducing-ambient-agents/
- **UX for Agents, Part 2: Ambient** — companion UX post.
  https://www.blog.langchain.com/ux-for-agents-part-2-ambient/
- **LangGraph Platform GA** — deploying long-running stateful agents.
  https://blog.langchain.com/langgraph-platform-ga/
- **LangChain and LangGraph v1.0 Milestones** — API stability point;
  useful anchor for version decisions.
  https://blog.langchain.com/langchain-langgraph-1dot0/
- **Previewing Interrupt 2026: Agents at Enterprise Scale** — where
  the enterprise-agent discourse is heading.
  https://blog.langchain.com/previewing-interrupt-2026-agents-at-enterprise-scale/

### 14.7 Customer case studies (concrete patterns in production)

Each case study is a real system's architecture; they compress years
of mistakes into ~15 minutes of reading.

- **How LangChain built its GTM Agent** — automatic feedback loop
  where every edit teaches the agent; weekly cron compacts memory to
  prevent bloat.
  https://blog.langchain.com/how-we-built-langchains-gtm-agent/
- **How Kensho built a multi-agent framework with LangGraph** —
  trusted financial data retrieval with multi-agent coordination.
  https://blog.langchain.com/customers-kensho/
- **How Exa built a Web Research Multi-Agent System with LangGraph
  and LangSmith**.
  https://blog.langchain.com/exa/
- **How Cleric's AI SRE leveled up with continuous learning through
  LangSmith** — direct continual-learning case study.
  https://blog.langchain.com/customers-cleric/
- **monday Service + LangSmith: Code-First Evaluation Strategy from
  Day 1**.
  https://blog.langchain.com/customers-monday/

### 14.8 Suggested LangChain-blog-only reading path (one weekend)

If you want a dense, LC-only curriculum:

1. *Agent Frameworks, Runtimes, and Harnesses* (vocabulary).
2. *The Anatomy of an Agent Harness* (definitions).
3. *Deep Agents* + *Context Management for Deep Agents* (reference
   harness).
4. *How and when to build multi-agent systems* → *Choosing the Right
   Multi-Agent Architecture* → *Benchmarking Multi-Agent
   Architectures*.
5. *Context Engineering* + *How agents can use filesystems for context
   engineering* + *Autonomous context compression*.
6. *Memory for agents* + *Your harness, your memory* + *Continual
   learning for AI agents*.
7. *Agent Evaluation Readiness Checklist* → *How we build evals for
   Deep Agents* → *Evaluating Deep Agents: Our Learnings* →
   *Evaluating Deep Agents CLI on Terminal Bench 2.0*.
8. *Better Harness: A Recipe for Harness Hill-Climbing with Evals*
   (ties training + eval + harness together).
9. *Improving Deep Agents with harness engineering* (the
   Terminal-Bench Top-30 → Top-5 case).
10. Two customer stories (GTM Agent + Cleric SRE) for grounding.

---

## 15. Codebases to read, replicate, modify

Reading production agent code is the fastest way to internalize harness
design. This section names the repos worth studying, grouped by
purpose, with specific read/replicate/modify tasks for each.

### 15.1 How to learn from an agent codebase

A repeatable method — the goal is *pattern recognition*, not coverage:

1. **Orient (15 min).** Read README + any architecture doc. Find the
   main entry point — usually the agent loop.
2. **Trace one turn (1 hr).** Pick a trivial prompt and walk every line
   the request touches: model call → tool dispatch → permission check
   → state update. Draw the sequence by hand.
3. **Map the harness primitives.** For every repo, locate and name:
   (a) the loop, (b) tool registry, (c) permission/safety layer,
   (d) context/compaction strategy, (e) state/memory backend,
   (f) subagent spawn, (g) extensibility hooks. If one is missing,
   note it — that's a design choice.
4. **Diff against another harness.** Compare primitive-for-primitive
   with one you've already read. The *difference* is where the design
   space lives.
5. **Modify one thing.** Add a hook. Swap a backend. Break compaction
   on purpose and watch the failure mode. Don't just read.
6. **Replicate the smallest primitive.** Port one pattern into a
   minimal Python script of your own. If you can't reproduce it
   standalone, you haven't understood it.

Budget: ~1 day per "read" repo, ~3 days per "replicate" repo. Don't
read more than two in parallel.

### 15.2 Harness design — read these first

Ordered from smallest/most readable to most production-hardened.

- **huggingface/smolagents** — "You can read the entire codebase and
  understand exactly what happens." Code-first loop (agent writes and
  executes Python to act). Start here if you want a single weekend.
  https://github.com/huggingface/smolagents
  *Task:* trace one full agent turn; swap the default tool set.

- **langchain-ai/deepagents** — Python, built on LangGraph. The seven-
  layer middleware stack is the clearest production example of harness
  primitives mapped onto code. Already deep-dived in §12.1a.
  https://github.com/langchain-ai/deepagents
  *Task:* add a custom middleware (e.g. a logging one) between
  `FilesystemMiddleware` and `SubAgentMiddleware`.

- **anthropics/claude-agent-sdk-python** — Official SDK backing Claude
  Code. Async `query` iterator, custom tools as in-process MCP servers,
  hook system. Smaller than you'd expect.
  https://github.com/anthropics/claude-agent-sdk-python
  *Task:* define one `PreToolUse` hook and one custom MCP tool; watch
  them fire on a real run.

- **anthropics/claude-agent-sdk-demos** — End-to-end examples: research
  agent, code reviewer, orchestrator. Great reference for how the SDK
  is meant to be composed.
  https://github.com/anthropics/claude-agent-sdk-demos

- **anthropics/claude-code** — The Claude Code CLI itself is now
  published. Large, but the loop + permission + compaction code is the
  highest-fidelity primary source for everything in §12.1.
  https://github.com/anthropics/claude-code
  *Task:* locate the `queryLoop`; map the five compaction stages in
  code; cross-check against arXiv:2604.14228.

- **openai/openai-agents-python** — Production replacement for Swarm.
  Handoffs, tracing, guardrails, voice agents. The canonical OpenAI
  take on what's minimal.
  https://github.com/openai/openai-agents-python
  *Task:* diff its "handoff" primitive against deepagents'
  `SubAgentMiddleware` — compare explicit-transfer vs. spawned-context.

- **openai/swarm** — The pre-Agents-SDK educational repo. Tiny, still
  widely read for pedagogy.
  https://github.com/openai/swarm

- **NousResearch/hermes-agent** — Memory-first, self-improving harness;
  five execution backends (local/Docker/SSH/Singularity/Modal). Most
  interesting for the skill-learning loop.
  https://github.com/nousresearch/hermes-agent
  *Task:* read the skill acquisition / memory layer — compare against
  deepagents' `MemoryMiddleware`.

- **HKUDS/OpenHarness** — Messaging-first open harness with built-in
  personal agent (Ohmo). Different surface from the above — worth
  reading as a counterpoint.
  https://github.com/HKUDS/OpenHarness

### 15.3 Coding agents — for SWE-bench / Terminal-Bench-style tasks

- **All-Hands-AI/OpenHands** (formerly OpenDevin) — Universal agent
  controller, sandboxed execution, multi-session. Series A backed;
  best-funded OSS coding agent. Read the sandbox layer especially.
  https://github.com/All-Hands-AI/OpenHands
- **princeton-nlp/SWE-agent** — Original research codebase behind the
  SWE-bench leaderboard runs. Tight coupling between the "Agent-
  Computer Interface" (ACI) and tool design — read this for ACI
  concepts.
  https://github.com/princeton-nlp/SWE-agent
- **Aider-AI/aider** — AI pair programmer. Minimal harness, aggressive
  prompt engineering, git-centric state. Good example of a specialized
  harness.
  https://github.com/Aider-AI/aider
- **anthropics/claude-code** (again — it is a coding agent).

*Replicate task:* port a minimal SWE-agent-style ACI (`view_file`,
`edit_file`, `run_tests`) onto deepagents. Run it on 10 SWE-bench
tasks locally.

### 15.4 Multi-agent frameworks

- **microsoft/autogen** — Conversation-based multi-agent framework.
  Large, mature, lots of patterns in one repo.
  https://github.com/microsoft/autogen
- **geekan/MetaGPT** — Role-based multi-agent (Product Manager,
  Architect, Engineer, QA). Canonical example of hierarchical
  coordination.
  https://github.com/geekan/MetaGPT
- **joaomdmoura/crewAI** — Role + process abstraction. Lightweight
  compared to AutoGen; good first read for multi-agent.
  https://github.com/joaomdmoura/crewAI
- **langchain-ai/langgraph** — The graph runtime underneath most LC
  multi-agent examples. Read the state-reducer / checkpointer code.
  https://github.com/langchain-ai/langgraph
- **langchain-ai/open_deep_research** — LC's open clone of Deep
  Research. Canonical pattern: planner → many parallel subagents →
  synthesizer.
  https://github.com/langchain-ai/open_deep_research

*Modify task:* take one of these, change the coordination structure
from centralized to peer-to-peer (or vice versa). Measure what breaks.

### 15.5 Evals & benchmarks — codebases to study

- **UKGovernmentBEIS/inspect_ai** — The UK AI Safety Institute's eval
  framework. Used for serious LLM evals: agentic tasks, coding,
  reasoning. Web viewer + VS Code extension.
  https://github.com/UKGovernmentBEIS/inspect_ai
- **UKGovernmentBEIS/inspect_evals** — 200+ community-contributed
  evaluations built on Inspect. Best corpus of *real* eval
  implementations.
  https://github.com/UKGovernmentBEIS/inspect_evals
- **openai/evals** — Framework + open registry. Reference for
  eval-as-data-spec.
  https://github.com/openai/evals
- **stanfordnlp/dspy** — Programming-over-prompting. Eval-driven
  prompt/pipeline optimization. Read for `dspy.Evaluate` and the GEPA
  trusted-monitor tutorial.
  https://github.com/stanfordnlp/dspy
- **princeton-nlp/SWE-bench** — The benchmark harness. Worth reading
  alongside Berkeley RDI's "How we broke it" post (§13.3) to see
  exactly *where* the exploits live in code.
  https://github.com/princeton-nlp/SWE-bench
- **sierra-research/tau2-bench** — Tool-agent-user interaction
  benchmark (airline / retail / telecom). Verifier design is
  particularly good.
  https://github.com/sierra-research/tau2-bench
- **StonyBrookNLP/appworld** — Long-horizon agent environment; the
  Apple LOOP paper (§7) used this.
  https://github.com/StonyBrookNLP/appworld
- **xlang-ai/OSWorld** — Full Ubuntu VM desktop agent benchmark.
- **web-arena-x/webarena** — Web-agent benchmark.

*Replicate task:* pick any simple eval in `inspect_evals`; port it to
`openai/evals` format. Feel the abstraction mismatch — that's the
design lesson.

### 15.6 Post-training & RL-on-LMs codebases

- **huggingface/trl** — SFT, DPO, PPO, GRPO. The library everyone
  builds on. Read `trl/trainer/grpo_trainer.py`.
  https://github.com/huggingface/trl
- **huggingface/open-r1** — Open reproduction of DeepSeek-R1. Data
  pipeline + training recipes.
  https://github.com/huggingface/open-r1
- **huggingface/alignment-handbook** — Zephyr/Tulu-style recipes.
  Read the YAML configs, not just the code.
  https://github.com/huggingface/alignment-handbook
- **huggingface/smol-course** — Tutorial + runnable scripts for SFT /
  DPO / preference alignment. Laptop-friendly.
  https://github.com/huggingface/smol-course
- **allenai/open-instruct** — AI2's production code for Tulu 3.
  Fullest open reference pipeline for modern post-training.
  https://github.com/allenai/open-instruct
- **volcengine/verl** — Production RL-for-LM framework; read the
  rollout engine + scheduler.
  https://github.com/volcengine/verl
- **OpenRLHF/OpenRLHF** — The other main production framework.
  https://github.com/OpenRLHF/OpenRLHF
- **WooooDyy/AgentGym-RL** — Multi-turn agent RL training loop.
  Already called out in §7.
  https://github.com/WooooDyy/AgentGym-RL

*Replicate task:* run a tiny GRPO on TRL with a synthetic verifiable
reward (e.g. "output must parse as JSON"). Then swap the verifier for
a code-exec one. This is the core RLVR skill.

### 15.7 Cookbooks & recipe repos (the fastest way to up-skill)

Cookbooks are short, runnable notebooks — each is 30–90 minutes and
teaches one pattern.

- **anthropics/claude-cookbooks** — Tool use, agent patterns, MCP,
  extended thinking, caching, RAG. Browsable at
  platform.claude.com/cookbook.
  https://github.com/anthropics/claude-cookbooks
- **anthropics/anthropic-cookbook** — Older repo; still useful for
  baseline patterns.
- **openai/openai-cookbook** — Canonical OpenAI how-to collection;
  the agents/ and evals/ directories are the relevant ones.
  https://github.com/openai/openai-cookbook
- **langchain-ai/langsmith-cookbook** — Eval + tracing recipes.

*Daily discipline:* work one cookbook notebook per day for 30 days.
Don't just read — run it, break it, rewrite one cell.

### 15.8 MCP — the protocol binding harnesses to tools

If you work on harnesses, you will end up reading MCP source.

- **modelcontextprotocol/modelcontextprotocol** — the spec.
- **modelcontextprotocol/servers** — reference server
  implementations; best way to learn the protocol is to read and run
  them.
  https://github.com/modelcontextprotocol/servers
- **modelcontextprotocol/python-sdk** / **typescript-sdk** — the
  client/server SDKs.

### 15.9 A 90-day learn-by-reading plan

A concrete schedule if you want one:

**Weeks 1–2 — Small, readable harnesses**
- Read `smolagents` end-to-end.
- Read `openai/swarm` (tiny).
- Work through 5 `anthropic-cookbooks` recipes.

**Weeks 3–4 — deepagents deep dive**
- Read all seven middleware modules.
- Write a custom middleware and a custom backend.
- Reproduce one `open_deep_research` pattern from scratch on top of
  deepagents.

**Weeks 5–6 — Production harnesses**
- Read `claude-agent-sdk-python` and write two demo agents.
- Read `openai/openai-agents-python`; diff against deepagents.
- Skim `anthropics/claude-code` main loop; cross-ref arXiv:2604.14228.

**Weeks 7–8 — Evals**
- Build a Hamel-Husain-style eval on one of your agents (§13.1).
- Port one `inspect_evals` eval into `openai/evals` format.
- Run the Berkeley RDI "Agent-Eval Checklist" on a benchmark of your
  choice.

**Weeks 9–10 — Coding-agent replication**
- Stand up OpenHands or SWE-agent locally.
- Run 10 SWE-bench Verified tasks end-to-end.
- Port a minimal ACI onto deepagents; compare pass rates.

**Weeks 11–12 — Post-training**
- Tiny GRPO run on TRL with a synthetic verifier.
- Swap the verifier for a code-exec reward.
- Wire into AgentGym-RL; run one multi-turn agent RL job.

**Week 13 — Synthesis**
- Write your own minimal harness from scratch (one file, <500 lines):
  loop + 3 tools + permission check + compaction + one subagent.
  If you can write this without copy-pasting, you've internalized the
  design space.

---

## 16. Deep-dives on the four best harnesses

We can't read each codebase cover-to-cover, but secondary material does
most of the work for us — architecture docs, reverse-engineering
writeups, the authors' own blog posts, and tutorial walkthroughs. For
each of the four harnesses worth studying, this section lists the
highest-signal reading (plus primary-source facts pulled directly from
those sources).

### 16.1 Claude Code (Anthropic)

The most analyzed harness in the industry. A leak of ~512K lines of
production TypeScript in early 2026 produced an explosion of
architecture writeups; combined with Anthropic's own posts and the
VILA-Lab design-space paper, there's more secondary material on Claude
Code than on any other agent system.

**Primary facts worth memorizing**

- **Five-layer architecture** (clean dependency direction, each layer
  only talks to the one below): Presentation (Ink-based React
  terminal renderer) → Application (QueryEngine + commands + hooks) →
  Domain (Tool interface, AppState, permission decisions) →
  Infrastructure (API clients, persistence, MCP) → Bootstrap (150+
  property session singleton).
- **Tool interface = core design pattern.** Every capability
  implements identity + Zod I/O schema + execution method +
  `checkPermissions()`. Enables 50+ tools to plug in without
  modifying the query loop. MCP tools plug in identically.
- **Cascading permission gates (4):** static rules (`settings.json`) →
  tool's own `checkPermissions()` → permission mode (`bypass/auto/
  default/plan`) → user prompt. Safe ops auto-approve, destructive
  ones require consent.
- **Five-layer compaction:** (1) Tool Result Budget — persist >20K
  char results to disk; (2) Snip — clear old tool results, preserve
  structure; (3) Microcompaction — remove old tool results per turn;
  (4) Context Collapse — model-side projection; (5) Autocompact —
  triggered ~13K below window, tries session-memory compression first,
  falls back to forked-agent full summary.
- **Async generators as the core primitive** — streaming events, not
  blocking responses.
- **Session Memory file (~12K tokens)** maintained by a background
  process with sections: current state · task spec · errors ·
  learnings · worklog. Source for lightweight SM-Compact.
- **Multi-strategy tool execution:** read-only tools batch in
  parallel; write tools execute sequentially; permission checks
  inline.
- **Post-compaction cleanup:** invalidates six caches (microcompact
  state, context collapse, memory files, message cache, classifier
  approvals, speculative shell checks) to prevent state corruption.

**Primary-source reading (ordered)**

1. **Anthropic: Harness design for long-running application
   development** — already in §12.1 sources. Start here.
   https://www.anthropic.com/engineering/harness-design-long-running-apps
2. **Dive into Claude Code: The Design Space of Today's and Future AI
   Agent Systems** (arXiv:2604.14228) — the academic treatment;
   VILA-Lab. The most complete single document.
   https://arxiv.org/html/2604.14228v1 ·
   repo: https://github.com/VILA-Lab/Dive-into-Claude-Code
3. **Zain Hasan — Inside Claude Code: An Architecture Deep Dive** —
   clearest engineer-friendly walkthrough of the five layers, tool
   interface, permission cascade, and compaction.
   https://zainhas.github.io/blog/2026/inside-claude-code-architecture/

**Reverse-engineering analyses (from the leak)**

- **WaveSpeedAI: Claude Code architecture Deep Dive — What the Leaked
  Source Reveals** — tool system, query engine, multi-agent patterns,
  context compression.
  https://wavespeed.ai/blog/posts/claude-code-architecture-leaked-source-deep-dive/
- **bits-bytes-nn: Claude Code Architecture Analysis** — compact
  annotated view.
  https://bits-bytes-nn.github.io/insights/agentic-ai/2026/03/31/claude-code-architecture-analysis.html
- **Han HELOIR YAN (Data Science Collective): Everyone Analyzed
  Claude Code's Features. Nobody Analyzed Its Architecture.**
  https://medium.com/data-science-collective/everyone-analyzed-claude-codes-features-nobody-analyzed-its-architecture-1173470ab622
- **JIN System Architect: Claude Code Runtime Architecture — A Deep
  Dive for Engineers**.
  https://medium.com/jin-system-architect/claude-code-runtime-architecture-a-deep-dive-for-engineers-who-actually-want-to-understand-whats-31e859c2038e
- **redreamality: Claude Code Leak — A Deep Dive into Anthropic's AI
  Coding Agent Architecture**.
  https://redreamality.com/blog/claude-code-source-leak-architecture-analysis/
- **Varonis: A Look Inside Claude's Leaked AI Coding Agent** —
  security-oriented perspective.
  https://www.varonis.com/blog/claude-code-leak
- **ComeOnOliver/claude-code-analysis** (GitHub) — comprehensive
  reverse-engineering repo; module/pattern index.
  https://github.com/ComeOnOliver/claude-code-analysis
- **liz-in-tech/open-claude-code** — mirrors Claude Code source
  (v2.1.88) for study.
  https://github.com/liz-in-tech/open-claude-code

**Epsilla: 12 Reusable Agentic Harness Design Patterns from Claude
Code** (§12.1a has the full breakdown).
https://www.epsilla.com/blogs/2026-04-18-deep-dive-12-reusable-agentic-harness-design-patte

### 16.2 Hermes Agent (Nous Research)

The most interesting harness *as a research artifact*: memory-first,
self-improving, with a closed learning loop that turns experience
into reusable skills. Released Feb 25 2026; MIT; ~95.6k stars by mid-
April.

**Primary facts worth memorizing**

- **Scale:** 47 registered tools across 19 toolsets · 18 platform
  adapters (Telegram, Discord, Slack, WhatsApp, Signal, Matrix, …) ·
  6 terminal execution backends (local, Docker, SSH, Modal, Daytona,
  Singularity).
- **Three entry points → one core:** CLI, Gateway, ACP → `AIAgent`.
- **The loop:** "User input → `HermesCLI.process_input()` →
  `AIAgent.run_conversation()` → `prompt_builder.build_system_prompt()`
  → `runtime_provider.resolve_runtime_provider()` → API call →
  `tool_calls?` → `model_tools.handle_function_call()` → loop → final
  response → display → save to SessionDB."
- **`AIAgent` (`run_agent.py`, ~10,700 lines)** handles provider
  selection, prompt construction, tool execution, retries, fallback,
  callbacks, compression, persistence — across three API modes.
- **Skill system:** skills are *context injected during prompt
  assembly* — not runtime plugins. Bundled vs optional; per-platform
  config (`skills_config.py`); cron jobs can attach skills + scripts
  as context.
- **Memory:** pluggable provider system. `Memory manager
  orchestration` + `Memory provider ABC`; conversation-scoped files
  (`MEMORY.md`, `USER.md`) feed prompt assembly.
- **Persistence:** SQLite session storage with FTS5 full-text search;
  lineage tracking + atomic writes.

**Primary-source reading**

1. **Official architecture doc** —
   https://hermes-agent.nousresearch.com/docs/developer-guide/architecture
2. **Full docs site** — https://hermes-agent.nousresearch.com/docs/
3. **README** (GitHub) — https://github.com/nousresearch/hermes-agent

**Deep-dives & guides**

- **Inside Hermes Agent: A Deep Dive into Its Technical Architecture**
  (Hecate, Medium, April 2026). Best single secondary source.
  https://medium.com/@hecate_he/inside-hermes-agent-a-deep-dive-into-its-technical-architecture-175dcf67d671
- **Lushbinary: Hermes Agent Developer Guide — Setup & Self-Improving
  AI**.
  https://lushbinary.com/blog/hermes-agent-developer-guide-setup-skills-self-improving-ai/
- **NxCode: What Is Hermes Agent? Complete Guide to the Self-
  Improving AI (2026)**.
  https://www.nxcode.io/resources/news/hermes-agent-complete-guide-self-improving-ai-2026
- **DataCamp: Nous Research Hermes Agent — Setup and Tutorial Guide**.
  https://www.datacamp.com/tutorial/hermes-agent
- **mudrii/hermes-agent-docs** — third-party comprehensive docs mirror
  (v0.2.0).
  https://github.com/mudrii/hermes-agent-docs
- **Hermes Agent Orange Book** (alchaincyf/hermes-agent-orange-book) —
  "橙皮书系列" — Chinese-language tutorial book (machine-translate if
  needed; the code examples are valuable regardless).
  https://github.com/alchaincyf/hermes-agent-orange-book
- **Petronella: 2026 Release Tracker** — follows v0.9 → v0.10 → v0.11
  feature changes.
  https://petronellatech.com/blog/hermes-agent-ai-guide-2026
- **DEV Community: Hermes Agent Review** (Apr 2026).
  https://dev.to/tokenmixai/hermes-agent-review-956k-stars-self-improving-ai-agent-april-2026-11le
- **heyuan110: Hermes Agent Hands-On** — review from actual daily use.
  https://www.heyuan110.com/posts/ai/2026-04-14-hermes-agent-guide/

**What to study specifically:** the closed-loop learning pathway
(solve task → write skill doc → store outcome → retrieve next time).
This pattern is missing from every other harness in this note.

### 16.3 OpenClaw — with Pi Agent at its core

OpenClaw is the messaging-first harness (~354k stars). The twist: its
agent runtime is a separately-designed minimal agent called **Pi**
(by Armin Ronacher / pi-mono). OpenClaw embeds Pi's `AgentSession` via
`createAgentSession()` — not via subprocess or RPC.

**Primary facts — Pi (the minimal core)**

- **Radical minimalism:** "the shortest system prompt of any agent
  that I'm aware of and it only has four tools: **Read, Write, Edit,
  Bash**." — Armin Ronacher.
- **Design thesis:** "if you want the agent to do something that it
  doesn't do yet, you ask the agent to extend itself." Celebrates
  *code writing and running code* as the extensibility mechanism.
- **Session extensions persist state** — enables hot-reloading and
  branching workflows.
- **No MCP integration** — deliberate. Ronacher's critique: MCP
  tools "need to be loaded into the system context on session start.
  That makes it very hard to impossible to fully reload."
- **Session portability across providers** is a stated design goal.

**Primary facts — OpenClaw (the platform wrapping Pi)**

- **Embeds Pi via `createAgentSession()`** — in-process, not RPC.
- **Entry:** `run.ts` → `runEmbeddedPiAgent()` with modules for
  single-attempt logic, parameters, response payloads, vision image
  injection, and attempt result types.
- **Tools = Pi base (`codingTools`: read/bash/edit/write) + OpenClaw
  additions**: messaging, browser, canvas, sessions, cron, gateway;
  channel-specific tools for Discord / Telegram / Slack / WhatsApp.
  Filtering by profile / provider / agent / group / sandbox.
- **OpenClaw customizes:** replaces `bash` with `exec/process`;
  customizes `read/edit/write` for sandbox use.
- **Image injection is prompt-local**: load image refs from the
  *current* prompt and pass via `images` for that turn only —
  no re-scanning older turns.
- **Session IDs are stable and chosen by OpenClaw.** Legacy Pi/Tau
  session folders are deliberately ignored.

**Primary-source reading**

1. **Armin Ronacher — "Pi: The Minimal Agent Within OpenClaw"**
   (Jan 31 2026). Read this first. The design philosophy is where the
   real lessons live.
   https://lucumr.pocoo.org/2026/1/31/pi/
2. **OpenClaw Pi Integration Architecture** (official docs).
   https://docs.openclaw.ai/pi · mirror:
   https://github.com/openclaw/openclaw/blob/main/docs/pi.md
3. **OpenClaw Agent Runtime concept docs** —
   https://openclaw-ai.com/en/docs/concepts/agent

**Deep-dives & guides**

- **nader.substack: How to Build a Custom Agent Framework with Pi —
  The Agent Stack Powering OpenClaw**. The clearest "build your own
  with the same bones" article.
  https://nader.substack.com/p/how-to-build-a-custom-agent-framework
- **All Things Open: OpenClaw — Anatomy of a viral open-source AI
  agent** (already cited in §12.1a).
  https://allthingsopen.org/articles/openclaw-viral-open-source-ai-agent-architecture
- **robotpaper.ai: Reference Architecture — OpenClaw (Feb 2026,
  Opus 4.6)** — the best single architecture breakdown.
  https://robotpaper.ai/reference-architecture-openclaw-early-feb-2026-edition-opus-4-6/
- **Zylon: What Is OpenClaw? A Practical Guide to the Agent Harness
  Behind the Hype**.
  https://www.zylon.ai/resources/blog/what-is-openclaw-a-practical-guide-to-the-agent-harness-behind-the-hype
- **knightli: OpenClaw and Agent Harness — Why It Looks Like AGI**
  (April 2026). Enterprise framing.
  https://www.knightli.com/en/2026/04/10/openclaw-agent-architecture-enterprise-ai/
- **DeepWiki: Agent Harness Plugins** — `openclaw/docs` section 5.4.
  https://deepwiki.com/openclaw/docs/5.4-agent-harness-plugins
- **vallettasoftware: OpenClaw Architecture & Setup Guide (2026)**.
  https://vallettasoftware.com/blog/post/openclaw-2026-guide
- **Advith Aatreya (Medium): OpenClaw Models Explained (2026) —
  Architecture, Safety & Use Cases**.
  https://medium.com/@aatreyaadvith/openclaw-models-explained-2026-architecture-safety-use-cases-bdcfc1f3f8f4
- **Raspberry Pi: Turn your Raspberry Pi into an AI agent with
  OpenClaw** — delightfully concrete tutorial with the real SDK.
  https://www.raspberrypi.com/news/turn-your-raspberry-pi-into-an-ai-agent-with-openclaw/
- **pchojecki (Medium): How I turned a Raspberry Pi 4 into an AI
  Agent with OpenClaw**.
  https://pchojecki.medium.com/how-i-turned-a-raspberry-pi-4-into-an-ai-agent-with-openclaw-ba4f58d39f98

**What to study specifically:** (1) the *Pi base* — 4-tools-and-a-
system-prompt discipline; (2) how OpenClaw wraps Pi with messaging
integrations and sandbox policies *without* modifying the core loop.
This is the cleanest separation-of-concerns in the harness world.

### 16.4 LangChain deepagents

The best *library* representation of the four patterns (planning tool +
subagents + filesystem + detailed prompt). Python, on LangGraph,
readable end-to-end in ~1 day. See §12.1a + §14.1 for the primitives.

**Primary facts worth memorizing**

- **Library + CLI + ACP:** Deep Agents SDK (library), Deep Agents CLI
  (terminal coding agent built on SDK), ACP integration (Agent Client
  Protocol — e.g. Zed editor connector).
- **Built-in tools:** `write_todos` (planning), virtual filesystem
  (`ls`, `read_file`, `write_file`, `edit_file`), `execute` (sandbox
  shell), `task` (subagent spawn).
- **Seven-layer middleware** (covered in §12.1a): TodoList · Memory ·
  Skills · Filesystem · SubAgent · Summarization · PatchToolCalls.
- **Four filesystem backends:** StateBackend (ephemeral, per-run) ·
  StoreBackend (persistent across threads via LangGraph Store) ·
  FilesystemBackend (disk) · CompositeBackend (route per-path,
  e.g. `/memories/` → Store, `/` → disk).
- **Summarization trigger at 85% context** with argument truncation
  first via `TruncateArgsSettings`; offloads older content to
  `/conversation_history/{thread_id}.md`.
- **`recursion_limit=1000`** for deep tool-call chains.
- **When to use it (per official docs):** complex multi-step tasks
  needing planning · large context needing filesystem mgmt · diverse
  backends (in-memory/disk/sandbox) · subagent delegation · cross-
  conversation memory · human-in-loop for sensitive ops. For simpler
  agents, LangChain recommends stock `create_agent` or custom
  LangGraph.

**Primary-source reading**

1. **Official Deep Agents Overview docs** —
   https://docs.langchain.com/oss/python/deepagents/overview
2. **GitHub README** — https://github.com/langchain-ai/deepagents
3. **DeepWiki: Architecture Overview** —
   https://deepwiki.com/langchain-ai/deepagents/1.3-architecture-overview
4. **DeepWiki: Examples and Use Cases** —
   https://deepwiki.com/langchain-ai/deepagents/8-examples-and-use-cases
5. **LangChain landing page** —
   https://www.langchain.com/deep-agents

**LangChain blog posts (already cross-referenced in §14)**

- Deep Agents (announcement) ·
  Context Management for Deep Agents ·
  How we build evals for Deep Agents ·
  Evaluating Deep Agents: Our Learnings ·
  Evaluating Deep Agents CLI on Terminal Bench 2.0 ·
  Building Multi-Agent Applications with Deep Agents ·
  Deep Agents Deploy (Claude Managed Agents alternative) ·
  Improving Deep Agents with harness engineering (Top 30 → Top 5) ·
  How Middleware Lets You Customize Your Agent Harness.

**Tutorials & walkthroughs**

- **DataCamp: LangChain's Deep Agents — A Guide With Demo Project**.
  Best hands-on intro.
  https://www.datacamp.com/tutorial/deep-agents
- **Krish Naik substack: Building Deep Agents with LangChain — A
  Complete Hands-On Tutorial**.
  https://krishcnaik.substack.com/p/building-deep-agents-with-langchain
- **Michiel Horstman (Medium): Deep Agents in LangChain — Building
  the Next Generation of Autonomous AI Systems with LangGraph**.
  https://michielh.medium.com/deep-agents-in-langchain-building-the-next-generation-of-autonomous-ai-systems-with-langgraph-3787b67e1805
- **NVIDIA Technical Blog: How to Build Deep Agents for Enterprise
  Search with NVIDIA AI-Q and LangChain**. Production case study.
  https://developer.nvidia.com/blog/how-to-build-deep-agents-for-enterprise-search-with-nvidia-ai-q-and-langchain/

**What to study specifically:** the middleware pattern itself.
deepagents' strongest contribution is that the seven middleware layers
*are* the harness — each one maps cleanly to a Claude Code / Hermes /
Pi primitive. Reading them side-by-side is the fastest way to build a
mental model of the design space.

### 16.5 Cross-harness comparison — structured study

A concrete exercise: make a table for each primitive below and fill
each column by reading the relevant source. When you finish the table,
you have internalized the harness design space.

| Primitive | Claude Code | Hermes Agent | OpenClaw (Pi) | deepagents |
|-----------|-------------|--------------|---------------|------------|
| Loop location | `QueryEngine` | `AIAgent.run_conversation()` | `runEmbeddedPiAgent()` / Pi `AgentSession` | `create_deep_agent` → `CompiledStateGraph` |
| Base tool count | 50+ (pluggable) | 47 across 19 toolsets | **4** (Read/Write/Edit/Bash) + OpenClaw add-ons | Planning + FS + `execute` + `task` |
| Extensibility | MCP · Plugins · Skills · Hooks | Skills (context-injected) + Platform adapters | Agent writes its own extensions | Middleware + backends |
| Permission model | Rules → tool `checkPermissions` → mode → prompt | Sandbox per backend | Sandbox policies at gateway | Declarative rules + interrupts |
| Context compression | 5 layers + Session Memory file | `Memory manager` + `MEMORY.md` | Prompt-local only (minimal) | Auto at 85% + arg truncate |
| Subagents | `AgentTool` w/ sidechain transcripts | Via skill scripts | Agent writes helpers | `SubAgentMiddleware` + `task` |
| Memory storage | Append-only JSONL + CLAUDE.md hierarchy | SQLite + FTS5; `MEMORY.md`/`USER.md` | Session-scoped persistence | StateBackend / StoreBackend / FilesystemBackend / CompositeBackend |
| Multi-surface | CLI + headless + SDK + IDE (one `queryLoop`) | CLI + Gateway + ACP → one `AIAgent` | Messaging (Telegram/Slack/Discord/WA) + CLI | Library + CLI + ACP |
| License | Proprietary | MIT | MIT | MIT |
| Language | TypeScript | Python | TypeScript | Python |

### 16.6 Suggested harness study sequence

1. **Read Ronacher's Pi post** (one sitting) — the clearest design-
   philosophy statement in the field; sets a minimalist baseline.
2. **Read deepagents official overview + DeepWiki architecture** —
   Python, small, clean. Maps primitives to named middleware.
3. **Read Zain Hasan's Inside Claude Code** — the five layers + tool
   interface + permissions + five-layer compaction.
4. **Read Anthropic's own harness-design post + arXiv:2604.14228** —
   the why behind the what.
5. **Read Inside Hermes Agent (Hecate, Medium) + official Hermes
   architecture doc** — the skill-learning loop and memory pluggability.
6. **Read OpenClaw's Pi integration doc + robotpaper.ai reference
   architecture** — how a minimal core gets wrapped for a messaging OS.
7. **Fill in the cross-harness comparison table (§16.5) from memory.**
   If you can, you're done. If not, re-read the gaps.

---

## Principles worth internalizing (cross-cutting)

- The model is commodity. The harness, eval, and environment are the
  product.
- Separate **generation** from **evaluation** — agents confidently
  praise mediocre work.
- Context is a scarce resource. Design for forgetting, not just
  remembering.
- Most agent benchmarks are exploitable. Assume your reward function is
  wrong until proven otherwise.
- Multi-agent ≠ better. Coordination overhead often exceeds gains
  unless structure matches the task.
- Error analysis > infrastructure. 60–80% of eval effort is manual
  trace reading, not building dashboards.
- Binary > Likert. Force decisions.
- RLVR beats RLHF wherever you can write a verifier. Preference data is
  the fallback, not the default.
- Assumptions expire. Every harness component encodes an assumption
  about what the model can't do solo — stress-test and prune quarterly.
