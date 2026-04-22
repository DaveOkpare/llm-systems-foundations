# Post-Training & Agents — Buildable Projects

Companion to `post-training-and-agents.md`. The study guide teaches the
*what* and *why*; this file is the *build*. Projects are ordered
foundations → full agent stack → capstones. Each names a time budget,
what you'll learn, and a reference codebase to diff against. All
Python/PyTorch unless noted.

Rule of thumb: if you can't rebuild a primitive standalone in <500
lines, you don't understand it. Most of these projects exist to prove
you do.

---

## Tier 1 — Post-training fundamentals (weekend each)

### 1. SFT from scratch on a tiny model — *weekend*
Take SmolLM3-135M or Qwen3-0.6B. Write a single-file SFT loop: tokenize
a small instruction dataset (Alpaca or Tulu subset), mask prompts in
the loss, train 1 epoch, eval on held-out. **You'll learn** how
instruction tuning differs from pretraining (prompt masking, `<|im_*|>`
templating, learning-rate scale). Reference: HuggingFace TRL's
`SFTTrainer` source, but *don't import it*.

### 2. DPO in one file — *weekend*
Implement Direct Preference Optimization end-to-end on an
already-SFT'd model using the `anthropic/hh-rlhf` preference dataset.
Write the DPO loss from the paper (log-ratio of chosen vs rejected,
with β). **You'll learn** why DPO replaced PPO in practice — and
exactly how the loss implicitly models a reward. Diff against
`trl/trainer/dpo_trainer.py`.

### 3. Tiny GRPO with a synthetic verifier — *weekend*
Pick a verifiable task: "model output must parse as valid JSON matching
schema X." Implement GRPO — sample G completions per prompt, score each
with the verifier, compute group-relative advantages, no critic. Track
KL to reference policy. **You'll learn** the RLVR loop without
infrastructure distractions. Reference: DeepSeek-R1 paper §2, and
`trl/trainer/grpo_trainer.py`.

### 4. Swap the verifier for code execution — *weekend*
Keep the GRPO loop from (3). Swap the JSON verifier for a Python-
execution reward: given a problem, the model emits code, you run it in
a sandbox, reward is unit-test pass count. **You'll learn** the actual
RLVR pattern used by DeepSWE and Agent-RLVR. Use
`subprocess`+`seccomp` or Docker for the sandbox.

### 5. Rejection-sampling SFT pipeline — *weekend*
For each prompt, sample N=16 completions from a base model, score each
with a reward model or verifier, keep the best, SFT on the result.
**You'll learn** why most 2024–26 production recipes (Llama 3, Qwen3,
Tulu 3) use this before DPO. Reference: AI2's `open-instruct`.

---

## Tier 2 — Agent harness primitives (1 week each)

Every item in this tier corresponds to one row in the cross-harness
comparison table (§16.5 of the study guide). Build them all and you
own the design space.

### 6. Minimal 4-tool harness (Pi-style) — *1 week*
Write a harness with exactly four tools: `read`, `write`, `edit`,
`bash`. Shortest system prompt you can make work. Loop is a plain
while-loop: assemble context → call model → dispatch tool → append
result → repeat until `stop`. Target: <500 lines. **You'll learn**
that most of an agent is decisions about what *not* to include.
Reference: Armin Ronacher's Pi post and `openclaw/pi-mono`.

### 7. Cascading permission system — *1 week*
Bolt a 4-gate permission cascade onto project (6): static rules
(JSON config) → tool-level `check_permissions()` → permission mode
(`default/auto/plan/bypass`) → user prompt. Add a deny-first
rule evaluator. **You'll learn** Claude Code's actual safety model.
Reference: Zain Hasan's Inside Claude Code deep-dive; the CVE-Bench
reward-hacking cases in §13.3 for *what happens when you skip this*.

### 8. Filesystem-as-memory + `AGENTS.md` loader — *1 week*
Add a virtual filesystem (back it with disk; optionally pluggable to
a `StoreBackend`). On agent start, walk up the directory tree and load
the nearest `AGENTS.md` / `CLAUDE.md` into the system prompt.
Implement `ls/read/write/edit/glob/grep`. **You'll learn** why every
modern harness treats the filesystem as external memory. Reference:
deepagents' `FilesystemMiddleware`.

### 9. Five-stage compaction pipeline — *1 week*
Implement Budget → Snip → Microcompact → Context Collapse →
Autocompact. Each stage should be idempotent and have a configurable
threshold. Trigger Autocompact at 85% context; fork a summarizer
agent for full compression. Maintain a Session Memory file (~12K
tokens) with labeled sections. **You'll learn** why no single
compaction strategy is enough. Reference: arXiv:2604.14228 §
compaction; deepagents `SummarizationMiddleware`.

### 10. SubAgent spawn with sidechain transcripts — *1 week*
Add a `spawn_subagent(task, tools)` primitive. Spawned agent gets an
isolated context window and its own permission scope; returns only a
summary to parent. Full subagent history lives in a sidechain file.
Add "bubble mode" so subagent permission requests escalate to parent.
**You'll learn** context-isolation as the main defense against
context bloat. Reference: Claude Code's `AgentTool` pattern.

### 11. MCP server + client pair — *1 week*
Expose 3 tools as an MCP server (Python SDK). Write a client that
discovers, lists, and invokes them. Wire the client into harness (6)
as an extensibility path. **You'll learn** the protocol your harness
will speak to every tool for the next decade. Reference:
`modelcontextprotocol/python-sdk` and `modelcontextprotocol/servers`.

### 12. Lifecycle hooks (deterministic middleware) — *1 week*
Add 8 hook events: `PreToolUse`, `PostToolUse`, `PreModelCall`,
`PostModelCall`, `OnCompact`, `OnSubagentSpawn`, `OnSubagentReturn`,
`OnError`. Use them to enforce a non-negotiable rule (e.g. "format
all written code before commit"). **You'll learn** why deterministic
middleware is load-bearing — "if a task must happen, you cannot rely
on the LLM to remember to do it." Reference: Epsilla's Pattern 12.

---

## Tier 3 — Composed systems (1–2 weeks each)

### 13. Rebuild deepagents middleware stack — *2 weeks*
Combine projects 8–12 into a 7-layer middleware stack mirroring
deepagents: TodoList · Memory · Skills · Filesystem · SubAgent ·
Summarization · PatchToolCalls. Ship it as a library with a
`create_deep_agent()` factory. **You'll learn** that middleware is
harness design made modular. Diff against
`github.com/langchain-ai/deepagents`.

### 14. Four multi-agent patterns + benchmark — *2 weeks*
Implement **subagents · skills · handoffs · routers** as four
distinct coordination patterns. Write one task that favors each
(plus a "common" task). Measure: tokens, latency, success rate.
**You'll learn** empirically when multi-agent helps and when it
burns tokens. Reference: LangChain's "Choosing the Right Multi-Agent
Architecture" + "Benchmarking Multi-Agent Architectures."

### 15. Agentic memory: Storage → Reflection → Experience — *2 weeks*
Implement the three-stage memory evolution from the 2026 memory
survey. Storage: persist raw trajectories. Reflection: LLM refines
them into lessons. Experience: LLM abstracts patterns across
lessons into general principles. Feed all three back into prompt
assembly. **You'll learn** where pure RAG ends and *agent memory*
begins. Reference: "From Storage to Experience" survey; A-Mem paper.

### 16. Debate-based multi-agent for reasoning — *1 week*
Two instances of the same model argue opposite positions on a claim;
a judge agent picks a winner. Measure accuracy vs. single-agent CoT
on GSM8K and MMLU. **You'll learn** the debate paradigm and its
ceiling — sometimes useful, often not. Reference: Du et al. "Improving
Factuality and Reasoning via Multiagent Debate."

### 17. Hermes-style skill-learning loop — *2 weeks*
After every task, agent writes a `skills/<name>.md` file describing
what worked. On subsequent tasks, a retrieval step surfaces relevant
skills into the prompt. Weekly cron deduplicates and compresses the
skill library. **You'll learn** the closed experience-to-skill loop
missing from most harnesses. Reference: Nous Research's Hermes Agent
skill system + architecture docs.

### 18. Planner / Generator / Evaluator (Claude-Code pattern) — *2 weeks*
Three-agent pipeline on a long-horizon build task (e.g. "make a small
React app with feature spec X"). Planner writes spec; Generator
implements; Evaluator uses Playwright MCP to grade the running app
against a rubric. Implement "sprint contracts." Measure cost vs solo
agent. **You'll learn** why Anthropic runs 6 hours and \$200 instead
of 20 min and \$9 on hard tasks. Reference: Anthropic's harness-design
post.

---

## Tier 4 — Agent RL / long-horizon (2–3 weeks each)

### 19. Multi-turn tool-use RL (LOOP-style) — *2 weeks*
Take project 4 (code-exec GRPO) and turn it multi-turn: agent writes
code, sees stderr, revises, re-runs. Reward is final test-pass. Use
LOOP (no critic, no value network) from Apple's paper. Evaluate on a
tiny AppWorld subset. **You'll learn** why multi-turn RL blows up
naïve single-turn recipes. Reference: arXiv:2502.01600.

### 20. Reproduce a slice of AgentGym-RL — *3 weeks*
Pick one AgentGym-RL environment. Stand up the full loop: env
rollout → reward → GRPO update → eval. Train a 3B-ish model. Match
reported numbers within 20%. **You'll learn** that 70% of agent
training is environment + verifier engineering, not the model loop.
Reference: `github.com/WooooDyy/AgentGym-RL`.

### 21. Mini SWE-agent + RL on SWE-bench subset — *3 weeks*
Implement the SWE-agent ACI (`view_file`, `edit_file`, `run_tests`,
`diff`) on top of harness (13). Use Agent-RLVR-style rewards: test-
pass signal with guidance. Train on 100 SWE-bench Verified tasks,
eval on 50 held out. Target: beat base-model zero-shot by ≥10pp.
Reference: Agent-RLVR + SWE-agent papers.

### 22. Test-time curriculum agent — *2 weeks*
Given a test prompt, agent retrieves related training tasks from a
corpus, RL-finetunes on them (LoRA), then solves the test prompt.
**You'll learn** the TTC-RL trick from arXiv:2510.04786 — big pass@1
lifts on AIME25. Reference: the paper's code release if available.

### 23. Self-improving-at-test-time agent — *1 week*
Agent (1) detects hard samples via self-uncertainty, (2) generates
synthetic similar samples, (3) fine-tunes on them at test time.
Measure gain on a held-out set. **You'll learn** the self-data-aug
pattern from arXiv:2510.07841 (5.48% gains with 68× less data).

---

## Tier 5 — Evals & benchmarks (1–2 weeks each)

These are the projects that matter most for the Anthropic Agentic
Systems role. Do them even if you skip Tier 4.

### 24. Run the Hamel Husain workflow on your own agent — *1 week*
Pick any agent you've built. Collect 100 real traces. Open-code 50
(pen + paper OK). Build a failure taxonomy. Quantify per category.
Build a transition matrix: last-success → first-failure. **You'll
learn** that error analysis is the only eval that matters. Reference:
`hamel.dev/blog/posts/evals-faq/`.

### 25. LLM-as-judge with TPR/TNR validation — *1 week*
Pick one subjective failure mode from (24). Build an LLM-as-judge
prompt. Hand-label 100 examples. Iterate the judge prompt against
disagreements. Report TPR and TNR on 30 held-out examples. Deploy
only if both >80%. **You'll learn** that "LLM-as-judge" without
validation is worse than no eval. Reference: Hamel FAQ § judge
construction.

### 26. Port an Inspect eval to `openai/evals` — *1 week*
Pick one eval from `UKGovernmentBEIS/inspect_evals`. Port it to
`openai/evals` format. Compare scores on the same model. **You'll
learn** the two dominant eval abstractions and where they diverge.

### 27. Publish a mini agentic benchmark that passes ABC — *2 weeks*
Design 30 tasks in a narrow domain (e.g. log-parsing agents or
SQL-writing agents). For each task, write both an outcome-validity
check *and* a process-validity check. Run the UIUC Agentic Benchmark
Checklist against your benchmark. Fix everything it flags. Publish on
GitHub. **You'll learn** the benchmark-building discipline most papers
skip. Reference: arXiv:2507.02825 and the checklist site.

### 28. Adversarial probes — break your own benchmark — *1 week*
Run Berkeley RDI's Agent-Eval Checklist against (27). Try the 7
vulnerability categories: missing isolation, exposed answers, unsafe
`eval()`, unsanitized LLM judges, weak matching, broken eval logic,
compromised artifacts. Report which probes succeed and fix each.
**You'll learn** that "my benchmark is trustworthy" is a hypothesis
until proven. Reference: §13.3.

### 29. Production observability pipeline — *1 week*
Wire one of your agents to LangSmith or a custom trace store. Build a
dashboard for: success rate by failure category, tool-call hotspots,
tokens-per-successful-task, latency p50/p95. Sample 1% of prod traffic
into async LLM-as-judge. **You'll learn** that "you don't know what
your agent will do until it's in production." Reference: LangChain's
agent-observability posts.

---

## Tier 6 — Capstones

### 30. Your own production harness (deepagents parity) — *1 month*
Complete harness: loop · 4-gate permissions · 5-stage compaction ·
pluggable backends (State / Store / Disk / Composite) · subagents w/
sidechain · MCP · lifecycle hooks · skill library (Hermes-style) ·
planner/evaluator split. Deploy behind a CLI and an HTTP API.
Document every design decision in an `ARCHITECTURE.md`. **You've
built the full design space from §16.5.**

### 31. Ambient coding agent: Slack + GitHub + cron — *2–3 weeks*
Harness (30) watching: GitHub issues, Slack threads, a cron schedule.
Ambient mode: silent by default, speaks only when it has a proposed
PR or an unresolvable blocker. Memory persists across events. Sprint
contracts between planner subagent and evaluator subagent. **You've
built the LangChain-style "ambient agent" in production shape.**

### 32. Research agent with continual learning (Cleric-style) — *2 weeks*
Harness (30) wired to a research domain (e.g. "on-call SRE for this
repo"). Every incident produces a new skill file. Weekly cron:
deduplicate, cluster, compress. Ship a dashboard that shows the skill
library growing. **You've built the continual-learning loop** from §9.

### 33. End-to-end training + harness + eval — *1–2 months* (final
capstone)
Stack the whole book:

1. Define a verifiable agentic task (e.g. "SQL agent against a fixed
   schema").
2. Build a mini benchmark for it (project 27). Pass ABC + RDI probes.
3. Post-train a small model (project 4 + 19) on trajectories from
   your harness.
4. Deploy the trained model inside your harness (project 30).
5. Run full evals (project 24) + adversarial probes (project 28).
6. Write it up. Open-source it. You have the best possible portfolio
   piece for an agents role.

---

## Recommended ordering

- **Weeks 1–3:** Tier 1 (5 weekend projects).
- **Weeks 4–10:** Tier 2 (one per week).
- **Weeks 11–16:** Tier 3 (pick 3 of 6).
- **Weeks 17–22:** Tier 5 evals (do all — most career-relevant).
- **Weeks 23–30:** Tier 4 RL (pick 1–2).
- **Weeks 31+:** Capstones. Project 33 is the goal.

Skip Tier 4 if you don't have GPU budget; everything else runs on a
laptop or a single small cloud GPU.

## Reference repos to diff against per tier

- **Tier 1:** `huggingface/trl`, `allenai/open-instruct`,
  `huggingface/open-r1`.
- **Tier 2:** `openclaw/pi-mono`, `anthropics/claude-agent-sdk-python`,
  `modelcontextprotocol/python-sdk`.
- **Tier 3:** `langchain-ai/deepagents`, `NousResearch/hermes-agent`,
  `microsoft/autogen`, `langchain-ai/open_deep_research`.
- **Tier 4:** `WooooDyy/AgentGym-RL`, `volcengine/verl`,
  `OpenRLHF/OpenRLHF`, `princeton-nlp/SWE-agent`.
- **Tier 5:** `UKGovernmentBEIS/inspect_ai`, `openai/evals`,
  `stanfordnlp/dspy`, `princeton-nlp/SWE-bench`.

## What "done" looks like

For each project, you pass if you can:

1. Explain every file you wrote without opening it.
2. Rebuild the core primitive from scratch in a new language in a day.
3. Diff your implementation against the reference repo and name 3
   design decisions that differ — and why.

If you can't do all three, you've read code, not built intuition.
Redo the project.
