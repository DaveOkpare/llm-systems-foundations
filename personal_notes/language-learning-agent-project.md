# Project — Post-training a Reasoning Agent to Learn New Languages at Inference Time

Companion to `post-training-and-agents.md` and `post-training-and-agent-projects.md`.
This is a single concrete research project fleshed out end-to-end.

**Thesis.** Post-train a small reasoning model so it translates between
English and a language it has *never seen* during training by actively
*reasoning with Agent Skills* — structured artifacts like bilingual
dictionaries, conjugation tables, grammar rules, and parallel example
sentences — rather than by storing the language in its weights.

This is a real, published research neighborhood; it's also a
near-perfect capstone for the study guide because it touches every
topic: Agent Skills · test-time reasoning · RLVR · continual learning ·
evals with verifiable rewards.

---

## 1. Research context — what already exists

### 1.1 The benchmark you should target: MTOB

**MTOB — Machine Translation from One Book** (Tanzer et al.,
arXiv:2309.16575, published 2024). Kalamang, a language with <200
speakers and essentially zero presence on the web. Resources provided:

- A ~570-page field-linguistics grammar book.
- A bilingual word list: **2,531 entries with POS tags**.
- A small parallel corpus: **~400 train sentences, 100 test
  sentences**, filtered to exclude anything quoted in the grammar book.

Baselines: ~44.7 chrF (Kalamang → English), ~45.8 chrF (English →
Kalamang). Human who learned from the same materials: 51.6 / 57.0.
Gemini 1.5 Pro with its long context window got close to human
performance on English → Kalamang.

- Paper: https://arxiv.org/abs/2309.16575
- Site: https://lukemelas.github.io/mtob/
- Code + data: https://github.com/lukemelas/mtob

### 1.2 The uncomfortable finding you must confront

**Aycock et al., "Can LLMs Really Learn to Translate a Low-Resource
Language from One Grammar Book?"** (arXiv:2409.19151). Key result:
*almost all improvement comes from the parallel examples, not the
grammar explanations.* Replicated on Manchu (ACL 2025): "high-quality
dictionaries and good parallel examples are very helpful, while
grammars hardly help."

- arXiv:2409.19151 (paper): https://arxiv.org/abs/2409.19151
- Manchu follow-up: https://aclanthology.org/2025.acl-long.429/

**This is the project's central gap.** Grammar rules don't help base
LLMs much because they don't explicitly *apply* rules — they do soft
pattern-matching on whatever's in context. A **reasoning** model that
decomposes the task ("identify the verb → look up root → find
conjugation → apply rule") and uses skills as **tools** might
actually benefit from grammar in ways the base LLMs don't. Whether it
does is the project's research question.

### 1.3 Adjacent work — reasoning-for-MT is already a subfield

- **MT-R1-Zero** — first R1-Zero-style RL-for-MT with no SFT cold-start.
  Rule-metric mixed reward. Strong on low-resource.
- **R1-T1** — six CoT templates mirroring human translator strategies,
  RL-trained; self-evolving reasoning trajectories.
- **"Unlocking Reasoning Capability on MT in LLMs"** (arXiv:2602.14763).
- **Test-Time Scaling of Reasoning Models for MT** (arXiv:2510.06471).
- **"Read it in Two Steps: Translating Extremely Low-Resource
  Languages with Code-Augmented Grammar Books"** (arXiv:2506.01796) —
  closest thing to this project; uses code to represent rules.
- **LLM-Assisted Rule-Based MT for Low/No-Resource** (USC ANRG).

### 1.4 Why you should build this anyway

Everything above is either (a) pure in-context long-window MT or
(b) reasoning-RL on MT *without* tool-using skills. Nobody has fully
wired up:

- A **reasoning** model,
- Post-trained via **RLVR**,
- That treats dictionary / conjugation / grammar / examples as
  **Agent Skills** (dynamically loaded, not stuffed in context),
- With a **verifiable reward** (chrF + back-translation) making the
  skill-use behavior learnable.

That's the project.

---

## 2. Primer — Anthropic Agent Skills (the primitive you'll use)

An Agent Skill is a folder containing a `SKILL.md` with YAML
frontmatter plus optional scripts, references, and data files. When
the agent judges a skill relevant, the harness loads it into context;
when it isn't, the skill stays on disk. Skills can be split into
referenced files when `SKILL.md` grows unwieldy. They work identically
across Claude API, Claude Code, and claude.ai.

Minimum `SKILL.md`:

```markdown
---
name: kalamang-dictionary
description: Bilingual Kalamang↔English word list with POS tags and morphological notes. Load when translating between Kalamang and English.
---

# Kalamang Dictionary

Use `scripts/lookup.py <word>` to look up a Kalamang or English term.
Use `scripts/lookup.py --pos VERB <word>` to filter by part of speech.

## Coverage
...

## Files
- `data/word_list.tsv` — full 2,531-entry dictionary
- `scripts/lookup.py` — grep-with-POS-filter helper
```

**Required reading before you start building:**

- Overview: https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview
- API guide: https://platform.claude.com/docs/en/build-with-claude/skills-guide
- Claude Code docs: https://code.claude.com/docs/en/skills
- Anthropic's own skills repo: https://github.com/anthropics/skills
- "Equipping agents for the real world with Agent Skills" (Anthropic
  engineering): https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills
- Complete Guide PDF:
  https://resources.anthropic.com/hubfs/The-Complete-Guide-to-Building-Skill-for-Claude.pdf
- Skilljar intro course: https://anthropic.skilljar.com/introduction-to-agent-skills

---

## 3. The four skills you'll design

One folder each. Treat the skill files as the *syllabus* the model
uses at inference time.

### 3.1 `kalamang-dictionary`

- `data/word_list.tsv` — the 2,531-entry MTOB word list.
- `scripts/lookup.py` — lookup by Kalamang or English, filter by POS.
- `SKILL.md` — invocation examples, note on ambiguous entries.

### 3.2 `kalamang-morphology`

- `data/verb_conjugations.json` — extracted from the grammar book:
  stem + tense/aspect/mood affixes.
- `data/noun_morphology.json` — plural, possessive, case endings.
- `scripts/conjugate.py <root> <features>` — deterministic rule
  application.
- `scripts/decompose.py <surface_form>` — best-effort morpheme split.
- `SKILL.md` — when to reach for each script.

### 3.3 `kalamang-grammar`

- `rules/` — one markdown file per grammar topic (word order,
  negation, TAM marking, etc.), each <1K tokens.
- `SKILL.md` — index with one-line descriptions; forces the agent to
  load only the rule it needs.

### 3.4 `kalamang-examples`

- `data/parallel.tsv` — the MTOB training split.
- `scripts/retrieve.py <query>` — embedding-based retrieval of the
  k most similar training sentences. **Critical** — Aycock §1.2
  shows parallel examples dominate the signal.
- `SKILL.md` — when to retrieve, how many to use.

The design discipline matches Claude Code's rule (§12.1a, Pattern 3):
*lean index in context, details loaded on demand, logs on disk.*

---

## 4. The agent loop — what the model actually does at inference time

```
Input: "Translate to Kalamang: The child eats fish."

Reasoning trace (what the model produces):

1. Decompose: subject=child, verb=eats, object=fish.
2. CALL kalamang-examples.retrieve("child eats fish")
   → 3 parallel examples with children-verb-food patterns.
3. CALL kalamang-dictionary.lookup("child") → mu (N).
4. CALL kalamang-dictionary.lookup("fish") → ien (N).
5. CALL kalamang-dictionary.lookup("eat") → an (V, root).
6. CALL kalamang-grammar.rules/word_order.md
   → SOV default, verb takes subject-agreement suffix.
7. CALL kalamang-morphology.conjugate("an", {tense: "present",
   subject: "3sg"}) → anan.
8. Compose: mu ien anan.
9. CALL kalamang-examples.retrieve("mu ien")
   → sanity-check against similar forms.
10. Output: "mu ien anan."
```

The *post-training target* is to make the model reliably produce
traces like this, then to reward correct outputs via RLVR.

---

## 5. Post-training recipe

Five stages. Tier numbering references §1–§5 of
`post-training-and-agent-projects.md`.

### Stage 1 — Synthesize SFT data (Tier-1 work, ~1 week)

Generate ~5k reasoning traces using a strong teacher (Claude Opus or
similar) prompted with the skills and a held-out subset of the
parallel training sentences as targets. Quality-filter by chrF > 40
against the gold target. This is your **reasoning-trace SFT corpus**.

This is rejection-sampling SFT (project 5 in the companion file).

### Stage 2 — Base SFT on reasoning traces (~3 days of training)

Start from a small open reasoning model (e.g. Qwen3-1.7B / 4B / 8B
depending on budget). Standard SFT on the synthesized traces:
cross-entropy with prompt-masking. Don't train on the target
translation itself — train on the *tool-calling reasoning* leading to
it.

After Stage 2 the model should call skills in the right order. It
will still translate poorly.

### Stage 3 — Design the verifiable reward

Hybrid reward, computed per completed rollout:

- **R_chrF** = chrF++ between the model's final translation and the
  held-out gold reference (when one exists).
- **R_back** = back-translation score. Translate output Kalamang →
  English with a frozen strong model (Claude/Gemini); measure chrF to
  the original English prompt. Gives a signal even for prompts
  without gold.
- **R_skill** = small bonus for well-formed skill calls (parses
  cleanly, cites real entries). Guards against the model
  hallucinating tool output.
- **R_format** = penalty if output isn't a valid final
  `<translation>…</translation>` block.

Total: `R = α·R_chrF + β·R_back + γ·R_skill + δ·R_format`.
Start with `(α, β, γ, δ) = (0.5, 0.3, 0.1, 0.1)`; tune.

This is the core RLVR design work (Tier-1 projects 3–4; Tier-5
projects 24, 27).

### Stage 4 — GRPO with tool-execution rollouts (1–2 weeks of training)

Use TRL's `GRPOTrainer` or verl. Rollout requires a live harness
that actually executes the skills — you can't score a trace without
actually running `lookup.py` etc. Use the minimal harness you built
in Tier-2 project 6, extended with skill loading from Tier-3 project 17
(Hermes-style skill-learning loop).

Group size G = 8–16. KL β = 0.04. Learning rate 5e-6. Watch:

- **Skill-call validity rate** (should climb early).
- **Mean reward per group** (should climb steadily).
- **Output-translation chrF on held-out** (the real metric).
- **KL-to-reference** (shouldn't explode).

### Stage 5 — Continual learning at test time (bonus, 1 week)

When the agent succeeds on a hard test sentence, append a compact
"lesson" to a `kalamang-learned/` skill directory — e.g., "verbs of
consumption take suffix `-an` in present 3sg." Retrieval step in
future rollouts surfaces these lessons. This is the Hermes closed
learning loop (Tier-3 project 17) applied to a linguistic domain.

---

## 6. Evaluation plan

### 6.1 Primary metric

**chrF++ on MTOB test set**, both directions. This is what the
benchmark reports; this is what you compare against.

Targets (budget-dependent):

- **Minimum viable result:** beat the Aycock et al. finding that
  "grammar hardly helps" — show your reasoning-agent *does* improve
  when grammar is provided, vs. dictionary+examples alone.
- **Stretch:** match or beat MTOB baselines with a model 10× smaller
  than Gemini 1.5 Pro.
- **Reach:** match human-from-materials performance (51.6 / 57.0).

### 6.2 Ablations (this is where the paper lives)

Strip one skill at a time and remeasure:

1. All four skills.
2. Minus grammar.
3. Minus conjugation.
4. Minus dictionary.
5. Minus examples. *Hardest baseline* — Aycock shows examples
   dominate.
6. Minus reasoning trace (same skills, no CoT).
7. Base model, skills not even offered.

If the gap between (1) and (5) or (1) and (6) is substantial, you
have a real result.

### 6.3 Eval discipline — don't fool yourself

Apply the §13 methodology:

- **Hamel error analysis** (project 24) on 50 failed translations.
  Build a failure taxonomy: missing vocabulary · wrong morphology ·
  wrong word order · bad skill retrieval · reasoning hallucination.
- **LLM-as-judge** for semantic adequacy (project 25): validate
  TPR/TNR on 100 hand-labeled examples before using in the loop.
- **ABC checklist** (project 27) — make sure task validity holds:
  a task is solvable iff the model can reason through the skills.
  Log every skill call so outcome validity is auditable.
- **Berkeley RDI adversarial probes** (project 28) — confirm your
  eval can't be gamed (e.g. by echoing retrieved examples verbatim).

---

## 7. Pick the language — Kalamang is the obvious target, but…

**Why Kalamang:** MTOB is well-curated, peer-reviewed, has human
baselines, and the materials are licensed for research. Almost zero
chance of pretraining contamination.

**Why consider alternatives:**

- Kalamang's resources were built *by linguists*. They're cleaner
  than you'll find for most low-resource languages. Your system
  might work on Kalamang and fail in the wild.
- The Manchu follow-up (aclanthology.org/2025.acl-long.429) offers a
  second test bed.
- You could build your own micro-benchmark using a constructed
  language (Toki Pona, Esperanto removed from training by filtering)
  — cheap but less credible.

**Recommendation:** target Kalamang/MTOB as primary; add Manchu as a
generalization check if time permits.

---

## 8. Staged plan (realistic, ~10 weeks part-time)

| Week | Milestone | Output |
|------|-----------|--------|
| 1 | Read the five core papers (§1) + Agent Skills docs (§2). | Notes doc + gap statement. |
| 2 | Build the four skill folders from MTOB raw files. | 4 × `SKILL.md` + scripts + data. |
| 3 | Minimal harness that executes skills (fork your Tier-2 project 6). | Runs one hand-written trace end-to-end. |
| 4 | Synthesize SFT dataset with a teacher model. | ~5k filtered reasoning traces. |
| 5 | SFT on a small reasoning model (Qwen3-1.7B or similar). | Checkpoint that calls skills in order. |
| 6 | Design and implement the hybrid reward. | Reward function with unit tests on known-good and known-bad traces. |
| 7–8 | GRPO training with tool-executing rollouts. | Improved checkpoint; training curves. |
| 9 | Evaluation on MTOB test set, ablations, error analysis. | Results table. |
| 10 | (Optional) Add test-time continual learning. Write up. | Short paper / blog post / repo. |

Compute estimate: one A100 (40GB) or H100 for 3–5 days over weeks
7–8 is enough for the 1.7B–4B size. You can prototype on an RTX 4090.

---

## 9. Expected result — honest baseline

You *might* fail to beat the Aycock result. That's still a paper:
"reasoning-with-skills doesn't help on Kalamang either, and here's
why." More likely, you get a moderate lift from skill use on hard
sentences (long, morphologically complex), a null result on easy
ones. Either outcome is publishable.

The real prize is the infrastructure. You'll walk out with:

- An Agent-Skills-native post-training pipeline.
- A verifier design that handles a noisy, partially-labeled target.
- A reasoning-agent harness that actually *executes* tools during RL
  rollouts.
- A portfolio project that maps 1:1 onto Anthropic's Agentic Systems
  JD (§12): "novel harness design," "custom evals," "large-scale RL on
  language models," "coding agents, research automation," "memory/
  context engineering."

## 10. Reading list — curated for this project

**Must read before starting:**

1. MTOB paper — arXiv:2309.16575
2. "Can LLMs Really Learn to Translate…" — arXiv:2409.19151
3. Anthropic Agent Skills overview + engineering post.
4. MT-R1-Zero and R1-T1 papers (reasoning-for-MT RL precedents).
5. A-Mem paper (§9 of study guide) for the continual-learning piece.
6. §13 of the study guide (eval methodology).

**Worth a skim:**

- "Read it in Two Steps: Translating Extremely Low-Resource Languages
  with Code-Augmented Grammar Books" (arXiv:2506.01796).
- Manchu in-context MT (aclanthology.org/2025.acl-long.429).
- "Unlocking Reasoning Capability on MT" (arXiv:2602.14763).
- Test-Time Scaling of Reasoning Models for MT (arXiv:2510.06471).
- David Haberlah's multi-agent translation post (Medium) — good on
  the Translator/Evaluator/Reviewer pattern you'll mirror.

**Reference repos to diff against:**

- `lukemelas/mtob` — benchmark code.
- `huggingface/trl` — `GRPOTrainer`.
- `anthropics/skills` — reference skill layout.
- `langchain-ai/deepagents` — harness middleware as a model for your
  skill-loading layer.
- `WooooDyy/AgentGym-RL` — multi-turn-tool RL reference.
