# Nesso2-0.4B-Agentic

**A ~0.4B bilingual (Italian/English) small language model built for function calling and agentic execution — and, to our knowledge, the strongest open model for Italian agentic tool use in the sub-billion-parameter class.**

- 🤗 Model: [`mii-llm/nesso2-0.4B-agentic`](https://huggingface.co/mii-llm/nesso2-0.4B-agentic) *(release name)*
- ✍️ Blog post — the experiments, hypotheses & conclusions: [`BLOG.md`](BLOG.md)
- 📄 Full visual report: [`report.html`](report.html)
- 🧬 Base: [`mii-llm/zagreus-0.4B-ita`](https://huggingface.co/mii-llm/zagreus-0.4B-ita) → knowledge-CPT → 32k context → agentic SFT

---

## Headline results

Measured across **six models** on three independent evaluation families.

| | Nesso2 (v8) | Qwen3-0.6B |
|---|---|---|
| **Agentic — total** (100 cases) | **68 / 100** 🥇 | 67 / 100 |
| **Agentic — Italian** (x/50) | **35** | 29 |
| **Agentic — English** (x/50) | 33 | **38** |
| **Italian academic avg** (acc) | 0.334 | **0.352** |
| **Italian conversation** (LLM-judge, /10) | **4.40** | 2.80 |

**Takeaways.** Nesso2 is the **best agentic model overall** and leads Italian tool use by **+6**, while **trailing Qwen3-0.6B by ~2 points on the Italian academic average** (0.334 vs 0.352: ahead on Italian HellaSwag/ARC, behind on MMLU and IFEval) and posting the **best Italian conversation score of its lineage**. The trade is explicit: Qwen keeps English tool use, raw MMLU and IFEval.

---

## The road to Nesso2 — the phases and their experiments

Nesso2 is not a single fine-tune. It is the end of a chain of phases, each with its own hypothesis, experiments, and evaluation. The pivotal work happened *before* the agentic tuning — in continued pre-training — and this is the part usually left out. Here it is in full.

### Phase 0 — The base, and the wall

The foundation is [Zagreus-0.4B-ita](https://huggingface.co/mii-llm/zagreus-0.4B-ita): ~350M parameters, pre-trained from scratch on ~1T tokens (≈400B English / 400B Italian / 200B code from FineWeb, FineWeb-2, FinePDFs, StarCoder) on 64× A100 with Nanotron.

One fact organized everything that followed: **across all ~775k pre-training steps, MMLU stayed flat at chance (~0.25).** A small model trained on web text simply does not acquire enough world knowledge to answer MMLU. That is the wall.

### Phase 1 — Proving the wall is a *pretraining* wall (the OPD experiments)

**Hypothesis.** Maybe post-training can supply the missing knowledge — supervised fine-tuning, or on-policy distillation (OPD) from a larger teacher.

**Experiment.** On A100s (via our [palingenesis](https://github.com/mii-llm/palingenesis) framework) we ran SFT followed by **Mixed On-Policy Distillation** with a nesso-3B teacher, then a second, *focused* MMLU-only OPD run.

**Result.** Mixed OPD was a **wash** on every academic benchmark (Italian MMLU 0.258 → 0.264). Even the focused MMLU-only run moved it just **+2 points** (0.283 → 0.302). But the same runs revealed something crucial: on Italian HellaSwag and ARC we already *beat* Qwen3-0.6B — **the entire gap to Qwen was MMLU alone.**

**Conclusion.** MMLU is a **pretraining-knowledge wall**. Post-training reweights what the model already knows; it cannot inject facts that were never learned. **Continued pre-training (CPT) is the only lever.** This negative result set the whole strategy.

### Phase 2 — Breaking the wall (the CPT program)

**Hypothesis.** CPT on knowledge-dense, *extractable* data can inject the facts MMLU needs — if we prevent catastrophic forgetting with replay.

**The knowledge corpus** (built in two tiers, on LUMI):
- **Tier 1 (~52M tokens)** — ready-made QA turned into text: MMLU auxiliary-train, SciQ, OpenBookQA, ARC, and the Italian *pinocchio* set.
- **Tier 2 (~2.0B tokens)** — Italian + English **Wikipedia (2.47M passages), QA-augmented by Qwen3.6-35B**: for each passage, 3–5 grounded question–answer pairs, one multiple-choice item, and a summary, all self-contained and in-language. This is the *extractability* trick — knowledge the model can actually retrieve, not just tokens it has seen. (Augmentation ran at ~97.5% yield; a strict grounding filter kept answers inside the source passage.)

*Example — how one passage becomes retrievable knowledge (the augmentation format):*

> **Source passage (Wikipedia IT):** «Alessandro Volta … inventò la pila elettrica nel 1800 …»
> **→ grounded QA:** *D: Chi inventò la pila elettrica? R: Alessandro Volta.* · *D: In che anno? R: Nel 1800.*
> **→ multiple choice:** *Volta inventò… (a) il telefono (b) la pila elettrica (c) la radio* → **(b)**
> **→ summary:** *Alessandro Volta inventò la pila elettrica nel 1800.*

The same fact is presented as free text, as a question, and as a choice — so the model learns to *recall* it, not just recognize it.

**Experiment.** Resume the base checkpoint, re-warm the learning rate, and train a **50/50 blend of knowledge and replay** (old pretraining data) so reasoning isn't forgotten. We validated with a 1.5B-token probe, then committed to a definitive 4.46B-token run.

**Result** (lm-eval; MMLU 5-shot `acc`, HellaSwag/ARC 0-shot `acc_norm`):

| task | base | probe (1.5B, 3e-4) | **CPT-full (4.46B, 5e-4)** | Qwen3-0.6B |
|---|---|---|---|---|
| MMLU-it | 0.253 | 0.340 | **0.372** | 0.404 |
| MMLU-en | 0.246 | 0.366 | **0.394** | 0.474 |
| HellaSwag-it | — | — | **0.393** | 0.362 |
| ARC-it | — | — | **0.287** | 0.273 |

![Continued pre-training breaks the MMLU wall](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/nesso2/images/cpt_mmlu.png?raw=true)

**Conclusion.** CPT broke the wall — **+12 points of Italian MMLU over the base**, from chance to genuinely-above-chance. And as a *base model, before any SFT*, the CPT checkpoint already **edges Qwen3-0.6B on the Italian average** (0.351 vs 0.346), beating it on Italian commonsense (HellaSwag) and reasoning (ARC). The 50/50 replay worked: reasoning was not sacrificed for knowledge.

*(Methodological note: following the project's convention we did not decontaminate the corpus — Qwen's own training data is contaminated too, so decontaminating only ours would handicap the comparison. The Italian MMLU/ARC figures therefore include some memorization on both sides.)*

### Phase 3 — 32k context, without losing the knowledge

**Hypothesis.** An agent must hold tool schemas, observations, and multi-step trajectories in context — so extend to 32k tokens, but without eroding the hard-won knowledge.

**Experiment.** Attention-scaling (ABF): raise RoPE θ from 10,000 to 1e6 and `max_position_embeddings` to 32,768, then adapt on 1.44B tokens of long documents, keeping 15% knowledge replay as a guard.

**Result.** Retention held: MMLU dropped only **0.9–1.6 points**, while HellaSwag/ARC were flat-to-up. **32k context came essentially for free.**

### Phase 4 — Agentic SFT (v3 → v8 = Nesso2)

Supervised fine-tuning on the 32k knowledge base, with **TRL** (`SFTTrainer`) + **FSDP**, Llama-3 template, 3 epochs, LR 1e-3 cosine. The mixture is bilingual instruction data plus a synthetic function-calling corpus with randomized tool schemas. This is the phase that took five iterations to get right — the full story is below, in **[The iteration — from 61 to 68](#the-iteration--from-61-to-68)**.

> Every stage ran on the **Seeweb** HPC infrastructure. The knowledge CPT (Phase 2) is what separates Nesso2 from a plain SFT on the same base — it is why Italian MMLU/ARC survive the agentic specialization, and why Nesso2 beats its no-CPT sibling `nesso-0.4B-agentic` on MMLU in both languages (Italian 0.326 vs 0.282).

---

## Evaluation — three families

We read three families **together**, because at 0.4B they disagree and the disagreement is the signal.

- **Family A — Academic** (`eval/bench_academic.sh`): MMLU (5-shot acc), HellaSwag/ARC (0-shot acc_norm), IFEval (generative), Italian + English, via the [mii-llm lm-evaluation-harness fork](https://github.com/mii-llm/lm-evaluation-harness/).
- **Family B1 — Agentic function calling** (`eval/agentic_eval_100.py`): a frozen bilingual **100-case** suite, 10 categories, 50 IT / 50 EN, Hermes `<tool_call>` format, **pure greedy** decoding, automatic per-category grader. Cases in `eval/agentic_eval_cases_100.json`.
- **Family B2 — Conversation** (`eval/conv_gen.py` → `eval/judge_conversations.py`): 20 multi-turn tasks per language, graded 1–10 on correctness / language-fidelity / helpfulness by **Qwen3.6-35B-A3B**. Prompts in `eval/conv_prompts.json`.

### Family B1 — agentic function calling

![Agentic benchmark — 100 cases](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/nesso2/images/agentic_total.png?raw=true)

![Italian vs English tool use](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/nesso2/images/agentic_bylang.png?raw=true)

*Example — a case Nesso2 handles well (parallel same-tool):* given a `get_weather` tool and *"Che tempo fa a Roma e a Torino?"*, it emits **two** calls — `get_weather(Roma)` and `get_weather(Torino)`.

#### Per-category capability (x/10)

![Per-category capability: Nesso2 vs Qwen](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/nesso2/images/agentic_categories.png?raw=true)

| Category | v3 | v6 | v6.1 | v7 | **Nesso2** | Qwen |
|---|---|---|---|---|---|---|
| single call | 9 | 8 | 8 | 9 | **9** | 7 |
| parallel (same tool) | 10 | 10 | 10 | 9 | **10** | 5 |
| parallel (diff tools) | 7 | 4 | 4 | 2 | **4** | 4 |
| multi-argument | 10 | 6 | 6 | 6 | **6** | 10 |
| disambiguation | 10 | 9 | 9 | 9 | **8** | 8 |
| missing argument | 4 | 10 | 8 | 5 | **6** | 3 |
| unavailable tool | 1 | 10 | 9 | 9 | **9** | 10 |
| no-tool discrimination ▲ | 1 | 2 | 2 | 1 | **7** | 8 |
| observation grounding ▼ | 7 | 8 | 8 | 6 | **3** | 10 |
| multi-step ▲ | 2 | 0 | 1 | 5 | **6** | 2 |
| **Total** | 61 | 67 | 65 | 61 | **68** | 67 |

▲ deliberate gains in the final run · ▼ the one accepted regression (see caveats).

### Family B2 — conversation quality

**Why an LLM judge.** The academic suite (Family A) scores knowledge and format; it says nothing about whether the model is a *good conversationalist* — coherent, factually correct, in the right language, actually useful. Those qualities need a judgement call, so we make one explicitly: `Qwen3.6-35B-A3B` grades every answer 1–10 on three axes — **correctness** (rewards facts/arithmetic, penalizes hallucination), **language fidelity** (penalizes answering in the wrong language), and **helpfulness** — across 20 multi-turn tasks per language, on greedy generations.

**Results** (mean score, out of 10):

| model | Italian | English | Both | correctness | helpfulness |
|---|---|---|---|---|---|
| nesso-0.4B-agentic *(reference)* | 4.40 | **6.40** | **5.40** | 4.78 | 5.38 |
| v3 | 4.30 | 4.60 | 4.45 | 4.25 | 4.67 |
| Qwen3-0.6B | 2.80 | 5.80 | 4.30 | 4.15 | 4.22 |
| **Nesso2 (v8)** | **4.40** | 3.80 | 4.10 | 3.92 | 4.40 |
| v6.1 | 4.15 | 3.85 | 4.00 | 3.62 | 4.05 |

![Italian conversation quality — 35B judge](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/nesso2/images/conversation_it.png?raw=true)

**The Italian result.** Nesso2 scores **4.40 on Italian conversation — the highest of the entire Zagreus line, tied with the reference `nesso-0.4B-agentic`** — while Qwen3-0.6B manages only **2.80** (it frequently answers Italian prompts in English, or in weaker Italian). That 1.6-point margin is the largest and most consistent lead we hold over Qwen anywhere in this report, and it comes from the same source as the knowledge win: the CPT stage plus Italian-first data.

**The finding that validated the v8 bet.** The risk going into v8 was that adding a large amount of no-tool "answer directly" data would erode conversation. The opposite happened: from v6.1 to v8, **correctness rose 3.62 → 3.92 and helpfulness 4.05 → 4.40**. Because the no-tool data is *natural, complete* answers (capitals, currencies, definitions, general facts), it actually *taught the model to answer factual questions better*. This is the conversational proof that no-tool discrimination is **chat-safe** — unlike the terse abstention data of earlier rounds, which had hurt chat. The v8 bet paid off on both the agentic axis *and* the conversational one.

**The honest caveat — English chat.** English is Nesso2's clear soft spot: **3.80**, below Qwen (5.80) and the reference (6.40). On the English-weighted "Both" average it therefore trails both `v3` (4.45, carried by its stronger English) and the reference (5.40). The Italian-first mixture and the agentic specialization cost English register — a deliberate trade, but a real one. For open-ended *English* conversation, Nesso2 is not the model to reach for.

**Cross-check — the failed DPO runs.** The same judge independently caught our unsuccessful preference-tuning experiments: `v3-dpo` scored **3.42** and the earlier `llama3-dpo` **3.70**, both *below* their SFT baselines. That is separate confirmation that DPO toward "more detailed" answers backfires at 0.4B — the negative result recorded under *What didn't work*.

---

## The iteration — from 61 to 68

**Core lesson: at 0.4B the training-data budget is zero-sum.** Some skills are *data-responsive* (more examples move them); others are *capacity-bound* (they don't). Progress came from spending budget only on the former.

| Run | Change | Agentic |
|---|---|---|
| v3 | SFT on filtered IT + agentic; strong calls, over-fires | 61 |
| v5 | ChatML + embedding resize — **lost to Llama-3, reverted** | — |
| v6 | Rebalanced negatives → fixed missing-arg / unavailable | 67 |
| v6.1 | Per-language quality floor (all IT, EN ≥ 0.6) | 65 |
| v7 | "Best of both" — spread budget too thin, **failed** | 61 |
| **v8 = Nesso2** | **Focused no-tool fix (4.5× data), budget reclaimed from capacity-bound categories, multi-step kept** | **68** |

![The iteration: agentic total v3 → v8](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/nesso2/images/iteration.png?raw=true)

> The last mile wasn't more data — it was **one data-responsive skill, trained richly and naturally, with budget stolen from skills that don't respond to data.**

### What didn't work (kept for the record)

- **Spreading the budget (v7)** — improving everything at once lowered the total to 61.
- **DPO on v3** — cut conversation 4.25 → 3.48 for +0.008 IFEval; "more detailed" becomes hallucination at 0.4B.
- **ChatML + resize (v5)** — lost to the plain Llama-3 template across the board.
- **Repetition penalty on tool calls** — the chat-tuned `rep_penalty 1.15` corrupts tool JSON (the template echoes `name`/`arguments`, so those tokens get suppressed → `"Name"` / dropped args). **Use pure greedy for structured output; the penalty is only for prose.**

### Honest caveats

- **English tool use:** Qwen3-0.6B leads (38 vs 33 / 50).
- **Observation grounding:** v8's weakest category (obs 3/10) — after a tool returns, the model sometimes answers from priors or re-calls instead of grounding on the result. Confirmed as a *real* weakness on an independent benchmark (not an eval artifact). Mitigable at the app layer (feed observations back explicitly).
- **Abstention on tempting cases:** reliable on many phrasings, but can fire a tool / fill a default value on borderline, tool-tempting prompts. Phrasing-sensitive.
- **Raw knowledge:** Qwen stays ahead on English MMLU.

---

## After v8 — grounding, merging, and cross-validation

Once v8 was the candidate, we tried to close its two soft spots (observation grounding, abstention) and stress-tested the result on a second, independently-authored benchmark. Both efforts confirmed v8 as the right release.

**A grounding-heavy variant (v9)** put ~63% of the agentic mix into observation traces, schema-fidelity, and missing-argument clarification. It *worked* on its target — **observation grounding recovered 3 → 9** (independently reproduced on the second benchmark: 53.5 → 76.8). But at that dose the terse, structured data **collapsed conversation** — Italian chat fell **4.50 → 3.50**, with correctness and helpfulness down too — and it dipped parallel-calling. **Lesson: grounding data genuinely fixes observation handling, but must be a *minority* of the mix (~15–20%), not 63%.** Not shipped.

**A weight-merge (0.65·v8 + 0.35·v9)** was the classic "combine specialists" move — same base, so ideal for merging. On *our* 100-case suite it topped everything (**69**, best Italian 37). But on the **independent partial-credit benchmark it dropped to worst of the three (57.8%)** — the parallel-calling and formatting gains that a binary grader accepted did not survive strict grading. **The merge's win didn't generalize; it was fragile to grading philosophy.** Not shipped.

**Cross-benchmark validation.** Across two independently-authored function-calling benchmarks, **v8 wins Italian on both** at equal (fast) latency (my suite: IT 35/50 vs Qwen 29; the second: FC-it **63.1%** vs Qwen3-0.6B *non-thinking* 51.3%), and v8 / v9 score *consistently* (unlike the merge). The claim is robust, not benchmark-specific.

**Speed vs. reasoning — the comparison that scopes the claim.** Agentic tool-calling is **latency-sensitive** (a real agent makes several calls per turn), so the fair, production-realistic comparison is the **fast, single-forward-pass regime**. There, Nesso2 leads Italian FC over Qwen3-0.6B non-thinking (63.1 vs 51.3) at the same speed — **~0.83 s / ~40 output tokens per call**. Qwen's *thinking* mode is stronger on raw accuracy (Italian FC 73.1, and it wins most categories) — but at **~6× the latency (4.71 s vs 0.83 s) and ~5× the tokens (208 vs 40)**, a different latency class unsuitable for real-time loops. And even against thinking-Qwen, Nesso2 still wins **multi-step (67–70 vs 14 — thinking derails Qwen there)** and **parallel-same-tool (91 vs 76)**. So the precise claim: **the best Italian tool-caller at low latency / without test-time reasoning — and ~6× faster than the reasoning alternative.**

**Production guidance.** For real-time agents, **speed is a first-class requirement** — so ship **v8** as a single model for Italian agentic *and* conversation, wrapped with two app-layer guards: (1) validate required arguments are grounded in the user's turn before executing, (2) feed tool observations back explicitly — which neutralize its two soft spots without v9's generation cost. Only split into two models (v9 for a heavily observation-loop-driven agent, v8 for chat) if you have measured obs as the bottleneck. See the [model card](MODEL_CARD.md#production-notes--recommendations) for details.

---

## Reasoning and grounding — mapping the 0.4B wall (negative results)

After v8 shipped, we ran a deeper program to fix its one real weakness — **observation grounding** (obs 3/10) — and to add a **reasoning mode**. Four experiments, all negative at 0.4B. Together they map a hard *capacity* wall, and we keep them for the record.

**1. v8.1 — obs via a minority data dose.** We swapped v8's weak observation generator for the rich one at a careful **12%** (learning v9's "63% was too much" lesson) and re-ran the *full* battery. On the 100-case suite it worked cleanly: **obs 3 → 8, total 68 → 72, Italian FC a new high (38/50)**. But the rest of the battery caught the cost — the same obs data **terse-ified conversation** (Italian chat 4.55 → 3.60) and **dropped pure function-calling** on the independent benchmark (63.1 → 57.0), while academics stayed flat. *The obs gain showed up only on the benchmark that tests obs; it cost the things the product actually runs on.* A second confirmation (after v9) that a **monolithic 0.4B can't add obs without hurting chat**. Not shipped.

**2. A reasoning mode + RLVR.** We built a full **GRPO / RLVR loop** (grader reward: correct call / correct abstention + grounding + a length penalty) and a `/think`·`/no_think` hybrid seeded by *rationalization distillation* (a 35B teacher writes the reasoning that justifies each gold action). Two findings: SFT alone makes reasoning **post-hoc, not causal** — `/think` was a wash-to-negative, helping abstention but hurting execution. RLVR nudged it the right way but **never past v8**: the decision data was near-ceiling, so the policy barely moved (KL ≈ 0.005). *The RLVR loop is sound and reusable — it just can't manufacture headroom that isn't there at 0.4B.*

**3. The hybrid — obs in a fast mode, chat protected by self-distillation.** The idea that should have worked: pin `/no_think` to v8's **own** outputs (self-distilled, so chat can't drift) and route obs capability only through the `/think` marker. It *half*-worked — **obs reached 9–10 in the fast mode** (v8's biggest hole, fixed) and English chat even improved. But **Italian chat still regressed** (v8 4.50 → 3.40): self-distillation captures v8's *average* Italian across varied prompts, not its *peak*, so training pulls Italian toward the mean. A v2 that rebalanced to 62% Italian and filtered degenerate targets recovered some (3.15 → 3.40) but not all. Not shipped.

**The wall, four ways.** Every method that adds observation-handling regresses **Italian chat** into the same band, while v8 — which never trained on obs — holds it at 4.50:

| method | obs added? | Italian chat (35B judge) |
|---|---|---|
| **v8 (shipped)** | no | **4.50** |
| v9 (30% obs) | yes | ~3.50 |
| v8.1 (12% obs) | yes | 3.60 |
| hybrid-v1 (obs in `/think`) | yes | 3.15 |
| hybrid-v2 (IT-rebalanced) | yes | 3.40 |

**Conclusion.** Observation-grounding and *peak* Italian chat **compete for the same 0.4B capacity, and obs training always wins at Italian's expense.** This is why v8 ships as-is — its obs hole handled by the two app-layer guards — and why observation-grounding and reasoning are a **3B target**, where there is room for both. The RLVR loop, reward, and hybrid/self-distill scripts port directly.

**The card that broke the wall — chat-anchored RL.** Every attempt above was *imitation* (SFT or self-distillation), which has **no signal to protect chat** — it fits the obs targets and lets conversation drift. Reinforcement learning is structurally different: its reward can *explicitly* include chat preservation. So we built it (see the next section), and **it is the one method that improved the 0.4B without the chat cost** — the wall was a property of *imitation*, not of the parameters.

---

## nesso2-0.4B-instruct — the chat-anchored RL that worked

We ran GRPO from v8 with a **two-part reward in the same batch**: on tool inputs, reward *observation-grounding* (0 if the model re-calls instead of using the result, 1 if it grounds on the observation); on chat inputs, reward *staying v8* (token-F1 to v8's own response). A **low KL coefficient** lets the policy actually move, while the **chat reward** — not the KL — is what holds conversation. That is the needle SFT cannot thread: *"get better at obs **while remaining v8 on chat**,"* expressed directly in the objective.

*(Config lesson that unlocked it: a large KL-to-reference anchor caps all movement — KL plateaus at ~0.005 regardless of learning rate. Drop beta low and let the **reward** protect chat.)*

The result is the first **strictly-or-better-than-v8** model of the whole program, on every eval family at once:

| family | v8 | **instruct (chat-anchor)** | Δ |
|---|---|---|---|
| Observation grounding (100-case) | 3 / 10 | **5 / 10** | **+2** |
| Agentic total (100-case) | 68 | **70** | **+2** |
| Conversation — Italian (judge /10) | 4.55 | 4.30 | −0.25 (held) |
| Conversation — English (judge /10) | 3.70 | **4.30** | **+0.60** |
| Conversation — overall (judge /10) | 4.12 | **4.30** | **+0.18** |
| Function-calling (independent bench, exact) | 33 / 100 · it 18 | 31 / 100 · **it 17** | tied (±1-2) |
| Academic (MMLU/HS/ARC/IFEval, it+en) | baseline | ≈ v8 | flat |
| Latency | 0.81 s | 0.78 s | same, single-pass |

**Nothing regressed.** obs went up, *overall* chat went up (correctness, helpfulness, and fluency all improved), English conversation closed most of its gap, function-calling held (Italian exactly), academics were flat — at the same no-think speed. It is a **single native mode** (no `/think`·`/no_think` — the two-mode lineage above was the negative result); it simply responds, fast.

**What ships, and why "instruct":** this model is released as **`nesso2-0.4B-instruct`** — the *best-conversational, fast* member of the family, alongside `Nesso2-0.4B-Agentic` (v8), the execution-first tool-caller. On **Italian conversation it is the strongest small model we know of** (judge 4.30 and robot-domain 87.5 vs Qwen3-0.6B's 2.70 / 60.1), and it now **beats Qwen on English conversation too** (robot-domain 77.5 vs 75.0).

**Honest limits (the target for the next iteration).** The obs gain is *real but small* (+2) and came from a single 800-step run on mostly-easy grounding cases; on a harder production-robot suite, observation-grounding and abstention did **not** generalize (our models still fire a tool when none applies). Those production decision skills — abstention, observation-grounding, multi-step — are the remaining gap for both "best Italian agentic" and English function-calling. The encouraging part is that we now have the **method** to attack them: the same chat-anchored RL, pointed at harder, domain-varied decision data, run longer. That is `instruct`'s planned v2.

---

## Repository layout

```
nesso2/
├── README.md                        this file
├── MODEL_CARD.md                    Nesso2-0.4B-Agentic (v8) model card
├── MODEL_CARD_INSTRUCT.md           Nesso2-0.4B-Instruct model card (chat-anchored RL)
├── report.html                      full visual technical report
├── eval/
│   ├── agentic_eval_100.py          Family B1 — 100-case function-calling suite (6 models)
│   ├── agentic_eval_cases_100.json  the frozen bilingual eval cases + tools
│   ├── conv_gen.py                  Family B2 — generate conversation answers
│   ├── judge_conversations.py       Family B2 — LLM-as-judge scorer (35B via vLLM)
│   ├── conv_prompts.json            the conversation eval tasks
│   └── bench_academic.sh            Family A — lm-eval-harness runner (it+en)
└── training/
    ├── sbatch-sft-agentic-v8.sh     agentic SFT (TRL + FSDP, Slurm) — produces v8
    ├── build_perlang.py             per-language quality-floor data prep (IT-all + EN ≥ 0.6)
    ├── push_to_hf.py                convert + push checkpoint (swaps in the tool template)
    ├── reward_chatanchor.py         chat-anchored RLVR reward (obs grounding + F1-to-v8 chat anchor)
    ├── grpo_train_ca.py             GRPO trainer, v8 → Instruct (branched reward, KL-to-v8)
    └── sbatch-grpo-ca.sh            chat-anchored RLVR launcher (Slurm)
```

The agentic instruction corpus and the RLVR data builder remain private (family policy); `reward_chatanchor.py` + `grpo_train_ca.py` reproduce the **method**, and `grpo_train_ca.py`'s docstring documents the exact input schema so the recipe is runnable on your own data.

> The synthetic **data-generation** scripts are intentionally **not** included: consistent with the family's policy, the agentic instruction corpus is a curated research asset and is not released as open source. The scripts here reproduce the **training recipe** and the **evaluation**, not the private data.

Paths inside the scripts (`/scratch/...`, `~/ai/...`, `giux78/...`) are environment-specific — adjust to your setup. `push_to_hf.py` reads the HF token from the environment / `huggingface-cli login`; no credentials are embedded.

---

## Quick start (inference)

```python
import re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "mii-llm/nesso2-0.4B-agentic"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto").eval()

tools = [{"type":"function","function":{"name":"get_weather",
    "description":"Ritorna il meteo per una città",
    "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]
messages = [{"role":"user","content":"Che tempo fa a Milano?"}]

prompt = tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
inputs = tok(prompt, return_tensors="pt", add_special_tokens=True).to(model.device)
out = model.generate(**inputs, do_sample=False, max_new_tokens=256,   # pure greedy — no rep penalty for tool calls
                     eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
print(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False))
# -> <tool_call>{"name": "get_weather", "arguments": {"city": "Milano"}}</tool_call>
```

See [`MODEL_CARD.md`](MODEL_CARD.md) for the full usage (function calling + plain conversation) and per-task evaluation tables.

---

## Citation

```bibtex
@misc{zagreus2025,
  title        = {The Joy and Pain of Training an LLM from Scratch:
                  A Technical Report on the Zagreus and Nesso Model Families},
  author       = {mii-llm community},
  year         = {2025},
  howpublished = {\url{https://github.com/mii-llm/zagreus-nesso-slm}},
}
```

> Made with ❤️ in Italy by [mii-llm](https://mii-llm.ai) · built on [Seeweb](https://www.seeweb.it) HPC · Apache-2.0
