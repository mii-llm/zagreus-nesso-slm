# Knowing When *Not* to Call a Tool

## How a 0.4B model came to lead Italian agentic tool use — told as a chain of hypotheses, experiments, and conclusions

Nesso-0.4B-agentic was already the strongest Italian model in our practical benchmark. That is a comfortable place to stop. We didn't.

The reason is that a good agentic model is not the one that calls the most tools. It is the one that calls the *right* tool, fills the *right* arguments, refuses when a tool is missing, and — the part everyone underestimates — answers directly when no tool is needed at all. A model that reaches for a function every time a function is offered is not an assistant; it is a reflex.

This post is the record of turning that reflex into judgement. It is written as a chain of hypotheses, experiments, and conclusions, including the experiments that failed, because at this size the failures are where the real constraints announce themselves. The result is **Nesso2-0.4B-agentic**: the best agentic score of six models we tested, a decisive lead in Italian, and — unusually for a specialized model — no loss of general knowledge or conversational quality.

Everything below is measured across the same six models: our agentic line (`v3`, `v6`, `v6.1`, `v7`, and the released `v8` = Nesso2), plus `Qwen3-0.6B` as the external rival.

---

## The governing idea: at 0.4B, the data budget is zero-sum

Before the experiments, the frame that organizes them.

A 0.4B model cannot be good at everything; capacity is the binding constraint. When we studied the agentic skills one by one, they fell into two kinds. Some are **data-responsive**: add more examples and the score moves. Others are **capacity-bound**: the model has hit its ceiling, and more data changes nothing. Nearly every mistake we made came from spending training budget on the second kind. Nearly every win came from spending it on the first.

Hold that distinction. It explains all six models.

---

## Experiment 1 — Put knowledge in pretraining, not in the fine-tune

**Hypothesis.** A small model cannot absorb facts from a few thousand supervised examples. If we want it to retain knowledge *and* specialize for tools, knowledge has to come from a dedicated continued-pretraining (CPT) stage, before any agentic data touches it.

**Experiment.** On top of the Zagreus-0.4B-ita base we ran a knowledge CPT on curated Italian/English Wikipedia and model-augmented QA, then extended context to 32k, and only then applied the agentic SFT. We compared the result against a sibling that skipped the CPT — the original Nesso-agentic — on the same academic suite.

**Conclusion.** It worked. Nesso2 reaches an Italian academic average of **0.334**, statistically level with Qwen3-0.6B's **0.336**, and it beats its no-CPT sibling on knowledge outright — Italian MMLU **32.6 vs 28.2**, English MMLU **27.0 vs 24.0**. Knowledge is a pretraining job. Once it lives in the weights, the fine-tune can specialize hard without eroding it. This single decision is why, later, heavy agentic tuning cost us no measurable knowledge.

---

## Experiment 2 — Measure with three families, not one

**Hypothesis.** At 0.4B a single benchmark actively misleads, because the skills trade off against each other and a single average hides the shape.

**Experiment.** We evaluated every candidate on three independent families: **academic** (MMLU, HellaSwag, ARC, IFEval, in Italian and English, via our lm-eval-harness fork); **agentic function calling** (a frozen bilingual 100-case suite, ten categories, fifty Italian and fifty English, scored by an automatic per-category grader); and **conversation** (twenty multi-turn tasks per language, graded 1–10 by a 35B judge model).

**Conclusion.** The families disagree, and the disagreement is the signal. Every one of our models beats Qwen in Italian tool use and loses to it in English. A model can win the agentic suite while sitting mid-table on chat. Read alone, any one family would have sent us in the wrong direction; read together, they told us exactly which lever to pull.

---

## Experiment 3 — Teach refusal (v6)

**Hypothesis.** Our v3 baseline was strong at emitting calls but trigger-happy: it invented arguments it hadn't been given and called tools that weren't offered. Refusal — asking for a missing argument, declining an unavailable tool — is a skill you can teach with data.

**Experiment.** We rebalanced the agentic mixture to include far more of these negative cases and retrained.

**Conclusion.** Refusal is strongly data-responsive. Missing-argument handling went from 4/10 to 10/10, unavailable-tool from 1/10 to 10/10, and the total climbed from **61 to 67**. The first confirmation that the responsive/bound distinction was real and actionable.

---

## Experiment 4 — Filter English, keep all Italian (v6.1)

**Hypothesis.** Our broad instruction corpus is noisier in English than in Italian. A single global quality threshold would throw away good Italian data to clean up bad English. A *per-language* floor would not.

**Experiment.** We kept every Italian row and applied a quality floor (score ≥ 0.6) only to English.

**Conclusion.** The per-language floor produced the best Italian conversation of that round and a solid, well-behaved model at **65**. Quality thresholds should be set per language, not globally — the weak language is the one to filter.

---

## Experiment 5 — Improve everything at once (v7) — the instructive failure

**Hypothesis.** If v6 fixed refusal and v6.1 fixed Italian register, a run that combined every improvement and also pushed the harder categories — different-tool parallel calls, multi-argument calls — should be the best of all.

**Experiment.** We built exactly that mixture, spreading the budget across many categories simultaneously.

**Conclusion.** It regressed to **61**, tied for last. This is the most important negative result in the project. The categories we spent new budget on — different-tool parallel and multi-argument — are *capacity-bound*: they did not move. The budget came out of the data-responsive categories, which fell. Spreading effort across a zero-sum budget doesn't average the gains; it dilutes them. v7 is the failure that made v8 possible.

---

## Experiment 6 — Concentrate on the one gap that is both open and responsive (v8 = Nesso2)

**Hypothesis.** Across all five earlier models, one category was uniformly weak: **no-tool discrimination** — answering directly when a tool is present but unnecessary. Every model scored 1–2 out of 10; Qwen scored 8. Crucially, this gap is *data-responsive* (Qwen proves it is learnable at this size) and *chat-safe* (a natural direct answer is a good conversational answer, unlike a terse refusal). So: fix only this, and take the budget from the capacity-bound categories that never responded anyway.

**Experiment.** We multiplied the no-tool data roughly **4.5×**, generated richly — real questions about capitals, currencies, definitions, general facts, with the *tempting* tool deliberately present in context and the correct behavior being a direct, natural answer. We reclaimed that budget from different-tool parallel and multi-argument. We kept the multi-step trajectory data that v7 had shown was responsive.

**Conclusion.** The bet paid, cleanly. No-tool discrimination went **2 → 7**, multi-step **1 → 6**, and firing on real tool calls held (single 9, parallel-same 10). The total reached **68 — the best of the six**, one point above Qwen. And because the no-tool data was natural language rather than refusal boilerplate, it *helped* conversation: Nesso2 posts the best Italian chat of the whole lineage (**4.40/10**, tied with the reference model) and the highest correctness of the line. One data-responsive skill, trained richly, funded by skills that don't respond to data.

---

## Two detours that did not work — kept for the record

**The DPO detour.** *Hypothesis:* preference-tuning the v3 checkpoint toward more detailed answers would raise quality. *Experiment:* a curriculum DPO run. *Conclusion:* conversation quality fell from **4.25 to 3.48** for a **+0.008** gain on IFEval. At 0.4B, "more detailed" is a euphemism for "more hallucinated." We shipped the SFT and dropped the DPO.

**The template detour.** *Hypothesis:* a ChatML template with warm-initialized new tokens would serve agentic formatting better. *Experiment:* v5, with resized embeddings. *Conclusion:* it lost to the plain Llama-3 template on every axis. Don't pay the cost of re-embedding for a chat format the base model never saw.

**The decoding trap.** One more, because it bit us in the released model card. *Hypothesis:* the decoding settings we validated for conversation — a repetition penalty to prevent loops — apply everywhere. *Experiment:* we used them for tool calls. *Conclusion:* they corrupt tool JSON. The prompt template already contains the words `name` and `arguments`, so a repetition penalty suppresses exactly those tokens, and the model emits `"Name"` or drops the arguments entirely. **Structured output wants pure greedy; the repetition penalty is only for prose.**

---

## What Nesso2 is, and is not

The honest scorecard:

| | Nesso2 | Qwen3-0.6B |
|---|---|---|
| Agentic total (/100) | **68** | 67 |
| Agentic — Italian (/50) | **35** | 29 |
| Agentic — English (/50) | 33 | **38** |
| Italian academic avg | 0.334 | **0.336** |
| Italian conversation (/10) | **4.40** | 2.80 |

Nesso2 is the best agentic model of the set and the best at Italian tool use by a clear margin, with no knowledge tax and the best Italian conversation of its family. It is **not** the best at English tool use — Qwen keeps that — and it does not out-know a model trained on far more data on English MMLU. One category, observation grounding, regressed as a side effect of the no-tool push (8 → 3 in the benchmark); notably it did not appear in the conversation judge, so we read it as a narrow evaluation artifact, and we flag it anyway.

None of that dents the claim, because the claim was always specific: for **Italian agentic tool use in the sub-billion-parameter class**, this is the strongest open model we could find.

---

## The transferable recipe

The method generalizes beyond this model, and it is what we carry to the 3B:

1. **Put knowledge in a CPT stage**, so the fine-tune can specialize without forgetting.
2. **Measure on several families and read them together**, because a single number lies at this scale.
3. **Diagnose each skill as data-responsive or capacity-bound** before spending a single example.
4. **Concentrate the budget on one open, responsive, side-effect-free gap** — don't spread it.
5. **Keep the failures in the notebook.** v7 and the DPO run taught us more than most of the wins.

The full recipe, the three evaluation suites, and a detailed visual report live alongside this post in the [`nesso2/`](https://github.com/mii-llm/zagreus-nesso-slm/tree/main/nesso2) folder. The model is [`mii-llm/nesso2-0.4B-agentic`](https://huggingface.co/mii-llm/nesso2-0.4B-agentic).

*Part of the [Zagreus–Nesso project](https://github.com/mii-llm/zagreus-nesso-slm) by [mii-llm](https://mii-llm.ai) — built from scratch on Seeweb HPC. Apache-2.0.*
