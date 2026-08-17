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
| **Italian academic avg** (acc) | 0.334 | 0.336 |
| **Italian conversation** (LLM-judge, /10) | **4.40** | 2.80 |

**Takeaways.** Nesso2 is the **best agentic model overall** and leads Italian tool use by **+6**, while effectively **tying Qwen3-0.6B on Italian academics** (no knowledge tax) and posting the **best Italian conversation score of its lineage**. The trade is explicit: Qwen keeps English tool use and raw English MMLU.

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
- **Observation grounding** regressed (8 → 3) — the no-tool data bled into it; did *not* appear in the conversation judge, so likely a narrow eval-format artifact, flagged regardless.
- **Raw knowledge:** Qwen stays ahead on English MMLU.

---

## Repository layout

```
nesso2/
├── README.md                        this file
├── MODEL_CARD.md                    the Hugging Face model card
├── report.html                      full visual technical report
├── eval/
│   ├── agentic_eval_100.py          Family B1 — 100-case function-calling suite (6 models)
│   ├── agentic_eval_cases_100.json  the frozen bilingual eval cases + tools
│   ├── conv_gen.py                  Family B2 — generate conversation answers
│   ├── judge_conversations.py       Family B2 — LLM-as-judge scorer (35B via vLLM)
│   ├── conv_prompts.json            the conversation eval tasks
│   └── bench_academic.sh            Family A — lm-eval-harness runner (it+en)
└── training/
    ├── sbatch-sft-agentic-v8.sh     agentic SFT (TRL + FSDP, Slurm)
    ├── build_perlang.py             per-language quality-floor data prep (IT-all + EN ≥ 0.6)
    └── push_to_hf.py                convert + push checkpoint (swaps in the tool template)
```

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
