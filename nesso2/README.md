# Nesso2-0.4B-Agentic

**A ~0.4B bilingual (Italian/English) small language model built for function calling and agentic execution — and, to our knowledge, the strongest open model for Italian agentic tool use in the sub-billion-parameter class.**

- 🤗 Model: [`mii-llm/nesso2-0.4B-agentic`](https://huggingface.co/mii-llm/nesso2-0.4B-agentic) *(release name)*
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

## How it was built — four stages

| Stage | What | Where |
|---|---|---|
| 1 · Pre-training | ~1T tokens (~400B EN / ~400B IT / ~200B code) from FineWeb, FineWeb-2, FinePDFs, StarCoder | 64× A100 · Seeweb · Nanotron |
| 2 · Knowledge CPT | Continued pre-training on curated Wikipedia + augmented QA to lift factual/MMLU capability | Seeweb · Nanotron |
| 3 · 32k context | Long-context extension, RoPE θ = 1e6 | Seeweb |
| 4 · Agentic SFT | Bilingual instruction + synthetic function-calling corpus (randomized tool schemas) | TRL SFTTrainer + FSDP · Llama-3 template · 3 ep · LR 1e-3 cosine |

The **knowledge CPT (stage 2)** is what separates Nesso2 from a plain SFT on the same base — it is why Italian MMLU/ARC survive the agentic specialization (Nesso2 beats its no-CPT sibling `nesso-0.4B-agentic` on MMLU in both languages).

---

## Evaluation — three families

We read three families **together**, because at 0.4B they disagree and the disagreement is the signal.

- **Family A — Academic** (`eval/bench_academic.sh`): MMLU (5-shot acc), HellaSwag/ARC (0-shot acc_norm), IFEval (generative), Italian + English, via the [mii-llm lm-evaluation-harness fork](https://github.com/mii-llm/lm-evaluation-harness/).
- **Family B1 — Agentic function calling** (`eval/agentic_eval_100.py`): a frozen bilingual **100-case** suite, 10 categories, 50 IT / 50 EN, Hermes `<tool_call>` format, **pure greedy** decoding, automatic per-category grader. Cases in `eval/agentic_eval_cases_100.json`.
- **Family B2 — Conversation** (`eval/conv_gen.py` → `eval/judge_conversations.py`): 20 multi-turn tasks per language, graded 1–10 on correctness / language-fidelity / helpfulness by **Qwen3.6-35B-A3B**. Prompts in `eval/conv_prompts.json`.

### Agentic capability profile (x/10)

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
