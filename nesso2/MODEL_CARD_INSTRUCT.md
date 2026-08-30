---
language:
- it
- en
license: apache-2.0
tags:
- small-language-model
- slm
- edge-ai
- italian
- bilingual
- conversational
- instruct
- function-calling
- agentic
- reinforcement-learning
- rlvr
base_model: giux78/zagreus-0.4B-cpt-full-32k-sft-agentic-v8
pipeline_tag: text-generation
---

# Nesso2-0.4B-Instruct

**Nesso2-0.4B-Instruct** is a bilingual Italian/English Small Language Model (SLM) optimized as the family's **best conversationalist at no-think speed**, while keeping the agentic tool-calling ability of its parent. It is the reinforcement-learning refinement of [**Nesso2-0.4B-Agentic**](MODEL_CARD.md) (v8), post-trained by the [mii-llm](https://mii-llm.ai) community (*Made in Italy – Large Language Model*) on the [Seeweb](https://www.seeweb.it) HPC infrastructure.

It is a **single, fast native mode** — no thinking block, no mode markers. It simply responds, in ~0.78 s per turn.

## Two models, one family

| | **Nesso2-0.4B-Agentic** (v8) | **Nesso2-0.4B-Instruct** (this model) |
|---|---|---|
| Focus | Execution-first tool-caller | Best conversationalist, fast |
| Method | Agentic SFT | Chat-anchored RLVR on top of v8 |
| Italian chat (judge /10) | 4.55 | 4.30 |
| Overall chat (judge /10) | 4.12 | **4.30** |
| English chat (judge /10) | 3.70 | **4.30** |
| Observation grounding (100-case) | 3 / 10 | **5 / 10** |
| Function calling | strongest | held (±1-2 cases) |

Pick **Instruct** for assistant/chat-heavy bilingual deployments where conversation quality matters most; pick **Agentic** when raw execution tool-calling is the priority.

## Model Details

- **Architecture:** ~438M-parameter dense decoder (Llama-3-family), 32 layers, hidden 960, 15 attention heads / 5 KV heads (GQA), RoPE θ = 1e6, vocab 128,256, tied embeddings. Long-context (32k) capable.
- **Mode:** single native mode — **no** `/think` · `/no_think` markers.
- **Tokenizer / template:** Llama-3 chat format; `bos = <|begin_of_text|>` (128000), `eos = <|eot_id|>` (128009), pad `<|finetune_right_pad_id|>`.
- **Languages:** Italian (primary) and English.
- **License:** Apache-2.0.

## Lineage

```
Zagreus-0.4B-ita (from-scratch base, Seeweb)
  → knowledge CPT + 32k context extension
  → Agentic SFT ...................... Nesso2-0.4B-Agentic (v8)
  → Chat-anchored RLVR ............... Nesso2-0.4B-Instruct  (this model)
```

## Training — chat-anchored RLVR

Instruct is produced by **GRPO (reinforcement learning)** from v8 with a **two-part reward evaluated in the same batch**:

- **On tool inputs** → reward *observation grounding*: 0 if the model re-calls a tool instead of using the returned result, up to 1 if it grounds its answer on the observation value.
- **On chat inputs** → reward *staying v8*: token-F1 between the model's answer and v8's own response to that prompt.

A **low KL coefficient** (β = 0.005) lets the policy actually move, while the **chat reward — not the KL term — holds conversation to v8**. This is the objective SFT cannot express: *"improve observation grounding **while remaining v8 on conversation**."* Standard imitation (SFT / distillation) has no signal to protect chat and always trades Italian conversation for agentic gains; the reward-level anchor is what avoids that.

**Key hyperparameters:** GRPO from v8 · 8 rollouts/prompt · β 0.005 · learning-rate 1e-5 · 800 steps · greedy-decoded reward rollouts. Mixed data: ~40% observation-grounding cases, ~60% conversation (Italian-weighted) with v8 as the anchor target.

> A practical lesson worth recording: a **large** KL-to-reference coefficient caps all policy movement (KL plateaus at ~0.005 regardless of learning rate). Keep β low and let the **reward** protect what you want preserved.

## Chat Template

Identical to Nesso2-0.4B-Agentic (standard Llama-3). Use the tokenizer's `apply_chat_template`. For **function calling**, tools are rendered into the system message; for **plain chat**, no tools are needed. See [MODEL_CARD.md](MODEL_CARD.md#chat-template) for the full template and worked examples — the API is the same.

**Decoding:** **pure greedy** for tool calls (a repetition penalty corrupts the JSON); a light `repetition_penalty ≈ 1.15` is fine for free-form chat.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL = "mii-llm/nesso2-0.4B-instruct"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto").eval()

# Plain conversation (Italian) — the model's strong suit
messages = [
    {"role": "system", "content": "Sei un assistente utile."},
    {"role": "user", "content": "Il robot R-47 è pronto. Conferma brevemente e indica che può iniziare il turno."},
]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(**{"input_ids": inputs}, max_new_tokens=256, do_sample=False,
                     repetition_penalty=1.15, no_repeat_ngram_size=6, eos_token_id=tok.eos_token_id)
print(tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True))
```

For **function calling**, render tools into the system message and decode with **pure greedy** (no repetition penalty). The tool-calling interface is identical to [Nesso2-0.4B-Agentic](MODEL_CARD.md#function-calling).

## Evaluation

All numbers are versus v8 (Nesso2-0.4B-Agentic), on the same graders, at the same ~0.78 s single-pass latency. **Nothing regressed.**

### Conversation quality (LLM-as-judge, `Qwen3.6-35B-A3B`)

20 bilingual multi-turn tasks / language, graded 1–10 on correctness / language-fidelity / helpfulness.

| model | Italian | English | Overall | corr | help | lang |
|---|---:|---:|---:|---:|---:|---:|
| **Nesso2-0.4B-Instruct** | 4.30 | **4.30** | **4.30** | **4.08** | **4.42** | **9.10** |
| Nesso2-0.4B-Agentic (v8) | 4.55 | 3.70 | 4.12 | 3.88 | 4.12 | 8.65 |
| Qwen3-0.6B | 2.70 | 5.80 | 4.25 | 4.15 | 4.20 | 7.03 |

Instruct is the **best overall conversationalist of the family**, improving correctness, helpfulness and fluency over v8, and **closing most of the English conversation gap** (3.70 → 4.30) while keeping Italian essentially intact (−0.25). On **Italian conversation it crushes Qwen3-0.6B** (4.30 vs 2.70), and on an independent **robot-domain** conversation suite it leads in *both* languages (Italian 87.5% vs 60.1%; English 77.5% vs 75.0%).

### Agentic function calling (bilingual, 100 cases)

| model | total | observation grounding |
|---|---:|---:|
| **Nesso2-0.4B-Instruct** | **70 / 100** | **5 / 10** |
| Nesso2-0.4B-Agentic (v8) | 68 / 100 | 3 / 10 |

Observation grounding — v8's weakest category — improves, with no other category regressing. On an independent partial-credit FC benchmark, Instruct holds function-calling (Italian 63.2% vs v8 63.1%; exact-match 31/100 vs 33/100 — a statistical tie).

### Academic (MMLU / HellaSwag / ARC / IFEval, it + en)

Flat versus v8 (all within ±0.01) — the RL stage does not touch knowledge benchmarks.

## Limitations

- **Production decision skills** (abstention when no tool applies, observation grounding, multi-step, disambiguation) remain the family's weak point. On a hard production-robot suite the model can **call a tool when none is appropriate**; abstention is domain- and phrasing-sensitive and does not fully generalize. These are the target of the next RL iteration.
- **English function-calling** trails Qwen3-0.6B on abstention/observation-heavy tasks.
- **Raw knowledge (MMLU)** — Qwen retains an edge from its far larger pre-training budget.
- Single mode only — there is no test-time reasoning / thinking mode.

## Citation

```bibtex
@misc{nesso2_04b_instruct_2026,
  title  = {Nesso2-0.4B-Instruct: chat-anchored RL for a bilingual Italian agentic SLM},
  author = {mii-llm community},
  year   = {2026},
  note   = {https://mii-llm.ai}
}
```
