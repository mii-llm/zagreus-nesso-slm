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
- function-calling
- agentic
- structured-output
- tool-use
- llama
- nanotron
- trl
base_model: mii-llm/zagreus-0.4B-ita
model_type: llama
pipeline_tag: text-generation
library_name: transformers
---

# Nesso2-0.4B-Agentic

**Nesso2-0.4B-Agentic** is a bilingual Italian/English Small Language Model (SLM) optimized for **function calling, structured output generation, and multi-step agentic execution**, with a deliberate emphasis on **Italian** tool use. It is post-trained on top of a knowledge-enriched, long-context checkpoint of [Zagreus-0.4B-ita](https://huggingface.co/mii-llm/zagreus-0.4B-ita) — a foundational model trained from scratch by the [mii-llm](https://mii-llm.ai) community (*Made in Italy – Large Language Model*) on the [Seeweb](https://www.seeweb.it) HPC infrastructure.

Designed for **sovereign edge inference**, Nesso2-0.4B-Agentic targets deployment scenarios that require reliable tool use, structured JSON output, correct *tool vs. no-tool* discrimination, and multi-step agentic reasoning — all within a compact ~0.4B parameter footprint and a **32k-token context window**.

On our bilingual function-calling benchmark it is, to our knowledge, the **strongest open SLM for Italian agentic tool use in its size class**, edging out `Qwen3-0.6B` overall and beating it by a wide margin on Italian.

> ⚠️ This model is at the **SFT (Supervised Fine-Tuning)** stage. DPO (Direct Preference Optimization) is planned; updated results will be published upon completion.

---

## Model Details

| Property | Value |
|---|---|
| **Architecture** | Llama-style (dense, GQA) |
| **Parameters** | ~438M |
| **Hidden size** | 960 |
| **Layers** | 32 |
| **Attention heads** | 15 (KV heads: 5) |
| **Head dim** | 64 |
| **Intermediate size** | 2560 |
| **Context length** | 32,768 tokens |
| **RoPE theta** | 1,000,000 |
| **Tokenizer** | Llama-3 (`vocab_size`: 128,256) |
| **Tied embeddings** | Yes |
| **Precision** | BF16 |
| **Languages** | Italian, English |
| **Base model** | mii-llm/zagreus-0.4B-ita |
| **Post-training framework** | TRL (SFTTrainer) + FSDP |
| **Chat template** | Llama-3 (with tool-calling extension) |

---

## Lineage

Unlike a single-stage SFT model, Nesso2-0.4B-Agentic is the tip of a multi-stage pipeline designed to give a tiny model both **knowledge** and **agentic skill**:

```
zagreus-0.4B-ita  (base, pre-trained from scratch, ~1T tokens)
        │
        ▼
  + Knowledge CPT           continued pre-training on a curated knowledge
        │                   corpus (Italian/English Wikipedia, augmented QA)
        ▼                   to lift factual/MMLU capability
  + 32k Context Extension   long-context adaptation (RoPE θ = 1e6)
        │
        ▼
  + Agentic SFT (v8)        supervised fine-tuning on bilingual instruction +
                            function-calling data (this model)
```

The **knowledge CPT** stage is what separates this model from a plain SFT on the same base: it measurably improves factual benchmarks (see Evaluation) and is the reason Italian MMLU/ARC hold up despite the heavy agentic specialization.

---

## Training Details

### Base Model Pre-training

The foundation, `Zagreus-0.4B-ita`, was pre-trained on approximately **1 trillion tokens**:

| Dataset | Description |
|---|---|
| [FineWeb (350BT sample)](https://huggingface.co/datasets/HuggingFaceFW/fineweb/viewer/sample-350BT) | ~350B tokens of English web text |
| [FineWeb-2 (ita_Latn)](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/viewer/ita_Latn) | Italian web text |
| [FinePDFs (ita_Latn)](https://huggingface.co/datasets/HuggingFaceFW/finepdfs/viewer/ita_Latn) | Italian PDF documents |
| [StarCoder Data](https://huggingface.co/datasets/bigcode/starcoderdata) | ~250B tokens of code |

**Token distribution**: ~400B English + ~400B Italian + ~200B Code
**Infrastructure**: 64× NVIDIA A100 (8 nodes × 8 GPUs) on Seeweb HPC
**Framework**: [Nanotron (mii-llm fork)](https://github.com/mii-llm/nanotron)

### Knowledge CPT + 32k Context Extension

Continued pre-training (Nanotron) on a curated knowledge corpus to break the small-model MMLU ceiling, followed by a long-context extension stage to 32,768 tokens (RoPE θ raised to 1e6). Run on the [Seeweb](https://www.seeweb.it) HPC infrastructure.

### Post-training (Agentic SFT)

Supervised fine-tuning with **TRL** (`SFTTrainer`) + **FSDP**, on the [Seeweb](https://www.seeweb.it) HPC infrastructure.

The instruction dataset is a **bilingual (Italian/English)** mixture combining broad conversational/instruction data with a **synthetic function-calling corpus** covering single- and parallel tool calls, argument disambiguation, missing-argument handling, unavailable-tool refusal, observation grounding, multi-step trajectories, and — critically — **no-tool discrimination** (answering directly when a tempting tool is present but unnecessary). Tool schemas and argument names are randomized to discourage memorization.

**Key hyperparameters:**

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW (fused) |
| Learning rate | `1e-3` |
| LR scheduler | Cosine with min-LR floor (`min_lr_rate` = 0.3) |
| Warmup ratio | 0.03 |
| Epochs | 3 |
| Per-device batch size | 2 |
| Gradient accumulation | 8 |
| Sequence length | 8192 |
| Gradient checkpointing | On |
| Precision | BF16 |
| FSDP strategy | FULL_SHARD |
| EOS token | `<\|eot_id\|>` (128009) |
| Pad token | `<\|finetune_right_pad_id\|>` (128004) |

---

## Chat Template

This model uses the **Llama-3** conversation format (**not** ChatML). Tools are provided through the `tools` argument of `apply_chat_template`, and the model emits calls as Hermes-style `<tool_call>` blocks.

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant with access to tools.<|eot_id|><|start_header_id|>user<|end_header_id|>

What is the weather in Rome today?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

A tool call looks like:

```
<tool_call>
{"name": "get_weather", "arguments": {"city": "Roma"}}
</tool_call>
```

Special tokens:
- `bos_token`: `<|begin_of_text|>` (128000)
- `eos_token`: `<|eot_id|>` (128009)
- `pad_token`: `<|finetune_right_pad_id|>` (128004)

> ⚠️ The saved inference template does **not** emit the BOS token itself — tokenize the rendered string with `add_special_tokens=True` (as shown below) so that exactly one BOS is prepended. Do not double-add it.

---

## Usage

### Function calling

```python
import re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "mii-llm/nesso2-0.4B-agentic"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
).eval()

def chat(messages, tools=None, max_new_tokens=256):
    # Render with the Llama-3 tool template, then tokenize adding exactly one BOS.
    prompt = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(model.device)
    n = inputs["input_ids"].shape[1]

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,                 # PURE greedy — best for structured tool calls
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    text = tokenizer.decode(out[0][n:], skip_special_tokens=False)
    answer = re.split(r"<\|eot_id\|>|<\|end_of_text\|>", text)[0].strip()

    calls = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", answer, flags=re.S)
    return answer, calls

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Ritorna il meteo per una città",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]

messages = [
    {"role": "system", "content": "Sei un assistente che può usare strumenti quando servono."},
    {"role": "user", "content": "Che tempo fa a Milano?"},
]

answer, calls = chat(messages, tools=tools)
print("ANSWER:", answer)
print("CALLS :", calls)   # -> [{"name": "get_weather", "arguments": {"city": "Milano"}}]
```

> 💡 **Tip**: For **function calling and structured output**, use **pure greedy** decoding (`do_sample=False`, no repetition penalty). The tool-call JSON is short and the prompt template already contains the structural tokens (`name`, `arguments`, quotes, braces) — a repetition penalty or `no_repeat_ngram_size` will suppress exactly those tokens and corrupt the JSON (e.g. emitting `"Name"` or dropping `arguments`). For **long free-form conversation**, a light `repetition_penalty` (≈1.15) can help avoid loops on a model this small, but keep it **off** for tool calls.

### Plain conversation (no tools)

For free-form chat, drop the `tools` argument and add a light repetition penalty to keep a model this small from looping. The same `tokenizer`/`model` loaded above are reused.

```python
def chat_plain(messages, max_new_tokens=256):
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(model.device)
    n = inputs["input_ids"].shape[1]

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.15,         # light penalty helps free-form text (NOT for tool calls)
        no_repeat_ngram_size=6,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    text = tokenizer.decode(out[0][n:], skip_special_tokens=False)
    return re.split(r"<\|eot_id\|>|<\|end_of_text\|>", text)[0].strip()

# Single turn
messages = [
    {"role": "system", "content": "Sei un assistente utile e conciso."},
    {"role": "user", "content": "Spiegami in due frasi cosa è il machine learning."},
]
reply = chat_plain(messages)
print(reply)

# Multi-turn: append the assistant reply and the next user turn, then call again.
messages.append({"role": "assistant", "content": reply})
messages.append({"role": "user", "content": "Fammi un esempio concreto."})
print(chat_plain(messages))
```

Example output for the single-turn call:

```
Il machine learning (ML) è una branca dell'intelligenza artificiale che permette ai
computer di apprendere dai dati, migliorando le proprie prestazioni senza essere
esplicitamente programmati per un compito specifico.
```

---

## Evaluation

Three complementary evaluation families were used. Academic benchmarks were run with our [fork of lm-evaluation-harness](https://github.com/mii-llm/lm-evaluation-harness/); agentic and conversational quality were measured with dedicated bilingual test suites.

### 1. Academic benchmarks

MMLU is 5-shot `acc`; HellaSwag / ARC are 0-shot `acc_norm`; IFEval is `inst_level_loose_acc` (generative, chat template). All numbers are `acc` on a 0–1 scale.

#### Italian

| Model | IFEval IT ↑ | ARC IT ↑ | HellaSwag IT ↑ | MMLU IT ↑ | **Avg IT** |
|---|---|---|---|---|---|
| Qwen/Qwen3-0.6B | **0.3058** | 0.3040 | 0.3598 | **0.4025** | **0.3355** |
| **Nesso2-0.4B-agentic** | 0.2960 | 0.3040 | **0.4090** | 0.3260 | 0.3338 |
| mii-llm/nesso-0.4B-agentic | 0.3120 | 0.3010 | 0.4070 | 0.2820 | 0.3255 |

#### English

| Model | IFEval EN ↑ | ARC EN ↑ | HellaSwag EN ↑ | MMLU EN ↑ | **Avg EN** |
|---|---|---|---|---|---|
| Qwen/Qwen3-0.6B | 0.2758 | **0.3430** | **0.4742** | **0.4013** | **0.3736** |
| **Nesso2-0.4B-agentic** | 0.3790 | 0.3040 | 0.4730 | 0.2700 | 0.3565 |
| mii-llm/nesso-0.4B-agentic | **0.4120** | 0.3040 | 0.4690 | 0.2400 | 0.3563 |

#### Overall

| Model | Avg IT | Avg EN | **Overall** |
|---|---|---|---|
| Qwen/Qwen3-0.6B | 0.3355 | 0.3736 | **0.3545** |
| **Nesso2-0.4B-agentic** | 0.3338 | 0.3565 | 0.3451 |
| mii-llm/nesso-0.4B-agentic | 0.3255 | 0.3563 | 0.3409 |

**Takeaways.** On **Italian** academics, Nesso2-0.4B-agentic effectively **ties Qwen3-0.6B** (0.3338 vs 0.3355) and leads it on Italian HellaSwag and ARC — the knowledge CPT stage closes the gap that similarly-sized SLMs usually cede to Qwen. It also **outperforms its sibling `nesso-0.4B-agentic` on MMLU** in both languages (Italian 0.326 vs 0.282; English 0.270 vs 0.240), which is precisely the CPT stage paying off. Qwen retains a clear edge only on MMLU (a knowledge-heavy benchmark favoring its far larger pre-training budget).

### 2. Agentic function calling (bilingual, 100 cases) ⭐

Our frozen function-calling suite: 100 bilingual cases across 10 categories (single / parallel-same / parallel-different tool calls, multi-argument, disambiguation, missing-argument, unavailable-tool refusal, **no-tool discrimination**, observation grounding, multi-step). Greedy decoding, Hermes `<tool_call>` format, per-category automatic scoring.

| Model | Italian /50 | English /50 | **Total /100** |
|---|---|---|---|
| **Nesso2-0.4B-agentic** | **35** | 33 | **68** |
| Qwen/Qwen3-0.6B | 29 | **38** | 67 |

Nesso2-0.4B-agentic is **best overall** and **decisively ahead on Italian tool use (+6)**, while Qwen keeps an English advantage. This is the benchmark the model is optimized for, and where its real-world value over general-purpose SLMs shows.

### 3. Conversational quality (LLM-as-judge)

20 bilingual multi-turn tasks per language, graded 1–10 by `Qwen3.6-35B-A3B` on correctness / language-fidelity / helpfulness (greedy answers). Mean overall score:

| Model | Italian ↑ | English ↑ | **Both** |
|---|---|---|---|
| mii-llm/nesso-0.4B-agentic | 4.40 | **6.40** | **5.40** |
| Qwen/Qwen3-0.6B | 2.80 | 5.80 | 4.30 |
| **Nesso2-0.4B-agentic** | **4.40** | 3.80 | 4.10 |

Despite its agentic specialization, Nesso2-0.4B-agentic delivers the **best Italian conversational quality of its lineage** (4.40, tied with `nesso-0.4B-agentic`) and **strongly outscores Qwen3-0.6B in Italian chat** (4.40 vs 2.80). English conversation remains its relative weak spot.

### Discussion

Nesso2-0.4B-agentic is a **task-specialized** model: its post-training prioritizes structured-output fidelity, tool-calling accuracy, no-tool discrimination, and agentic planning. Thanks to the knowledge-CPT stage, this specialization comes **without the usual academic tax on Italian** — the model matches Qwen3-0.6B on Italian benchmarks and beats it on the agentic suite, while remaining a genuinely useful Italian conversationalist. Its edge over general-purpose SLMs of similar size is best assessed on **agentic and function-calling tasks**, not academic leaderboards.

---

## Related Models

| Model | Description |
|---|---|
| [Zagreus-0.4B-ita](https://huggingface.co/mii-llm/zagreus-0.4B-ita) | Base pre-trained model (this model's foundation) |
| [Nesso-0.4B-agentic](https://huggingface.co/mii-llm/nesso-0.4B-agentic) | Sibling agentic SFT trained directly on the base (no CPT) |
| [Nesso-0.4B-instruct](https://huggingface.co/mii-llm/nesso-0.4B-instruct) | Optimized for conversational and instruction-following tasks |

---

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{zagreus2025,
  title        = {The Joy and Pain of Training an LLM from Scratch:
                  A Technical Report on the Zagreus and Nesso Model Families},
  author       = {mii-llm community},
  year         = {2025},
  howpublished = {\url{https://github.com/mii-llm/zagreus-nesso-slm}},
}
```

---

## Acknowledgements

- **Antonio Baldassarra** (CEO, Seeweb) and **Marco Cristofanilli** (Head of AI, Seeweb) for infrastructure sponsorship
- The **Hugging Face** team for Nanotron, datatrove, FineWeb, and FineWeb-2
- The **mii-llm** open-source community

---

## License

Released under the **Apache 2.0** license.

> Made with ❤️ in Italy by [mii-llm](https://mii-llm.ai)
