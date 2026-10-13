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
datasets:
- Salesforce/xlam-function-calling-60k
- Team-ACE/ToolACE
- nvidia/When2Call
model_type: llama
pipeline_tag: text-generation
library_name: transformers
---

# Nesso3-0.4B-Agentic

**Nesso3-0.4B-Agentic** is a bilingual Italian/English Small Language Model (SLM) for **function calling, structured output and multi-step agentic execution**, with a deliberate emphasis on **Italian** tool use. It is the successor of [Nesso2-0.4B-Agentic](https://huggingface.co/mii-llm/nesso2-0.4B-agentic), post-trained on the same knowledge-enriched, 32k-context checkpoint of [Zagreus-0.4B-ita](https://huggingface.co/mii-llm/zagreus-0.4B-ita) — a foundation model trained from scratch by the [mii-llm](https://mii-llm.ai) community (*Made in Italy – Large Language Model*) on the [Seeweb](https://www.seeweb.it) HPC infrastructure.

What changed versus v1 is **how the model is taught to use tools**: instead of memorising a fixed toolbox, it learns **one compact serving contract** and **generalises across thousands of procedurally-generated tool schemas** (2,531 distinct tools in the synthetic corpus), plus public function-calling data (xLAM, ToolACE, When2Call; about one third translated to Italian). The goal is a model that keeps working when the tools it is given are **new**, which is the situation you are in at deployment; on our novel-tool robot benchmark it gains +10 points over v1 (with the caveats reported below).

> ⚠️ This model is at the **SFT (Supervised Fine-Tuning)** stage. No RL, DPO or distillation stage was applied to this checkpoint. (We did try on-policy distillation and chat-anchored RL on top of earlier SFT checkpoints; neither made this release. The full experiment log, with the failures, is in the [GitHub report](https://github.com/mii-llm/zagreus-nesso-slm/tree/main/nesso3).)
>
> ⚠️ **Read the Usage section before integrating.** This model does **not** use the `tools=` argument of `apply_chat_template` (its Llama-3 template ignores it). Tool schemas go **inside the system message**, in a compact `<tools>` block — see [Chat template & tool contract](#chat-template--tool-contract). Passing `tools=` silently drops them and the model will behave as if it had none.

**Naming.** Nesso3 is the third generation of the Nesso family (Nesso → Nesso2 → Nesso3). The number is the generation, not the parameter count: this model has ~0.4B parameters.

📚 **Evidence.** Experiment log (what worked and what did not), raw result files and evaluation scripts: [github.com/mii-llm/zagreus-nesso-slm/nesso3](https://github.com/mii-llm/zagreus-nesso-slm/tree/main/nesso3).

---

## Highlights

| Benchmark | Nesso3-0.4B-Agentic | Nesso2-0.4B-Agentic (v1) | Qwen3-0.6B |
|---|---|---|---|
| Bilingual function calling, 100 cases ↑ | **74** | 68 | 67 (non-thinking) · 76 (thinking) |
| &nbsp;&nbsp;↳ Italian half (/50) | **38** | 35 | 29 |
| Held-out Italian benchmark, healthcare booking (40 cases) ↑ | **68.5** | 65.3 | 63.9 |
| Production robot benchmark, overall (84 cases) ↑ | 63.6 | 53.3 | **71.1** |
| Generic English tools stress test (33 scenarios) ↑ | 48 | – | **84** |
| Academic average, IT+EN (MMLU / ARC / HellaSwag / IFEval) ↑ | 0.3555 | 0.3531 | **0.3955** |
| Conversation, LLM-as-judge, both languages (/10, mean of 3 runs) ↑ | 3.94 | 4.18 | **4.26** |

In short: **the strongest of our models for Italian tool use — ahead of Qwen3-0.6B on the Italian half of the 100-case suite and on the unseen Italian healthcare benchmark — but not a Qwen replacement for English-first or generic-tool deployments**, where Qwen3-0.6B is clearly ahead (see [Limitations](#limitations); we report the benchmarks where it loses next to those where it wins).

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
| **Base model** | mii-llm/zagreus-0.4B-ita (+ knowledge CPT + 32k extension) |
| **Post-training framework** | TRL (SFTTrainer), 8-GPU data parallel |
| **Chat template** | Llama-3; tool schemas inline in the system message (see below) |

---

## Lineage

```
zagreus-0.4B-ita  (base, pre-trained from scratch, ~1T tokens)
        │
        ▼
  + Knowledge CPT           continued pre-training (~4.5B tokens) on a curated
        │                   knowledge corpus: Italian/English Wikipedia with
        ▼                   grounded QA augmentation + exam-style QA
  + 32k Context Extension   long-context adaptation (RoPE θ = 1e6, ~1.4B tokens)
        │
        ▼
  + Agentic SFT (compact contract)   this model
```

The knowledge CPT stage is what keeps Italian MMLU/ARC/HellaSwag competitive despite heavy agentic specialisation; see the [Nesso2-0.4B-Agentic card](https://huggingface.co/mii-llm/nesso2-0.4B-agentic) for the academic benchmarks of the shared base.

---

## Training Details

### Base pre-training, knowledge CPT, 32k extension

Identical to [Nesso2-0.4B-Agentic](https://huggingface.co/mii-llm/nesso2-0.4B-agentic) (Nanotron, [mii-llm fork](https://github.com/mii-llm/nanotron), on the [Seeweb](https://www.seeweb.it) HPC infrastructure): ~1T-token pre-training (FineWeb, FineWeb-2 ita, FinePDFs ita, StarCoder), then knowledge CPT and a 32k extension.

### Agentic SFT

Supervised fine-tuning with **TRL** (`SFTTrainer`), single node / 8 GPUs, on the [Seeweb](https://www.seeweb.it) HPC infrastructure. The data mixture has three parts:

| Component | Size | What it teaches |
|---|---|---|
| **Conversational floor** (omnia: all Italian, English score ≥ 0.6) | ~2.27M rows, ×1 | General chat and instruction following; keeps the model a usable conversationalist |
| **Synthetic compact function-calling corpus** | 36k rows, ×4 | Single / parallel (same and different tools) / multi-argument calls, argument disambiguation, missing-argument questions, unavailable-tool refusal, *no-tool* answering, observation grounding, multi-step trajectories — over **2,531 procedurally-generated tool schemas**, 50/50 Italian/English |
| **Public function-calling data** | ~33k rows, ×1 | [xLAM-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) (CC-BY-4.0, ~20k rows: single and parallel calls on real-world APIs), [ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) (Apache-2.0, ~7k rows: tool selection, tool-result turns), [When2Call](https://huggingface.co/datasets/nvidia/When2Call) (CC-BY-4.0, ~5k rows: ask / "cannot do that" decisions). Re-rendered into the compact contract, reasoning traces removed; about one third translated to Italian with Qwen3.6-35B-A3B and filtered (argument values kept verbatim, language and length checks) |

Why this mix: a 0.4B model **locks into one prompt format** and over-fits to a fixed toolbox if trained on a handful of tools. Memorising *one* contract while seeing *thousands* of schemas is what lets it transfer to unseen tools. Decontamination (8-gram overlap on user turns): the synthetic corpus against the 100-case and healthcare sets; the public data additionally against the robot and generic-tools prompts. Note that the robot benchmark is in a domain *close* to part of our synthetic data (warehouse/robotics) with different tool schemas — read it as "novel tools", not "novel domain".

Public function-calling data is credited to its authors: APIGen / xLAM (Salesforce), ToolACE (Team-ACE), When2Call (NVIDIA).

**Key hyperparameters:**

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW (fused) |
| Learning rate | `1e-3` |
| LR scheduler | Cosine with min-LR floor (`min_lr_rate` = 0.3) |
| Warmup ratio | 0.03 |
| Epochs | 3 (5,094 optimizer steps) |
| Effective batch | 128 sequences (2 per device × 8 accumulation × 8 GPUs) |
| Max sequence length | 8,192 |
| Gradient checkpointing | On |
| Precision | BF16 |
| Final train loss / eval loss | 1.307 / 1.26 |
| EOS token | `<\|eot_id\|>` (128009) |
| Pad token | `<\|finetune_right_pad_id\|>` (128004) |

---

## Chat template & tool contract

The model uses the **Llama-3** conversation format (**not** ChatML). Tool schemas are placed **inside the system message**, after a short context sentence, as compact JSON between `<tools>` tags. Tool results come back as a `tool` turn containing the raw JSON. The model emits calls as Hermes-style `<tool_call>` blocks (one per call; several blocks = parallel calls).

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Sei un assistente operativo. Usa gli strumenti disponibili quando serve.
<tools>
[{"type":"function","function":{"name":"get_weather","description":"Meteo attuale per una città","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]
</tools><|eot_id|><|start_header_id|>user<|end_header_id|>

Che tempo c'è a Roma adesso?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

```

A tool call and the follow-up turn look like:

```
<tool_call>
{"name": "get_weather", "arguments": {"city": "Roma"}}
</tool_call>
```
```
<|start_header_id|>tool<|end_header_id|>

{"city": "Roma", "temp_c": 24, "condition": "soleggiato"}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

```

Special tokens:
- `bos_token`: `<|begin_of_text|>` (128000)
- `eos_token`: `<|eot_id|>` (128009)
- `pad_token`: `<|finetune_right_pad_id|>` (128004)

> ⚠️ **Format-lock.** At this size the model is tied to the contract above. Verbose, OpenAI-style tool prompts, ChatML or a different `<tools>` layout will degrade it noticeably. Keep the system context short (one sentence) and the tool list compact (`separators=(",", ":")`).
>
> ⚠️ The saved inference template does **not** emit the BOS token itself — tokenize the rendered string with `add_special_tokens=True` (as below) so exactly one BOS is prepended.

---

## Usage

### Function calling (full loop)

Tested with `transformers` 5.x (the config stores RoPE under `rope_parameters`). With an older `transformers`, check that `model.config.rope_theta` is `1000000.0` before trusting the output.

```python
import re, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "mii-llm/nesso3-0.4B-agentic"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
).eval()

CONTEXT = "Sei un assistente operativo. Usa gli strumenti disponibili quando serve."

def system_with_tools(context, tools):
    """The model's native contract: short context + compact <tools> block inside the system message."""
    return context + "\n<tools>\n" + json.dumps(tools, separators=(",", ":"), ensure_ascii=False) + "\n</tools>"

def chat(messages, max_new_tokens=256, plain=False):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(model.device)  # exactly one BOS
    n = inputs["input_ids"].shape[1]
    extra = dict(repetition_penalty=1.15, no_repeat_ngram_size=6) if plain else {}
    out = model.generate(
        **inputs, max_new_tokens=max_new_tokens,
        do_sample=False,                       # PURE greedy for tool calls
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, **extra,
    )
    text = tokenizer.decode(out[0][n:], skip_special_tokens=False)
    return re.split(r"<\|eot_id\|>|<\|end_of_text\|>", text)[0].strip()

def parse_calls(answer):
    return [json.loads(c) for c in re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", answer, flags=re.S)]

tools = [
    {"type": "function", "function": {"name": "get_weather", "description": "Meteo attuale per una città",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "convert_currency", "description": "Converte un importo tra due valute",
        "parameters": {"type": "object",
            "properties": {"amount": {"type": "number"}, "from": {"type": "string"}, "to": {"type": "string"}},
            "required": ["amount", "from", "to"]}}},
    {"type": "function", "function": {"name": "search_restaurants", "description": "Cerca ristoranti in una città",
        "parameters": {"type": "object",
            "properties": {"city": {"type": "string"}, "cuisine": {"type": "string"}}, "required": ["city"]}}},
]
system = {"role": "system", "content": system_with_tools(CONTEXT, tools)}

# 1) Selects the right tool among several and fills the arguments
ans = chat([system, {"role": "user", "content": "Quanto sono 250 euro in sterline?"}])
print(parse_calls(ans))
# -> [{'name': 'convert_currency', 'arguments': {'amount': 250, 'from': 'EUR', 'to': 'GBP'}}]

# 2) Emits parallel calls for a multi-part request
ans = chat([system, {"role": "user", "content": "Che tempo fa a Roma e a Torino?"}])
print(parse_calls(ans))
# -> [{'name': 'get_weather', 'arguments': {'city': 'Roma'}}, {'name': 'get_weather', 'arguments': {'city': 'Torino'}}]

# 3) Answers directly when no tool is needed, and says so when no tool fits
print(chat([system, {"role": "user", "content": "Qual è la capitale della Francia?"}]))
# -> La capitale della Francia è Parigi.
print(chat([system, {"role": "user", "content": "Prenota un volo per Parigi."}]))
# -> Non posso prenotare un volo: nessuno degli strumenti disponibili lo consente.

# 4) Ask-for-missing-argument
print(chat([system, {"role": "user", "content": "Convertimi 100 euro."}]))
# -> Per procedere ho bisogno di un dato. In quale valuta desideri convertire?

# 5) Close the loop: run the tool, feed the observation back as a `tool` turn
question = "Che tempo c'è a Roma adesso?"
ans = chat([system, {"role": "user", "content": question}])
call = parse_calls(ans)[0]                      # {'name': 'get_weather', 'arguments': {'city': 'Roma'}}
observation = {"city": "Roma", "temp_c": 24, "condition": "soleggiato"}   # <- result of YOUR get_weather()
final = chat([system,
              {"role": "user", "content": question},
              {"role": "assistant", "content": ans},
              {"role": "tool", "content": json.dumps(observation, ensure_ascii=False)}])
print(final)
# -> Ci sono 24°C e la condizione è soleggiata.
```

> 💡 **Decoding.** Use **pure greedy** (`do_sample=False`, no repetition penalty) for tool calls: the call is short JSON and a repetition penalty suppresses exactly the structural tokens (`name`, `arguments`, quotes, braces). For long free-form conversation, a light `repetition_penalty ≈ 1.15` with `no_repeat_ngram_size=6` avoids loops — keep it **off** for tool calls.

### Plain conversation (no tools)

```python
reply = chat([{"role": "system", "content": "Sei un assistente utile e conciso."},
              {"role": "user", "content": "Spiegami in due frasi cosa è il machine learning."}], plain=True)
print(reply)
```

Example output:

```
Il machine learning (ML) è una branca dell'intelligenza artificiale che permette ai computer
di apprendere dai dati, migliorando le proprie prestazioni nel tempo senza essere
esplicitamente programmati per ogni task specifico.
```

---

## Production notes & recommendations

This is a ~0.4B model. Used well it is a fast (≈0.8 s median per call on a single GPU, ~50 generated tokens) and cheap Italian tool-caller; used naively it will make mistakes a 7B model would not. Three application-layer guards make it robust:

1. **Validate arguments before executing.** Check that every required argument value is grounded in the user's message (or the conversation). Treat unsupported values as missing and ask the user. This prevents acting on hallucinated values.
2. **Treat "I need more information" as a soft failure when the request was complete.** On short, fully-specified requests the model sometimes asks for an argument that is already in the prompt (see Limitations). If your application can tell the request was complete, retry once with a rephrased/extended prompt, or fall back to a deterministic parser.
3. **Feed observations back explicitly** (step 5 above) and, if the model re-calls the same tool instead of answering, short-circuit and ask it to answer from the returned data. **After parallel calls, return all results in one `tool` turn (a JSON list).** In a spot test with `get_weather` for Roma and Torino, two separate `tool` turns made the model re-emit the two calls with invented arguments, while a single turn holding the list produced an answer (with a couple of missing spaces in the generated sentence). One example, not a benchmark: test your own flow.

**Decoding:** pure greedy for tool calls; light repetition penalty only for free-form chat.

---

## Evaluation

All benchmarks below are **in-house, small, and bilingual**; the raw result files and the scripts that produced them are in the [GitHub repository](https://github.com/mii-llm/zagreus-nesso-slm/tree/main/nesso3) (`results/`, `eval/summarize_results.py` rebuilds every table); every model is evaluated greedily on **its own native prompt contract** (our model: the compact contract above; Qwen3-0.6B: its native `tools` chat template; Nesso2 v1: its own template), scoring tool name, argument presence/values, call count and abstention automatically. See *Training Details* for what was decontaminated against which suite. They are useful for ranking models of this size, not as absolute guarantees.

### 1. Agentic function calling — 100 bilingual cases ⭐

100 cases (50 IT / 50 EN), 10 categories of 10: single, parallel-same-tool, parallel-different-tools, multi-argument, disambiguation, missing-argument, unavailable-tool, no-tool, observation grounding, multi-step.

| Model | Italian /50 | English /50 | **Total /100** |
|---|---|---|---|
| **Nesso3-0.4B-Agentic** | **38** | 36 | **74** |
| Nesso2-0.4B-Agentic (v1) | 35 | 33 | 68 |
| Qwen3-0.6B (non-thinking) | 29 | **38** | 67 |
| Qwen3-0.6B (thinking) | 34 | 42 | 76 |

Per category (out of 10):

| Category | **Nesso3 (this model)** | Nesso2 v1 | Qwen3-0.6B | Qwen3-0.6B (thinking) |
|---|---|---|---|---|
| single | 7 | 9 | 7 | 8 |
| parallel, same tool | 8 | 10 | 5 | 7 |
| parallel, different tools | 3 | 4 | 4 | 4 |
| multi-argument | 8 | 6 | 10 | 9 |
| disambiguation | 7 | 8 | 8 | 7 |
| **missing argument** | **9** | 6 | 3 | 10 |
| unavailable tool | 10 | 9 | 10 | 10 |
| **no-tool** | **10** | 7 | 8 | 8 |
| observation grounding | 6 | 3 | **10** | **10** |
| **multi-step** | **6** | 6 | 2 | 3 |
| **Total** | **74** | 68 | 67 | 76 |

What the new training bought over v1: far better **abstention** (missing-argument 6→9, no-tool 7→10, unavailable-tool 9→10) and **observation grounding** (3→6). What it gave back: a little on plain single / parallel-same calls (−2 each). Qwen's edge here is **observation grounding** and **multi-argument** calls; Qwen in thinking mode reaches 76 but at a very different latency class (≈4.7 s per call versus ≈0.8 s for non-thinking models in our independent benchmark — see the [Nesso2 card](https://huggingface.co/mii-llm/nesso2-0.4B-agentic)).

### 2. Held-out Italian benchmark — healthcare booking (CUP) 🇮🇹

40 Italian cases in a domain **no model was trained on**: ten Italian healthcare-booking tools (`cerca_prestazione`, `verifica_disponibilita`, `prenota_appuntamento`, `verifica_esenzione`, `calcola_ticket`, `consulta_ricetta`, `recupera_referto`, `invia_promemoria`, `annulla_prenotazione`, `elenco_strutture`), 10 sub-skills × 4 cases.

| Sub-skill (%) | **Nesso3 (this model)** | Nesso2 v1 | Qwen3-0.6B |
|---|---|---|---|
| single call | 100 | 100 | 75 |
| multi-argument | 85 | 74 | 80 |
| parallel, same tool | 85 | 70 | 35 |
| parallel, different tools | **85** | 55 | 43 |
| disambiguation | 86 | 93 | 55 |
| missing argument | **50** | 35 | 42 |
| unavailable tool | 75 | 85 | 100 |
| no-tool | 68 | 25 | 100 |
| observation grounding | 22 | 65 | 69 |
| multi-step | 30 | 52 | 40 |
| **Overall** | **68.5** | 65.3 | 63.9 |

On an unseen Italian domain the model is the best at the **productive** skills (single, multi-argument, parallel calls including *different* tools: 85% here, though still only 3/10 on the 100-case set — improved, not solved) and at asking for missing arguments. It is clearly **weaker than both baselines at observation grounding and multi-step completion** in this benchmark (22% and 30%) — read the Limitations before building multi-step flows.

### 3. Production robot benchmark (bilingual, 84 cases)

A simulated warehouse-robot deployment with tool schemas the model has not seen (domain close to part of the training data): 60 function-calling cases (30 IT / 30 EN, 10 sub-skills including safety/emergency-stop) + 24 conversation cases (12 IT / 12 EN).

| Model | Overall | Function calling (60) | Conversation (24) |
|---|---|---|---|
| Qwen3-0.6B | **71.1** | **72.5** | 67.6 |
| **Nesso3-0.4B-Agentic** | 63.6 | 59.2 (IT 53.7 · EN 64.6) | **74.9** (IT 79.2 · EN 70.6) |
| Nesso2-0.4B-Agentic (v1) | 53.3 | 43.8 | 76.9 |

Nesso3 recovers +10 points overall over v1 on novel tools (the "does it generalise beyond its training toolbox" question this release targets) and beats Qwen on conversation, but **Qwen3-0.6B remains clearly ahead on robot function calling**, especially in English.

### 4. Generic English tools stress test (33 scenarios)

A small, deliberately plain suite: seven everyday English-named tools (`get_weather`, `calculate`, `convert_units`, `web_search`, `send_email`, `get_stock_price`, `create_event`), 18 EN + 15 IT scenarios, plus two-step chains.

| Model | single call | argument precision | tool selection | no-tool | multi-step | **Overall** |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | 77% | 100% | 71% | 100% | 66% | **84%** |
| Nesso-0.4B-Agentic (older sibling) | 88% | 62% | 57% | 16% | 0% | 54% |
| **Nesso3-0.4B-Agentic** | 77% | 25% | 14% | 100% | 0% | 48% |

This is the benchmark where we **lose clearly**. With only seven plain tools the model often invents a tool name (`multiply`, `sqrt`) or answers from memory instead of calling `calculate` / `web_search`, and it did not complete the two-step chains. Qwen3-0.6B is natively strong on generic tool sets like this one. If your use case is "a handful of generic English tools", prefer Qwen3-0.6B.

### 5. Academic benchmarks

Run with our [fork of lm-evaluation-harness](https://github.com/mii-llm/lm-evaluation-harness/): MMLU is 5-shot `acc`; HellaSwag / ARC are 0-shot `acc_norm`; IFEval is `inst_level_loose_acc` (generative, chat template). Scale 0–1. **All four models were run with the same script** (Qwen3-0.6B with its default chat template).

#### Italian

| Model | IFEval IT ↑ | ARC IT ↑ | HellaSwag IT ↑ | MMLU IT ↑ | **Avg IT** |
|---|---|---|---|---|---|
| Qwen/Qwen3-0.6B | **0.3765** | 0.2678 | 0.3607 | **0.4039** | **0.3522** |
| **Nesso3-0.4B-Agentic** | 0.3046 | **0.3028** | **0.4086** | 0.3264 | 0.3356 |
| Nesso2-0.4B-Agentic (v1) | 0.2962 | 0.3037 | 0.4087 | 0.3258 | 0.3336 |
| mii-llm/nesso-0.4B-agentic | 0.3118 | 0.3011 | 0.4072 | 0.2822 | 0.3256 |

#### English

| Model | IFEval EN ↑ | ARC EN ↑ | HellaSwag EN ↑ | MMLU EN ↑ | **Avg EN** |
|---|---|---|---|---|---|
| Qwen/Qwen3-0.6B | **0.4640** | **0.3456** | **0.4729** | **0.4728** | **0.4388** |
| **Nesso3-0.4B-Agentic** | 0.3897 | 0.3055 | 0.4725 | 0.3343 | 0.3755 |
| Nesso2-0.4B-Agentic (v1) | 0.3789 | 0.3038 | 0.4726 | 0.3356 | 0.3727 |
| mii-llm/nesso-0.4B-agentic | 0.4125 | 0.3038 | 0.4686 | 0.2939 | 0.3697 |

#### Overall

| Model | Avg IT | Avg EN | **Overall** |
|---|---|---|---|
| Qwen/Qwen3-0.6B | 0.3522 | 0.4388 | **0.3955** |
| **Nesso3-0.4B-Agentic** | 0.3356 | 0.3755 | 0.3555 |
| Nesso2-0.4B-Agentic (v1) | 0.3336 | 0.3727 | 0.3531 |
| mii-llm/nesso-0.4B-agentic | 0.3256 | 0.3697 | 0.3476 |

MMLU 0-shot, for reference: Nesso3-0.4B-Agentic IT 0.3341 / EN 0.3470; Qwen3-0.6B IT 0.3726 / EN 0.4026.

The agentic SFT did **not** cost academic ability: Nesso3 is level with v1 on every task (all differences are within noise). Against Qwen3-0.6B the picture is mixed: we are **ahead on Italian HellaSwag (+4.8) and Italian ARC (+3.5)** and level on English HellaSwag, but **Qwen is ahead overall (0.396 vs 0.356)**: clearly on MMLU (+7.5 IT, +13.9 EN 5-shot), on IFEval in both languages (+7.2 IT, +7.4 EN), and on English ARC. These gaps are pre-training-knowledge and instruction-following gaps, not something the agentic post-training causes or fixes.

> ℹ️ Earlier versions of the Nesso2-0.4B-Agentic card listed different Qwen3-0.6B figures (notably IFEval and Italian ARC) and an English MMLU of 0.2700 for Nesso2 v1. Re-running every model with the identical script gives the numbers above (Nesso2 v1 English MMLU: 0.3356).

### 6. Conversational quality (LLM-as-judge)

20 bilingual multi-turn tasks per language, graded 1–10 by `Qwen3.6-35B-A3B` on correctness / language-fidelity / helpfulness (greedy answers, repetition-penalised decoding for all models). The same answer sets were judged in several separate runs; the table shows the **mean of the three runs whose raw files are kept in the repository**. The judge is noisy: most models moved by ≤0.1 between runs, but Nesso3 moved by 0.35 on identical answers (Both: 4.15 → 3.80; a fourth run, whose file was not kept, gave 3.73), so differences below ~0.4 should not be over-read.

| Model | Italian ↑ | English ↑ | **Both** | correctness | language | helpfulness |
|---|---|---|---|---|---|---|
| mii-llm/nesso-0.4B-agentic | 4.20 | **6.35** | **5.28** | 4.83 | 9.38 | 5.38 |
| Nesso2-0.4B-Agentic (v1) | **4.35** | 4.00 | 4.18 | 3.83 | 8.62 | 4.35 |
| Qwen/Qwen3-0.6B | 2.73 | 5.78 | 4.26 | 4.11 | 6.93 | 4.23 |
| **Nesso3-0.4B-Agentic** | 3.65 | 4.23 | 3.94 | 3.71 | 8.55 | 3.98 |

Nesso3 is **somewhat below v1 in free-form Italian chat** (3.65 vs 4.35; the gap is present in every run: 3.85 / 3.60 / 3.50 vs 4.50 / 4.50 / 4.05) and level in English (4.23 vs 4.00); overall it is a little under v1 and Qwen3-0.6B (3.94 vs 4.18 and 4.26). It is still clearly ahead of Qwen3-0.6B in Italian (3.65 vs 2.73) and behind it in English (4.23 vs 5.78). Training the model on one strict tool contract with a large amount of function-calling data came at some cost in free-form Italian conversation. If your application is **chat-first**, Nesso2-0.4B-Agentic (v1) or Nesso-0.4B-agentic is the better choice; Nesso3 is the better choice when **tool use** is the job.

For scale: the 3B model we used as a distillation teacher scores 6.37 on the same judge, so there is real headroom in chat that a 0.4B SFT did not capture (our on-policy-distillation attempts to close it are documented, with their costs, in the GitHub report).

---

## Limitations

We would rather you hear these from us:

- **"Ask-reflex" on complete requests.** On short, fully-specified requests the model sometimes **asks for an argument that is already in the prompt** instead of calling the tool — e.g. *"Dimmi il meteo a Milano."* → "In quale città?". In a 14-prompt smoke test with three plain tools (7 Italian + 7 English, all arguments present) it produced the correct call for **8 of 14**; longer or more natural phrasings ("Che tempo c'è a Torino adesso?") are more reliable. This is the flip side of its good abstention behaviour and is shared by every checkpoint of this family. It is stronger when the system prompt offers a **single** tool: with only `get_weather` and *"Che tempo c'è a Roma adesso?"* the model answered with a complaint about missing arguments, while the same sentence with three tools in the prompt produced the call. Use the application-layer guard in *Production notes*.
- **Observation grounding and multi-step are uneven.** 6/10 on the 100-case set, but 22% (observation) and 30% (multi-step) on the held-out Italian healthcare benchmark, and 0% on two-step chains in the generic-tools stress test. The model may re-call a tool instead of answering from the returned data (notably when several parallel results come back as separate `tool` turns, see *Production notes*), or stop after the first step. Keep multi-step flows short and supervised.
- **Generic / English-first tool sets.** Tool-name grounding is weaker with few, plainly-named English tools (it can invent a plausible name). Qwen3-0.6B is stronger there (see §4) and on English robot function calling (§3).
- **Format-lock.** It follows *its* contract. Different prompt layouts degrade it.
- **Different-tool parallel calls and exact multi-step completion remain capacity-bound** at ~0.4B (3–4/10 on the 100-case set).
- **Evaluation caveat.** The agentic suites are small (40–100 cases), in-house, and partly synthetic; differences of 1–2 cases are noise. Academic benchmarks are standard but small differences (≈±0.01) are within noise.
- **Free-form Italian conversation is weaker than v1** (judge Italian 3.65 vs 4.35; overall 3.94 vs 4.18, noisy) — see §6.
- **Knowledge and instruction following.** MMLU (0.33 vs 0.40 IT, 0.33 vs 0.47 EN) and IFEval (0.30 vs 0.38 IT, 0.39 vs 0.46 EN) stay clearly below Qwen3-0.6B: pre-training-knowledge and instruction-following gaps that this post-training does not close.

Nothing here is a safety layer: validate and authorise every tool call in your application.

---

## Related Models

| Model | Description |
|---|---|
| [Zagreus-0.4B-ita](https://huggingface.co/mii-llm/zagreus-0.4B-ita) | Base pre-trained model (this model's foundation) |
| [Nesso2-0.4B-Agentic](https://huggingface.co/mii-llm/nesso2-0.4B-agentic) | Previous release: verbose-contract agentic SFT (memorises a fixed set of common tools); better free-form Italian conversation and stronger on the observation/multi-step sub-skills of the held-out healthcare benchmark |
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
- **Salesforce** (APIGen / xLAM), **Team-ACE** (ToolACE) and **NVIDIA** (When2Call) for the public function-calling datasets
- The **mii-llm** open-source community
