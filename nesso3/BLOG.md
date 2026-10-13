# Nesso3: a 0.4B model that calls tools in Italian, and everything that failed on the way

## We are releasing Nesso3-0.4B-Agentic, with the recipe, the raw results and the experiments that did not work

Small models are easy to make look good on a demo. The hard part is a model that still works when the tools it is given are **new**, in **Italian**, and it has to decide whether to call a tool at all.

Today we are releasing **[Nesso3-0.4B-Agentic](https://huggingface.co/mii-llm/nesso3-0.4B-agentic)**, a ~0.4B bilingual (Italian/English) model for function calling and agentic execution. This post covers how to use it, what it does well, where it loses to Qwen3-0.6B, the recipe, and the ten experiments, successes and failures, that led here. Every number links back to raw result files in the [repository](https://github.com/mii-llm/zagreus-nesso-slm/tree/main/nesso3).

---

## TL;DR

- **Best of our models for Italian tool use.** 74/100 on our bilingual function-calling suite (Nesso2: 68, Qwen3-0.6B: 67), and 38/50 on the Italian half (Qwen3-0.6B: 29; 34 with thinking on).
- **It generalises to tools it has not seen.** On a robot benchmark with unseen tool schemas it goes from 53.3% (Nesso2) to 63.6%, and on a held-out Italian healthcare-booking benchmark it scores 68.5% vs 63.9% for Qwen3-0.6B.
- **It is not a Qwen replacement for everything.** Qwen3-0.6B is clearly better on generic English tools (84% vs 48%), robot function calling, MMLU, IFEval and English chat. We publish those numbers too.
- **One contract, thousands of tools.** The model learned a single compact prompt format over 2,531 procedurally generated tool schemas, plus public data (xLAM, ToolACE, When2Call).
- **SFT only.** We also tried reinforcement learning and on-policy distillation. Neither made the release, and we explain why below.

![Nesso3 vs Nesso2 vs Qwen3-0.6B](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/nesso3/images/nesso3_vs_baselines.png?raw=true)

---

## Try it in a minute

The model uses a Llama-3 chat template and its own tool contract. Three rules matter:

1. **Put the tools inside the system message**, as compact JSON between `<tools>` tags. Do **not** pass `tools=` to `apply_chat_template`: this model's template ignores it and your tools would silently disappear.
2. **Decode greedily** (`do_sample=False`, no repetition penalty).
3. **Tokenize with `add_special_tokens=True`** so exactly one BOS is added.

```python
import re, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "mii-llm/nesso3-0.4B-agentic"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto").eval()

CONTEXT = "Sei un assistente operativo. Usa gli strumenti disponibili quando serve."

def system_with_tools(context, tools):
    return context + "\n<tools>\n" + json.dumps(tools, separators=(",", ":"), ensure_ascii=False) + "\n</tools>"

def chat(messages, max_new_tokens=256):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(model.device)
    n = inputs["input_ids"].shape[1]
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                         eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
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
```

Here is what the model does with that system prompt (real outputs from the released weights):

| User says | Nesso3 does |
|---|---|
| *Quanto sono 250 euro in sterline?* | `convert_currency(amount=250, from="EUR", to="GBP")`. It picks the right tool among three and fills the arguments. |
| *Che tempo fa a Roma e a Torino?* | Two parallel `get_weather` calls, one per city. |
| *Qual è la capitale della Francia?* | "La capitale della Francia è Parigi." It answers directly instead of reaching for a tool. |
| *Prenota un volo per Parigi.* | "Non posso prenotare un volo: nessuno degli strumenti disponibili lo consente." It refuses instead of inventing a tool. |
| *Convertimi 100 euro.* | "Per procedere ho bisogno di un dato. In quale valuta desideri convertire?" It asks for the missing argument. |

And closing the loop, where you run the tool and hand the result back:

```python
question = "Che tempo c'è a Roma adesso?"
ans = chat([system, {"role": "user", "content": question}])
print(parse_calls(ans))
# [{'name': 'get_weather', 'arguments': {'city': 'Roma'}}]

observation = {"city": "Roma", "temp_c": 24, "condition": "soleggiato"}   # the result of YOUR get_weather()
print(chat([system, {"role": "user", "content": question}, {"role": "assistant", "content": ans},
            {"role": "tool", "content": json.dumps(observation, ensure_ascii=False)}]))
# Ci sono 24°C e la condizione è soleggiata.
```

Two behaviours to know before you integrate: on short, fully specified prompts the model sometimes **asks for an argument that is already there** (we measured the right call on 8 of 14 such prompts, and it is worse when only one tool is offered), and after parallel calls you should return **all results in a single `tool` turn** (a JSON list). Always validate arguments in your application before executing anything.

---

## The idea: memorise one contract, generalise across many tools

The previous model, Nesso2, learned a fixed toolbox in a verbose format. It was good on the tools it had seen and fell apart on new ones: on our robot benchmark, with tool schemas it had never encountered, observation grounding was 33% and "answer without a tool" 12%.

At 0.4B parameters a model cannot learn "tool use" in the abstract from a handful of tools; it locks onto whatever format it saw. So we made a two-part bet:

- **Memorise one contract.** Tools always appear as `{context}\n<tools>\n[compact JSON]\n</tools>` in the system message; calls are `<tool_call>{"name": ..., "arguments": ...}</tool_call>`; results come back as `tool` turns with raw JSON.
- **Generalise across thousands of tools.** The synthetic training corpus uses **2,531 procedurally generated tool schemas**, half Italian and half English, covering single and parallel calls, multi-argument calls, disambiguation, missing arguments, unavailable tools, answering without tools, observation grounding and multi-step trajectories.

The first experiment confirmed it: on unseen robot tools observation grounding went from 33% to 79% and no-tool answering from 12% to 52%. The price: the model is **format-locked**. Give it a different prompt layout and it degrades. That is why the model card is strict about the contract.

---

## Results

### The 100-case function-calling suite

100 cases, 50 Italian and 50 English, ten skills of ten cases each, scored automatically, pure greedy decoding, each model on its own native prompt format.

![Per-category results on the 100-case suite](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/nesso3/images/categories_100case.png?raw=true)

| | Nesso3 | Nesso2 | Qwen3-0.6B | Qwen3-0.6B thinking |
|---|---|---|---|---|
| **Total** | **74** | 68 | 67 | 76 |
| Italian half (/50) | **38** | 35 | 29 | 34 |
| English half (/50) | 36 | 33 | 38 | 42 |

What the new training bought over Nesso2: much better **abstention**. Missing-argument goes 6 → 9, no-tool 7 → 10, unavailable-tool 9 → 10, and observation grounding 3 → 6. What it gave back: a little on plain single and same-tool-parallel calls. Qwen's strengths are observation grounding and multi-argument calls; its thinking mode reaches 76 but at roughly six times the latency (about 4.7 s against 0.8 s per call in our independent benchmark).

### Tools it was never trained on

| Benchmark | Nesso3 | Nesso2 | Qwen3-0.6B |
|---|---|---|---|
| Held-out Italian healthcare booking (40 cases, new domain) | **68.5%** | 65.3% | 63.9% |
| Production robot benchmark (84 cases, unseen schemas) | 63.6% | 53.3% | **71.1%** |
| Generic English tools (33 scenarios) | 48% | – | **84%** |

On the healthcare benchmark the model is strongest where it matters for products: single calls (100%), multi-argument and parallel calls (85%), including parallel calls to *different* tools. Its weak spots there are observation grounding (22%) and multi-step (30%). On the robot benchmark it is better than Qwen on conversation and worse on function calling, especially in English.

The generic-tools test is our clearest loss. With seven everyday English tools (`get_weather`, `calculate`, `web_search`...), Nesso3 still invents tool names like `multiply` and does not complete two-step chains. If that is your use case, use Qwen3-0.6B.

### General ability and chat

| | Nesso3 | Nesso2 | Qwen3-0.6B |
|---|---|---|---|
| Academic average (MMLU, ARC, HellaSwag, IFEval; IT+EN) | 0.3555 | 0.3531 | **0.3955** |
| Chat quality, LLM judge, Italian (/10) | 3.65 | **4.35** | 2.73 |
| Chat quality, LLM judge, English (/10) | 4.23 | 4.00 | **5.78** |

Agentic post-training cost nothing measurable in academics: Nesso3 is level with Nesso2 on every task. Qwen3-0.6B is clearly ahead on MMLU and IFEval, which are pre-training-knowledge and instruction-following gaps that SFT does not close. Where we are ahead of Qwen is Italian HellaSwag and ARC. On chat, Nesso3 is **weaker than Nesso2 in Italian** (3.65 vs 4.35): the price of training on one strict tool format. If your product is chat first, use Nesso2 or nesso-0.4B-agentic instead. (The judge is Qwen3.6-35B-A3B, run three times; it moved the same model's score by up to 0.35 between runs, so treat differences under 0.4 as noise.)

---

## The recipe

Nesso3 is the last step of a chain: [Zagreus-0.4B-ita](https://huggingface.co/mii-llm/zagreus-0.4B-ita) (pre-trained from scratch, ~1T tokens) → knowledge continued pre-training (~4.5B tokens of knowledge-dense text: Wikipedia-derived grounded QA and exam-style QA) → 32k context extension → **agentic SFT**. The first three stages are written up in the [Nesso2 report](https://github.com/mii-llm/zagreus-nesso-slm/tree/main/nesso2); this post covers the last one.

| Data component | Size | What it teaches |
|---|---|---|
| Chat floor (omnia v6: all Italian, English with score ≥ 0.6) | ~2.27M rows, ×1 | stay a usable conversationalist |
| Synthetic function-calling corpus | 36k rows, ×4 | the skills above over 2,531 tool schemas |
| Public function-calling data | ~33k rows, ×1 | xLAM-60k, ToolACE and When2Call, rendered into the compact contract, about a third translated to Italian |

Training: LR 1e-3, 3 epochs (5,094 steps), effective batch 128, max sequence length 8,192, cosine schedule down to a 0.3 floor, bf16. The script is [in the repository](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/nesso3/training/sbatch-sft-compact-v5b.sh). The synthetic corpus and its generators are not released; the public datasets are credited at the end.

The public data deserves a note. We audited every candidate before using it: none of them contains Italian (we translated a third with Qwen3.6-35B-A3B and kept only items passing automatic checks), several overlap heavily, some carry long reasoning traces that break a non-thinking contract, and When2Call's SFT split contains *no tool-call examples at all*, which makes a model over-refuse unless it is paired with call data. Everything was decontaminated against our evaluation sets with an 8-gram check.

---

## The road to Nesso3: ten experiments, four of them clear failures

At 0.4B the data budget is zero-sum: every skill we improved took something from another. Here is the full list, graded on the same 100-case suite.

![Every experiment on the 100-case suite](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/nesso3/images/journey_100case.png?raw=true)

**1. One compact contract over 2,531 tools: it works.** Robot observation grounding 33% → 79%. But the 100-case suite dropped (61 vs Nesso2's 68, whose memorised tools favour that toy suite) and parallel calls to different tools sat at zero.

**2. Procedural abstention data: it over-corrected.** We had found that the "missing argument" and "unavailable tool" generators produced only five canned scenarios each, so the model memorised phrases. We rewrote them to be diverse. The targets moved (healthcare unavailable-tool 38% → 100%), but answering directly without a tool collapsed (92% → 52%): diverse examples *teach* where canned ones are memorised, so the same share of refusals pulled much harder, and the model refused questions it could answer.

**3. Fewer refusals plus "answer me" data: it worked.** We lowered the refusal share below the original and added 64 kinds of questions to answer directly even when tools are present. Abstention gains held and direct answering recovered. Our best model of that branch (64).

**4. Patching an external benchmark's bugs: it backfired.** A third-party Italian benchmark flagged real bugs (inventing tool names, a search-tool confusion, English-only enum values in Italian rows). We patched them with targeted data; they were fixed, but the 100-case suite fell from 64 to 56 and "unavailable tool" on the robot benchmark from 52% to 12%: "always commit to a real tool" generalised into inventing calls. Along the way we found that the earlier model's perfect score on one negative-control case was an accident of a JSON parser rejecting malformed output. We stopped patching calibration with SFT data.

**5–6. Chat-anchored reinforcement learning: a win with a bill.** A GRPO reward that pays for grounding tool results *and* for staying close to the model's own chat answers lifted the 100-case score from 64 to 68, and the judge improved on every axis. Adding more rewards pushed it to 70 and cracked parallel-different-tool calls for the first time (0 → 2), but each fix undid part of another. We kept the method and did not combine it into this release.

**7. Public function-calling data: this is Nesso3.** Two arms with the same recipe: with and without ToolMind, a large dataset of long multi-step chains. **With it the model collapsed to 49/100. Without it, 74/100**, and generic English tools went from 36% to 48%. The long, noisy chains destroyed the basic skills.

**8. Adding Nemotron-SFT-Agentic-v2: it hurt.** A filtered 21,300-row slice of NVIDIA's dataset took observation grounding to 10/10 but dropped single-call and multi-argument scores from 7-8 down to 4, the 100-case total to 68, and the healthcare benchmark from 68.5% to 61.6%. It is one run, so we do not know whether the data or the mixing weight is at fault.

**9–10. On-policy distillation: tools up, instruction following down.**

![On-policy distillation trade-off](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/nesso3/images/opd_tradeoff.png?raw=true)

Our 3B model is a much better chatter than any 0.4B model (judge 6.4 vs 3.9), so we distilled it into Nesso3 on-policy, with a frozen copy of Nesso3 anchoring the tool prompts. A gentle run was almost free but also almost useless on chat. A strong run produced the best tool numbers we have measured (100-case 75, healthcare 74.4%) and a real bill: Italian instruction following fell 4.9 points, HellaSwag about 2 points, and the academic average 1.6 points. A model that claims to be the best Italian agentic small model should not lose five points of Italian instruction following to gain one tool case, so neither distilled model was released. The headroom is real, and the next thing we want to try is a longer or differently mixed distillation.

### What we learned

1. **Data is zero-sum at this size.** Patching one skill took from another in experiments 2, 4 and 8.
2. **Diversity teaches, canned data memorises.** The same refusal ratio behaves very differently once the examples vary.
3. **Public data helps if it is short and clean.** xLAM + ToolACE + When2Call gave the biggest gain of the programme; ToolMind's long chains erased it.
4. **No single benchmark picks the winner.** The robot benchmark favoured one model, healthcare another, the 100-case suite a third, and the judge a fourth. We read all of them and read raw outputs.
5. **Scores can be artefacts.** A parser rewarded malformed JSON; a uniform prompt format favoured Qwen's own style; the LLM judge moved identical answers by up to 0.35.
6. **Distillation moves what you point it at and costs what you do not protect.**

We also corrected our own earlier public numbers while writing this: the English MMLU of the Nesso2 card was wrong, and its Qwen3-0.6B academic column mixed evaluation protocols. We re-ran everything with a single script, fixed the cards, and retracted the claim that Nesso2 "ties Qwen on Italian academics". It does not.

---

## Limits, honestly

- **Ask-reflex:** on short complete requests it sometimes asks for an argument already given.
- **Observation grounding and multi-step are uneven:** 6/10 on the 100-case suite, but 22% and 30% on the held-out healthcare benchmark and 0% on two-step generic chains.
- **Generic and English-first tools:** use Qwen3-0.6B.
- **Chat and knowledge:** weaker Italian chat than Nesso2; MMLU and IFEval stay below Qwen3-0.6B.
- **Small benchmarks:** 40 to 100 cases, built by us, one run each. Differences of one or two cases are noise. The robot benchmark's domain is close to part of our synthetic data, so read it as "new tools", not "new domain".
- **Not tested:** chat-anchored RL on top of Nesso3, which sub-slice of the Nemotron data hurt, and multiple seeds for any run.

Nothing here is a safety layer: validate and authorise every tool call in your application.

---

## Everything is in the repository

- 🤗 **Model:** [mii-llm/nesso3-0.4B-agentic](https://huggingface.co/mii-llm/nesso3-0.4B-agentic)
- 📚 **Usage, results and the full experiment log:** [github.com/mii-llm/zagreus-nesso-slm/tree/main/nesso3](https://github.com/mii-llm/zagreus-nesso-slm/tree/main/nesso3)
- 🧪 **Raw result files** for every number in this post, and a script that rebuilds every table: [`results/`](https://github.com/mii-llm/zagreus-nesso-slm/tree/main/nesso3/results) and [`eval/summarize_results.py`](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/nesso3/eval/summarize_results.py)
- 🧬 **Earlier chapters** (the base model, the knowledge continued pre-training, the 32k extension): [the Nesso2 report](https://github.com/mii-llm/zagreus-nesso-slm/tree/main/nesso2)

Nesso3 is released under Apache-2.0. If you build something with it, or find a case where it fails that we have not listed, we would like to hear about it.

*A note on the name: Nesso3 is the third generation of the Nesso family (Nesso → Nesso2 → Nesso3). The 3 is the generation, not the parameter count; the model has about 0.4 billion parameters.*

### Acknowledgements

Built on the [Seeweb](https://www.seeweb.it) HPC infrastructure, thanks to **Antonio Baldassarra** and **Marco Cristofanilli**. Thanks to the Hugging Face team (Nanotron, datatrove, FineWeb, FineWeb-2), to **Salesforce** (APIGen / xLAM), **Team-ACE** (ToolACE) and **NVIDIA** (When2Call, Nemotron-SFT-Agentic-v2) for their public function-calling data, to the Qwen team for the baseline and the judge model, and to the whole **mii-llm** open-source community.

*Made with ❤️ in Italy by [mii-llm](https://mii-llm.ai).*
