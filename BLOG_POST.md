# Seven Small Models, One Large Ambition

## How Zagreus and Nesso bring open, bilingual intelligence to the edge

*Four Romance languages. Seven open models. Sixty-four NVIDIA A100 GPUs. Roughly one trillion training tokens. And a deceptively simple question: how much useful intelligence can fit into 0.4 billion parameters?*

> Prefer the story in chapters? Read [The Zagreus–Nesso Chronicles](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/blog-series/README.md), a seven-episode version covering the complete journey.

![Italian and English model leaderboard](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/images/01_leaderboard.png?raw=true)

The current AI race is usually narrated in billions: more parameters, more GPUs, more data, more capital. But the future of AI will not live only inside hyperscale data centers. It will also run in the smaller, quieter places—on private infrastructure, inside products, close to users, and eventually on everyday devices.

That is the territory explored by **Zagreus** and **Nesso**, two families of compact language models created by the [mii-llm](https://mii-llm.ai) community with infrastructure provided by [Seeweb](https://www.seeweb.it). The project began with an ambitious constraint: build a genuinely capable Small Language Model from scratch, specialize it for European languages, document the difficult parts, and release the result openly.

The outcome is a family of seven 0.4B-parameter models: four bilingual base models and three post-trained variants. Together, they offer a practical experiment in sovereign AI—models whose data lineage, training process, behavior, and deployment can be understood and controlled.

> **The goal was never to shrink a frontier model until it fit. It was to discover what a carefully designed small model could become on its own terms.**

---

## Meet the family

The Greek-inspired names mark two distinct stages in the models’ lives. **Zagreus** is the foundation: pretrained models that learn language from raw text. **Nesso** is behavior shaped for use: instruction following, conversation, and agentic execution.

| Model | Languages | Type | Designed for |
|---|---|---|---|
| [Zagreus-0.4B-ita](https://huggingface.co/mii-llm/zagreus-0.4B-ita) | English + Italian | Base | Italian language understanding, completion, and further fine-tuning |
| [Zagreus-0.4B-spa](https://huggingface.co/mii-llm/zagreus-0.4B-spa) | English + Spanish | Base | Spanish applications and custom post-training |
| [Zagreus-0.4B-por](https://huggingface.co/mii-llm/zagreus-0.4B-por) | English + Portuguese | Base | Portuguese language and regional evaluation |
| [Zagreus-0.4B-fra](https://huggingface.co/mii-llm/zagreus-0.4B-fra) | English + French | Base | French applications and downstream specialization |
| [Nesso-0.4B-instruct](https://huggingface.co/mii-llm/nesso-0.4B-instruct) | English + Italian | SFT | Conversation and general instruction following |
| [Nesso-0.4B-agentic](https://huggingface.co/mii-llm/nesso-0.4B-agentic) | English + Italian | SFT | Function calling, structured output, and agentic workflows |
| [Open-Zagreus-0.4B](https://huggingface.co/mii-llm/open-zagreus-0.4B) | English + Italian | Open SFT | Fully reproducible research from public data to weights |

The models are small enough to invite a different kind of thinking. Instead of asking whether one checkpoint can do everything, we can route work to a focused model: Italian conversation to Nesso, structured actions to Nesso Agentic, Portuguese language tasks to Zagreus Portuguese, and so on. Specialization becomes a feature rather than an apology.

---

## Zagreus: four foundations, four linguistic homes

Every Zagreus model is bilingual, pairing English with one Romance language. This is important. English provides access to an enormous body of general and technical knowledge, while dedicated Italian, Spanish, Portuguese, or French data gives each model a linguistic center of gravity.

### Zagreus Italian

The Italian checkpoint is the heart of the project and the foundation for the Nesso family. It was trained to understand both English and Italian, then evaluated throughout training rather than only at the final checkpoint. Its Italian benchmark average moved from **0.2849 at 95k steps** to a peak around **0.2981 at 365k** in the reported MMLU, HellaSwag, and ARC evaluation set. On the broader EVALITA suite, the base model reached an overall score of **0.3226**.

The curve is instructive: more training did not produce a perfectly monotonic benchmark climb. That is one of the honest lessons of building from scratch. Loss can improve while individual capabilities oscillate, making checkpoint selection an empirical decision rather than a ceremonial choice of “latest.”

![Zagreus Italian checkpoint progression](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/images/zagreus-ita.png?raw=true)

### Zagreus Spanish

Zagreus Spanish improved steadily across its reported multilingual evaluation. The combined MMLU, ARC, and HellaSwag average rose from **0.309 at 146k steps** to **0.321 at 518k**. It is a compact base for Spanish-language experimentation, especially where local control and inexpensive inference matter more than encyclopedic breadth.

![Zagreus Spanish checkpoint progression](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/images/zagreus-spa.png?raw=true)

### Zagreus French

The French model followed a similar trajectory, reaching a reported three-task average of **0.321 at 705k steps**. At only 0.4B parameters, it is not intended as a universal oracle. It is a trainable, inspectable foundation for teams that want to build French-first systems without beginning from a generic English-centric checkpoint.

![Zagreus French checkpoint progression](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/images/zagreus-fra.png?raw=true)

### Zagreus Portuguese

Zagreus Portuguese reached **0.3113** across ARC, HellaSwag, and MMLU at the 582k checkpoint. An additional Portuguese benchmark tells an even more interesting story: the 483k Zagreus checkpoint scored **0.3230** across nine tasks in `lm-evaluation-harness-pt`, ahead of the reported **0.2569** for Qwen3-0.6B-Base in that comparison.

That result captures the central bet behind the family. A smaller model with deliberate language coverage can outperform a larger general-purpose baseline inside the domain it was built to serve.

![Portuguese benchmark comparison](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/images/por-garcia.png?raw=true)

---

## Nesso: turning language into useful behavior

Pretraining teaches a model to continue text. Post-training teaches it how to help.

The Nesso models begin with the English–Italian foundation and use supervised fine-tuning to develop distinct behaviors. This stage consumes far less compute than pretraining, but the work becomes more editorial: dataset quality, formatting, response boundaries, and examples of desired behavior matter enormously.

### Nesso Instruct: the conversational model

**Nesso-0.4B-instruct** is tuned for dialogue and instruction following. In the standard English and Italian evaluation reported in the project, it achieved a combined average of **0.3345**, ahead of LiquidAI/LFM2-350M’s **0.2576** and within reach of Qwen3-0.6B’s **0.3545**.

Its strongest moments appear in practical language work: formal email writing, concise suggestions, creative prose, and straightforward instruction following. The custom 20-task benchmark also exposed its principal weakness—runaway repetition on some longer generations. That is not a footnote to hide; it is actionable deployment knowledge. Repetition penalties, correct end-of-sequence handling, generation limits, and further preference tuning are part of turning a promising checkpoint into a reliable product.

### Nesso Agentic: the action-oriented model

**Nesso-0.4B-agentic** is optimized for structured outputs, function calling, and tool-oriented tasks. On the project’s hands-on Italian benchmark it was the standout model, scoring **73/100**—ahead of Nesso Instruct, Qwen3-0.6B, Qwen3.5-0.8B, and Open Zagreus.

It led or shared the lead in vegetarian follow-up, code generation, translation, pros-and-cons analysis, calorie estimation, simple explanations, and Italian trivia. It also produced the benchmark’s best factorial implementation. In English it remained competitive at **66/100**, only one point behind Qwen3-0.6B in that test.

Those results make Agentic the clearest expression of the project’s edge thesis. A 0.4B model does not need to know everything if it can recognize intent, produce valid structure, and hand the right work to a calculator, database, search system, or application API.

### Open Zagreus: reproducibility all the way down

**Open-Zagreus-0.4B** answers a different question: what can be built when the post-training data is public too?

It was trained with [OpenItalianData](https://huggingface.co/datasets/DeepMount00/OpenItalianData), released by Michele Montebovi and the Italian open-source community. Its EVALITA overall score improved from the Italian base model’s **0.3226** to **0.3313**, with gains in summarization, textual entailment, sentiment, and word-in-context tasks.

Some individual tasks regressed, illustrating that fine-tuning is not a free universal upgrade. But Open Zagreus offers something more durable than a single leaderboard number: a reproducible path from public data to usable weights.

![Base and open-model EVALITA comparison](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/images/evalita_comparison.png?raw=true)

---

## What it takes to train 0.4B parameters from scratch

“Small” is relative. A 0.4B model may be compact at inference time, but creating it still requires serious systems engineering.

The team trained on **64 NVIDIA A100 GPUs** across eight nodes provided by Seeweb. The architecture is a modified dense Llama-style transformer with:

- 32 hidden layers
- a hidden size of 960
- 15 attention heads and 5 key-value heads
- a 4,096-token context window
- a 128,256-token vocabulary
- tied input and output embeddings
- bfloat16 training

The dense design was intentional. At this scale, mixture-of-experts routing can introduce overhead and leave experts underused; a dense network keeps every parameter working on every token.

The data pipeline combined English web text, language-specific FineWeb 2 and FinePDFs data, and code from StarCoderData. The project operated at roughly **one trillion tokens**, with the Italian run described as approximately **400B English, 400B Italian, and 200B code tokens**.

Before a GPU could learn from any of that material, it had to be tokenized. Using the Llama 3.2 tokenizer and Hugging Face’s [datatrove](https://github.com/huggingface/datatrove), tokenization alone ran continuously for more than three weeks and required several terabytes of storage. It is a useful corrective to the glamour of model training: much of the real work happens before the first loss curve appears.

For distributed pretraining, the team selected [Hugging Face Nanotron](https://github.com/huggingface/nanotron), using data parallelism across the 64 GPUs and Slurm to coordinate the cluster. Their changes are available in an [mii-llm Nanotron fork](https://github.com/mii-llm/nanotron), alongside an upstream contribution. Post-training used [Axolotl](https://github.com/axolotl-ai-cloud/axolotl), again distributed through Slurm.

At one recorded point, the run was processing roughly **183,000 tokens per second** globally, had consumed **480 billion tokens**, and reported a language-model loss around **2.06**. The estimated time remaining was still more than 130 days. Training from scratch contains joy—but the pain in the title is earned.

---

## The benchmarks are not a victory lap

The most valuable part of the evaluation is its candor. The project compares its models not only on established suites such as MMLU, ARC, HellaSwag, IFEval, EVALITA, and Portuguese community benchmarks, but also on a practical bilingual set of 20 tasks: math, code, translation, email, grammar, classification, cultural knowledge, and multi-turn advice.

![Italian versus English model performance](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/images/02_it_vs_en.png?raw=true)

The custom benchmark puts Nesso beside two strong external references, **Qwen3-0.6B** and **Qwen3.5-0.8B**. The result is not one universal winner:

- **Nesso Agentic led Italian** with 73/100 and remained strong in English at 66/100.
- **Qwen3.5-0.8B led English** with 71/100, especially on rich generative tasks.
- **Qwen3-0.6B was the strongest reasoner** on the tested math, logic, and classification prompts, aided by its thinking mode.
- **Nesso Instruct** showed useful conversational ability but suffered from repetition loops.
- **Open Zagreus** was extremely fast on simple Italian tasks, but unreliable on broad instruction following and unsuitable for the tested English route.

The failures were sometimes wonderfully specific. Every model struggled to produce a truly traditional carbonara. Every model stumbled over canonical Italian literature. Italian grammatical agreement exposed a shared blind spot. Several models delivered confident hallucinations about Rome or the Sistine Chapel. More output tokens did not always help; sometimes they merely gave a model more room to be wrong—or to repeat itself.

![Tasks that challenged every evaluated model](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/images/08_universal_failures.png?raw=true)

This is precisely why small-model benchmarking must go beyond aggregate accuracy. A score cannot tell an application developer that a model loops after a good answer, switches languages unexpectedly, invents cultural facts, or spends its token budget reasoning without completing the visible response.

---

## What the seven models teach us

### 1. Small models reward focus

Nesso Agentic does not win every category. It wins enough of the *right* categories to be compelling for an Italian edge assistant. Zagreus Portuguese shows the same effect at the foundation level: targeted language data can matter more than a modest parameter advantage.

### 2. Post-training changes the product, not just the score

The same foundation can become a conversational model, an agentic model, or a fully reproducible open model. Dataset design determines how capability is expressed. It can also introduce new weaknesses, including repetition and task-specific regressions.

### 3. Routing beats pretending

The benchmark suggests a practical architecture: detect the user’s language, identify the task, and choose the right model or tool. Use Nesso Agentic for Italian actions and structured workflows. Use external tools for arithmetic and factual retrieval. Reserve longer reasoning modes for problems that require them. Refuse to make one tiny model impersonate an entire AI stack.

### 4. Openness is infrastructure

Open weights are valuable. Open training code, evaluation commands, dataset references, checkpoint history, and disclosed failure modes are more valuable still. They let other teams reproduce results, diagnose behavior, and build models that reflect their own languages and constraints.

### 5. Edge intelligence is a systems problem

The model is only one component. Tokenization, storage, distributed training, checkpoint conversion, prompt templates, stopping criteria, retrieval, tool use, and routing all determine whether the final experience feels intelligent.

---

## A small beginning with room to grow

Zagreus and Nesso do not argue that 0.4B parameters are enough for every problem. Their more interesting claim is that **0.4B parameters can be enough for many useful problems—when language, data, behavior, and deployment are designed together**.

There is more work ahead. The instruction models are currently at the supervised fine-tuning stage, with preference optimization identified as a next step. Repetition needs tighter control. Cultural knowledge and Italian grammar need targeted data. Agentic models should lean more heavily on tools for arithmetic and grounded facts. English evaluation should be repeated with a clean, independently generated prompt run.

But the foundation is now public: four bilingual pretrained models, three distinct post-trained models, a reproducible training stack, and evaluation results that show both promise and limits.

In a field obsessed with making models larger, Zagreus and Nesso offer another direction: make them **closer, more focused, more transparent, and easier to own**.

That may be exactly the kind of intelligence the edge needs.

---

### Explore the project

- Browse all releases from [mii-llm on Hugging Face](https://huggingface.co/mii-llm)
- Read the [full technical report](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/README.md)
- Explore the [Nanotron training fork](https://github.com/mii-llm/nanotron)
- Join the [mii-llm community](https://mii-llm.ai)

*Built by the mii-llm community with compute infrastructure sponsored by Seeweb.*
