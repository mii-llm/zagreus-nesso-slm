# Episode 6 — The Benchmark That Refused to Flatter Us

## Wins, repetition loops, confident hallucinations, and universally bad carbonara

*This is Episode 6 of [The Zagreus–Nesso Chronicles](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/blog-series/README.md). [Start with Episode 1](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/blog-series/episode-01-why-build-from-scratch.md).*

Benchmarks are often arranged like award ceremonies. The table appears, the winning number is bold, and inconvenient details disappear below the fold.

The Zagreus–Nesso evaluation is more interesting because it keeps the inconvenient details.

Alongside established suites such as MMLU, ARC, HellaSwag, IFEval, EVALITA, and Portuguese community benchmarks, the project uses a hand-inspected set of 20 practical tasks in Italian and English. The prompts cover translation, math, code, classification, grammar, summarization, email, cultural knowledge, cooking, creative writing, and follow-up conversation.

Five sub-1B models participated in Italian:

- Nesso-0.4B-agentic
- Nesso-0.4B-instruct
- Open-Zagreus-0.4B
- Qwen3-0.6B
- Qwen3.5-0.8B

Open Zagreus was excluded from the English leaderboard after producing Italian regardless of the tested prompt language.

## Two languages, two leaders

Nesso Agentic led the Italian benchmark with **73/100**. Nesso Instruct scored 56, Qwen3.5 scored 55, Qwen3 scored 52, and Open Zagreus scored 40.

![Italian per-task heatmap](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/images/05_heatmap_it.png?raw=true)

English reversed the picture. Qwen3.5 led with **71/100**, followed by Qwen3 at 67, Nesso Agentic at 66, and Nesso Instruct at 52.

![English per-task heatmap](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/images/06_heatmap_en.png?raw=true)

The split matters. A single multilingual average would hide each model’s real shape. Nesso Agentic is strongest as an Italian-first practical assistant. Qwen3.5 benefits greatly from English on rich generative tasks. Qwen3’s thinking mode adds genuine value for formal reasoning but consumes time and output budget.

There is no universal champion—only a best fit for a particular language, task, and latency envelope.

## The things each model does well

**Nesso Agentic** was the most consistent Italian model. It led or shared the lead in code generation, translation, vegetarian follow-up, calorie estimation, pros and cons, simple explanations, and trivia. Its factorial function was the strongest code response in the Italian run.

**Nesso Instruct** excelled at concise, conventional language tasks. It wrote the best formal Italian email and an excellent quick-dinner response.

**Open Zagreus** was remarkably fast and handled straightforward Italian translation well.

**Qwen3-0.6B** showed the clearest formal reasoning. Its thinking mode solved the tested percentage problem and syllogism, and it classified the six requested words correctly.

**Qwen3.5-0.8B** produced the richest English creative writing, advice, email, and pros-and-cons responses. The extra capacity showed most clearly on open-ended generation.

## More tokens can mean more failure

The custom benchmark increased maximum generation length. That should have helped models finish more answers, and sometimes it did. It also revealed behaviors that shorter limits had concealed.

Nesso Instruct repeated “Buona giornata!” around a dozen times after an otherwise useful answer. In other tasks it looped corrected sentences or degraded into punctuation and emoji. Open Zagreus repeated advertising-style copy instead of completing a vegetarian follow-up. Qwen3 spent enough tokens reasoning that the visible answer sometimes ended mid-sentence. Qwen3.5 used the extra room to produce longer, more confident hallucinations.

This leads to a practical rule: **maximum tokens are not a quality dial**. They are a budget. The model may spend that budget completing, reasoning, repeating, or elaborating an error.

## The universal failure table

Four task families exposed shared weaknesses across both languages.

| Task | What went wrong |
|---|---|
| Traditional carbonara | Models added béchamel, cream, cheddar, veal, tomatoes, soy milk, vinegar, or used the wrong technique |
| Italian literature | Models listed García Márquez, Fitzgerald, Dostoevsky, invented books, or repeated the same title |
| Logical reasoning | Only Qwen3’s thinking mode solved the tested syllogism reliably in both languages |
| Grammar correction | Models missed or misexplained Italian gender agreement and only partly repaired the English errors |

![Universal failure tasks](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/images/08_universal_failures.png?raw=true)

The carbonara failure is funny, but it is not trivial. It reveals the difference between fluent association and grounded cultural knowledge. A model can produce a confident, well-formatted recipe whose ingredients violate the defining structure of the dish.

The same pattern becomes dangerous in higher-stakes domains. Fluency makes incorrect information feel finished.

## Why qualitative evaluation changes deployment

Aggregate accuracy can tell us whether a checkpoint improved. It cannot tell us whether the model:

- answers in the wrong language;
- gets the answer right and then repeats it 15 times;
- emits an invalid function argument;
- hides useful reasoning behind a truncated visible response;
- invents a proper noun with complete confidence;
- fails only after a multi-turn follow-up.

Those behaviors determine whether an application is delightful, embarrassing, or unsafe. They also point directly to interventions: language gating, repetition penalties, stop-token monitoring, retrieval, calculators, constrained decoding, selective reasoning modes, and targeted fine-tuning.

The benchmark’s real output is therefore not a ranking. It is a map from failure to system design.

In the final episode, we use that map to assemble a practical edge-AI stack—and ask what these seven models should become next.

---

**Previous:** [Episode 5 — Meet Nesso](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/blog-series/episode-05-meet-nesso.md)
**Next:** [Episode 7 — Small Models, Real Systems](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/blog-series/episode-07-edge-future.md)
**Series:** [View all episodes](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/blog-series/README.md)
