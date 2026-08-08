# Episode 4 — Meet Zagreus

## Four bilingual foundations for four linguistic homes

*This is Episode 4 of [The Zagreus–Nesso Chronicles](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/blog-series/README.md). [Start with Episode 1](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/blog-series/episode-01-why-build-from-scratch.md).*

One architecture. Four training runs. Four different linguistic identities.

The Zagreus family consists of base language models: networks pretrained to predict text, not yet shaped into conversational assistants. Each pairs English with one Romance language—Italian, Spanish, Portuguese, or French—and each is released as a foundation that others can evaluate, continue training, or adapt to a specific domain.

Why four models instead of one multilingual checkpoint? At 0.4B parameters, capacity is precious. A single model forced to distribute that capacity across many languages may become broad but shallow. Four bilingual models let each target language occupy a much larger share of the training mixture while English preserves access to general and technical material.

## Zagreus-0.4B-ita: the family’s center

[Zagreus-0.4B-ita](https://huggingface.co/mii-llm/zagreus-0.4B-ita) is the English–Italian foundation and the base from which the post-trained Nesso models grow.

The team evaluated checkpoints throughout training on Italian MMLU, HellaSwag, and ARC. The reported three-task average began at **0.2849 at 95k steps** and peaked around **0.2981 at 365k** among the listed checkpoints. On EVALITA—a broader suite spanning admission questions, FAQ matching, hate-speech detection, named entities, sentiment, summarization, entailment, and word-in-context—the base model reached **0.3226 overall**.

![Zagreus Italian checkpoint progression](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/images/zagreus-ita.png?raw=true)

The checkpoint history contains a valuable warning. Performance did not rise in a straight line. At 460k steps HellaSwag improved to 0.3778, while the combined average remained below the earlier 365k result because other tasks moved differently. Pretraining loss summarizes the learning objective; it does not guarantee that every downstream ability improves together.

That makes Zagreus Italian more than the best-developed member of the base family. It is also a record of the decisions required to turn a long training run into a release.

## Zagreus-0.4B-spa: steady progress in Spanish

[Zagreus-0.4B-spa](https://huggingface.co/mii-llm/zagreus-0.4B-spa) pairs English with Spanish web and PDF data. Across the reported Spanish MMLU, ARC, and HellaSwag checkpoints, its combined average rose from **0.309 at 146k steps** to **0.321 at 518k**.

![Zagreus Spanish checkpoint progression](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/images/zagreus-spa.png?raw=true)

The curve is modest but consistent. That is a useful profile for a base model: a compact checkpoint with room for domain-specific continued pretraining or supervised fine-tuning. A team building a Spanish support assistant, classifier, or private document interface does not need the base model to act like a polished chatbot. It needs a strong linguistic substrate it can shape.

## Zagreus-0.4B-fra: a French-first foundation

[Zagreus-0.4B-fra](https://huggingface.co/mii-llm/zagreus-0.4B-fra) follows the same bilingual design for French. Its reported three-task average reached **0.321 at 705k steps**, with HellaSwag at 0.417, ARC at 0.281, and French MMLU at 0.266.

![Zagreus French checkpoint progression](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/images/zagreus-fra.png?raw=true)

At this size, the model is best understood as infrastructure. It offers a transparent starting point for researchers who want to study French adaptation without inheriting a much larger, less specialized checkpoint. Its value lies in what can be built on top of it as much as in its zero-shot scores.

## Zagreus-0.4B-por: specialization pays

[Zagreus-0.4B-por](https://huggingface.co/mii-llm/zagreus-0.4B-por) produced one of the project’s most striking base-model results. At 582k steps it reached a reported average of **0.3113** across Portuguese ARC, HellaSwag, and MMLU.

The team also evaluated it using Eduardo Garcia’s Portuguese fork of the LM Evaluation Harness. Across nine tasks—including RTE, semantic similarity, exams, natural-language inference, hate speech, and sentiment—the 483k checkpoint averaged **0.3230**. In the reported comparison, **Qwen3-0.6B-Base scored 0.2569**.

![Portuguese community benchmark comparison](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/images/por-garcia.png?raw=true)

One benchmark cannot establish universal superiority, and the best result came from 483k rather than the later 582k checkpoint. Still, the comparison validates the core hypothesis: in a targeted language setting, deliberate data composition can outweigh a competitor’s parameter advantage.

## What a base model is—and is not

Base models are often judged unfairly because users prompt them as if they were assistants. Pretraining teaches text continuation. It does not reliably teach when to stop, how to follow a request, which parts of a conversation belong to the assistant, or how to produce a safe structured function call.

That rawness is useful. It gives researchers a less behaviorally opinionated foundation for experimentation. But it also explains why the model family needs a second name.

Zagreus learns language. Nesso learns how to respond.

---

**Previous:** [Episode 3 — Sixty-Four GPUs and a Falling Loss Curve](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/blog-series/episode-03-training.md)
**Next:** [Episode 5 — Meet Nesso](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/blog-series/episode-05-meet-nesso.md)
**Series:** [View all episodes](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/blog-series/README.md)
