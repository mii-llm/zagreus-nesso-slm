# Episode 1 — Why Build a Language Model from Scratch?

## The case for sovereign intelligence at the edge

*This is Episode 1 of [The Zagreus–Nesso Chronicles](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/blog-series/README.md), a seven-part series about building small language models from first principles.*

There is a perfectly reasonable response to the idea of training a language model from scratch: **why?**

Excellent open models already exist. They arrive pretrained, documented, and ready to fine-tune. Starting from zero means collecting data, tokenizing it, designing an architecture, coordinating a cluster, surviving failed runs, converting checkpoints, and evaluating behaviors that more established models solved long ago.

And yet, starting from zero is exactly what the [mii-llm](https://mii-llm.ai) community set out to do.

The project began when Antonio Baldassarra and Marco Cristofanilli of [Seeweb](https://www.seeweb.it) offered access to a cluster totaling 64 NVIDIA A100 GPUs. The brief was ambitious but sharply focused: create a state-of-the-art Small Language Model of roughly 500 million parameters, optimize it for edge use, and give several Romance languages first-class treatment.

That combination—small scale, linguistic focus, and full ownership—became the project’s north star.

## Intelligence is moving closer to the user

The biggest models are extraordinary shared utilities, but they are not the only shape AI will take. Language intelligence is also moving into private clouds, industrial systems, local applications, vehicles, appliances, and devices that cannot send every prompt to a distant hyperscale service.

At the edge, the economics and engineering change. A useful model must be compact enough to run cheaply, responsive enough to feel immediate, and focused enough to perform well without carrying the entire internet inside its weights. Privacy, predictable infrastructure costs, offline operation, and control over model updates become product requirements rather than philosophical extras.

This is where a 0.4B-parameter model becomes interesting. It cannot compete with a frontier model on universal knowledge. It does not need to. If it can understand the user’s language, follow a narrow instruction, create valid structure, and call the right tool, it can become an effective component inside a larger system.

> **At the edge, intelligence is not one enormous model. It is the right compact model, connected to the right context and tools.**

## Language is infrastructure too

The project also began from a linguistic imbalance. Most general-purpose language models can speak Italian, Spanish, Portuguese, and French, but “can produce text” is not the same as being designed around a language.

Tokenizer efficiency, cultural knowledge, grammatical reliability, evaluation coverage, and training-data distribution all shape the quality of the experience. When a model’s center of gravity is English, other languages may inherit capability without receiving equal attention.

The Zagreus family reverses that relationship. Each base model is bilingual: English supplies broad general and technical coverage, while one Romance language becomes a dedicated training partner.

- English + Italian
- English + Spanish
- English + Portuguese
- English + French

These are not translations of a single finished model. They are four foundations trained for four linguistic homes.

## Sovereignty means understanding the whole stack

“Sovereign AI” can sound abstract. Here it has a practical meaning: the ability to understand, reproduce, modify, and operate the system without depending on a black box.

That requires more than releasing model weights. It means documenting the dataset sources, tokenizer, architecture, distributed training configuration, checkpoint conversion, post-training recipe, and evaluation commands. It also means publishing failure modes instead of presenting benchmarks as marketing copy.

The result is a seven-model family:

1. Four **Zagreus** bilingual base models.
2. **Nesso Instruct**, tuned for conversation and instruction following.
3. **Nesso Agentic**, tuned for structured output and function-oriented tasks.
4. **Open Zagreus**, trained with a public post-training dataset for end-to-end reproducibility.

The value of the effort is therefore larger than any one score. The team now owns the knowledge required to create a model, not merely consume one. Other researchers can inspect the decisions, reproduce parts of the pipeline, and adapt the work to languages or constraints of their own.

## The joy and the pain

Training from scratch is equal parts scientific ambition and systems endurance. There is joy in watching a randomly initialized network begin to model language. There is pain in spending weeks on preprocessing before training can start, debugging compatibility across CUDA, NCCL, drivers, compilers, and Python libraries, and seeing an estimated completion time measured in months.

Both sides matter. The project is compelling not because everything worked cleanly, but because the difficult parts are visible. Its benchmarks contain regressions. Its instruction models sometimes repeat themselves. Tiny models hallucinate. Checkpoints improve unevenly. Those details turn a model release into reusable engineering knowledge.

Seven models eventually emerged from the experiment. But before the first GPU could learn anything, roughly a trillion tokens had to be found, organized, and transformed.

That is where Episode 2 begins.

---

**Next:** [Episode 2 — Before the First GPU Wakes Up](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/blog-series/episode-02-data-pipeline.md)
**Series:** [View all episodes](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/blog-series/README.md)
