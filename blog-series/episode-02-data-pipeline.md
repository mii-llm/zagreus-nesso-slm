# Episode 2 — Before the First GPU Wakes Up

## A trillion tokens, several terabytes, and three weeks of preparation

*This is Episode 2 of [The Zagreus–Nesso Chronicles](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/blog-series/README.md). [Start with Episode 1](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/blog-series/episode-01-why-build-from-scratch.md).*

The most photogenic moment in language-model training is a falling loss curve glowing on a dashboard. The least photogenic moment happens weeks earlier, while CPUs turn mountains of raw text into integers.

No model can learn directly from webpages, PDFs, or source-code files. It sees tokens: numerical IDs produced by a tokenizer. Constructing a good training corpus therefore means solving two different problems. First, choose what the model should read. Then transform that material into a format the training system can consume fast enough to keep dozens of expensive GPUs busy.

For Zagreus, that work operated at roughly **one trillion tokens**.

## Building bilingual diets

Each of the four base models combines a large English corpus with material in its target language and a substantial amount of source code. The project relied on openly available datasets from the Hugging Face ecosystem:

- **FineWeb**, supplying broad English web text.
- **FineWeb 2**, supplying language-specific web data for Italian, Spanish, Portuguese, and French.
- **FinePDFs**, adding language-specific material extracted from PDF documents.
- **StarCoderData**, contributing source code and technical patterns.

The Italian run is described as approximately **400 billion English tokens, 400 billion Italian tokens, and 200 billion code tokens**. That mixture reveals the intended personality of the model. English gives it reach; Italian gives it depth in its primary non-English language; code gives it exposure to precise syntax, structured patterns, and technical material.

The French, Portuguese, and Spanish models follow the same principle with their own language-specific FineWeb 2 and FinePDFs collections.

Dataset composition is architecture by other means. Long before researchers choose a learning rate, they decide which parts of human expression the model will encounter most often. Every percentage in the mixture is an opinion about the capabilities worth developing.

## Why use an existing tokenizer?

The team selected the tokenizer from **Llama 3.2 1B**, with a vocabulary of 128,256 tokens. Reusing a strong multilingual tokenizer brought three advantages.

First, its vocabulary already handles multiple writing patterns and languages well. Second, it connects the project to widely tested tooling in the Transformers ecosystem. Third, it avoids making tokenizer development a separate research project before model training can even begin.

This choice does not make Zagreus a derivative set of weights. The models were initialized and pretrained from scratch. The tokenizer is the alphabet they inherited, not the knowledge in their network.

## Tokenization is an industrial workload

The conversion pipeline used Hugging Face’s [datatrove](https://github.com/huggingface/datatrove). Parquet readers loaded documents, a `DocumentTokenizer` transformed text into sequences of IDs, and Slurm scheduled the work across cluster CPUs.

At small scale, tokenization feels like a preprocessing step. At one trillion tokens, it becomes its own distributed data project.

The team observed a rough storage rule: one gigabyte of raw text can expand to about three gigabytes of tokenized output. Depending on the exact storage format, sharding, and compression, a trillion-token corpus may demand **three to five terabytes**. The complete tokenization process ran continuously for **more than three weeks**.

That time is easy to underestimate. CPU allocation, disk throughput, file counts, sharding, retries, and output naming all matter. A malformed shard discovered late can waste days. Too few shards can bottleneck parallel training. Too many tiny files can punish the filesystem. A model-training pipeline is only as fast as the slowest system feeding it.

> **The first scaling challenge was not matrix multiplication. It was turning text into a dependable stream of tokens.**

## Data lineage is part of the model

Using open datasets gives the project an important kind of inspectability. Researchers can identify the major sources, understand the language balance, and build related experiments. It also makes the differences between the releases clearer.

The four Zagreus foundations use open corpora. Nesso Instruct and Nesso Agentic use an internally curated post-training collection developed through work in finance, cybersecurity, function calling, and agentic workflows. Their weights and evaluations are released, while that final behavioral dataset remains private.

Open Zagreus takes the other path. Its supervised fine-tuning uses [OpenItalianData](https://huggingface.co/datasets/DeepMount00/OpenItalianData), published by Michele Montebovi and the Italian open-source community. This makes the final behavior-shaping stage public too.

That distinction is worth stating plainly. “Open model” can refer to several layers—code, base data, post-training data, recipes, or weights. Open Zagreus exists to push reproducibility through the complete post-training path.

## The corpus is ready. Now make it learn.

After weeks of CPU processing, the result is not a model. It is an enormous, carefully sharded numerical library waiting to be read again and again at high speed.

The next challenge is to design a network small enough for edge use, large enough to learn useful abstractions, and stable enough to train across 64 GPUs. That requires an architecture, a parallelism strategy, a scheduler, compatible low-level libraries, and the patience to watch a job whose estimated finish may be months away.

In Episode 3, the GPUs finally wake up.

---

**Previous:** [Episode 1 — Why Build a Language Model from Scratch?](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/blog-series/episode-01-why-build-from-scratch.md)
**Next:** [Episode 3 — Sixty-Four GPUs and a Falling Loss Curve](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/blog-series/episode-03-training.md)
**Series:** [View all episodes](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/blog-series/README.md)
