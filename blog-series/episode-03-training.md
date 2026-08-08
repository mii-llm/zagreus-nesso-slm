# Episode 3 — Sixty-Four GPUs and a Falling Loss Curve

## Architecture, distributed systems, and the moment random weights begin to learn

*This is Episode 3 of [The Zagreus–Nesso Chronicles](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/blog-series/README.md). [Start with Episode 1](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/blog-series/episode-01-why-build-from-scratch.md).*

At the beginning of pretraining, a language model knows nothing. Its hundreds of millions of parameters are random values. Present it with Italian, French, code, or poetry and it has no concepts, no grammar, and no memory—only an objective: predict the next token a little better than it did before.

Then gradient descent begins.

For the Zagreus models, that transformation ran across **64 NVIDIA A100 GPUs** on eight Seeweb nodes. The cluster provided the raw compute, but getting all 64 accelerators to behave like one training machine required a carefully chosen architecture and a distributed software stack.

## A dense model for a small regime

Zagreus uses a modified Llama-style decoder architecture. Its core configuration includes:

| Component | Configuration |
|---|---:|
| Hidden layers | 32 |
| Hidden size | 960 |
| Attention heads | 15 |
| Key-value heads | 5 |
| Intermediate size | 2,560 |
| Context length | 4,096 tokens |
| Vocabulary | 128,256 tokens |
| Training precision | bfloat16 |

The network is fully dense: every token passes through the same complete set of model parameters. That decision was deliberate. Mixture-of-experts systems activate only selected experts for each token and can be efficient at large scale, but routing has a cost. With roughly 0.4B parameters, expert underuse and communication overhead can erase the theoretical advantage. A dense model is simpler, stable, and keeps its limited capacity working on every example.

The configuration also uses grouped-query attention—15 attention heads sharing five key-value heads—and tied word embeddings. These choices help balance capability and efficiency inside the parameter budget.

## Why Nanotron?

The team evaluated several training frameworks.

NVIDIA’s Megatron-LM is enormously capable but proved difficult to adapt cleanly to the cluster. Llama-Factory is approachable and versatile, though more naturally oriented toward fine-tuning. Andrej Karpathy’s nanoGPT offers exceptional educational clarity, while its successor nanochat covers an increasingly complete stack.

For this project, [Hugging Face Nanotron](https://github.com/huggingface/nanotron) offered the right combination: a relatively minimal pretraining library, native support for three-dimensional parallelism, multi-node operation, and tight integration with Hugging Face tools.

The project used data parallelism across all 64 GPUs. Each replica processed a portion of the batch, gradients were synchronized, and Slurm coordinated the eight nodes. The team also published an [mii-llm fork of Nanotron](https://github.com/mii-llm/nanotron) tailored for direct Slurm use and contributed fixes upstream.

## The cluster is a single machine—until it is not

Slurm makes a multi-node cluster feel coherent. It allocates nodes, starts tasks, sets job metadata, captures logs, and provides the environment that `torchrun` uses to establish distributed workers.

Underneath that clean interface is a long chain of dependencies: GPU drivers, CUDA, cuDNN, NCCL, InfiniBand networking, compilers, PyTorch, and Python packages. A mismatch at any layer can break the run. Distributed failures can be particularly elusive because one unhealthy process may leave dozens of others waiting.

The launch script therefore does more than call a Python file. It loads exact system modules, activates the training environment, configures NCCL networking, establishes a rendezvous endpoint, identifies every node’s rank, and launches the workers together.

This is one reason “we had enough GPUs” is not the same as “we could train a model.” Compute becomes useful only when the surrounding system can feed and coordinate it reliably.

## When the magic happens

One training log captures the scale of the effort. At about step 211,365, the system had consumed **480 billion tokens**. It was processing roughly **183,000 tokens per second** globally and reporting a language-model loss near **2.06**.

The estimated time remaining was still about **130 days**.

That line contains the project’s joy and pain in miniature. The falling loss proves that a network born from random values is learning statistical structure from human language. The estimated finish is a reminder that even a “small” language model can be a major computational undertaking.

Training also produced many checkpoints rather than one final artifact. Evaluation showed that capabilities did not rise smoothly with step count. A later checkpoint could improve HellaSwag while slipping on MMLU or ARC. Selecting a release meant comparing behavior, not merely choosing the greatest number in a folder name.

## From training checkpoint to usable model

Nanotron’s native checkpoints are optimized for distributed training, not immediate use with the Transformers library. After training, a conversion step maps the sharded checkpoint into Hugging Face format and attaches the Llama 3.2 tokenizer configuration.

Only then does the result become easy to load, evaluate, fine-tune, and share using familiar open-source tools.

That conversion marks the end of one journey and the beginning of another. Four networks have learned to model bilingual text. They are foundations, not assistants. Before shaping their behavior, it is time to meet each one.

---

**Previous:** [Episode 2 — Before the First GPU Wakes Up](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/blog-series/episode-02-data-pipeline.md)
**Next:** [Episode 4 — Meet Zagreus](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/blog-series/episode-04-meet-zagreus.md)
**Series:** [View all episodes](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/blog-series/README.md)
