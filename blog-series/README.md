# The Zagreus–Nesso Chronicles

## A seven-episode series about building useful small language models from scratch

Most stories about language models begin with scale. This one begins with a constraint: **what can a small, focused, open model become when it is built deliberately for European languages and edge deployment?**

The Zagreus–Nesso Chronicles follows the complete journey—from the first idea to one trillion tokens, distributed training, seven released models, candid benchmarks, and the practical lessons needed to deploy them.

| Episode | Title | The story |
|---:|---|---|
| 1 | [Why Build a Language Model from Scratch?](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/blog-series/episode-01-why-build-from-scratch.md) | Sovereign AI, edge intelligence, and the case for starting at zero |
| 2 | [Before the First GPU Wakes Up](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/blog-series/episode-02-data-pipeline.md) | Open datasets, tokenization, storage, and three weeks of CPU work |
| 3 | [Sixty-Four GPUs and a Falling Loss Curve](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/blog-series/episode-03-training.md) | Architecture, Nanotron, Slurm, and the reality of distributed pretraining |
| 4 | [Meet Zagreus](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/blog-series/episode-04-meet-zagreus.md) | Four bilingual foundations for Italian, Spanish, French, and Portuguese |
| 5 | [Meet Nesso](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/blog-series/episode-05-meet-nesso.md) | How post-training turns language completion into conversation and action |
| 6 | [The Benchmark That Refused to Flatter Us](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/blog-series/episode-06-benchmarks.md) | Wins, loops, hallucinations, bad carbonara, and why failure analysis matters |
| 7 | [Small Models, Real Systems](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/blog-series/episode-07-edge-future.md) | Routing, tools, deployment, openness, and what comes next |

![Italian and English model leaderboard](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/images/01_leaderboard.png?raw=true)

### The model cast

- **Zagreus-0.4B-ita** — English and Italian base model
- **Zagreus-0.4B-spa** — English and Spanish base model
- **Zagreus-0.4B-por** — English and Portuguese base model
- **Zagreus-0.4B-fra** — English and French base model
- **Nesso-0.4B-instruct** — English and Italian conversational model
- **Nesso-0.4B-agentic** — English and Italian model for structured and tool-oriented work
- **Open-Zagreus-0.4B** — the fully reproducible, public-data post-trained model

For the complete story in one article, read [Seven Small Models, One Large Ambition](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/BLOG_POST.md). For configurations, commands, and full benchmark tables, consult the [technical report](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/README.md).
