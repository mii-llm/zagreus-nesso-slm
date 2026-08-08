# Episode 5 — Meet Nesso

## How post-training turns language completion into conversation and action

*This is Episode 5 of [The Zagreus–Nesso Chronicles](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/blog-series/README.md). [Start with Episode 1](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/blog-series/episode-01-why-build-from-scratch.md).*

A pretrained model can complete a sentence. An assistant must do something more subtle: understand that a person has asked a question, infer what a useful response looks like, produce only its side of the conversation, and stop at the right moment.

That transformation happens during post-training.

The Nesso models begin with the English–Italian Zagreus foundation and undergo supervised fine-tuning, or SFT. Instead of learning from undifferentiated text, they learn from structured demonstrations: user messages paired with desired assistant responses. The amount of data is far smaller than in pretraining, and the GPU demand is lower. Yet every example carries more behavioral weight.

The project uses [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) for this stage, with distributed training managed through Slurm and fully sharded data parallelism. Examples are packed into 4,096-token sequences, while the loss is applied only to assistant turns. The model therefore learns the response rather than memorizing both sides as equally predictable text.

## Nesso-0.4B-instruct: a compact conversationalist

[Nesso-0.4B-instruct](https://huggingface.co/mii-llm/nesso-0.4B-instruct) is tuned for general conversation and instruction following in English and Italian.

On the project’s standard bilingual evaluation—combining IFEval, ARC, HellaSwag, and MMLU—it achieved an overall average of **0.3345**. That placed it ahead of the reported **0.2576** for LiquidAI/LFM2-350M and close to **0.3545** for Qwen3-0.6B.

Its practical strengths appear in tasks where form and tone matter. In the custom Italian benchmark, it produced the best formal email and the strongest quick-dinner list. It also handled short creative and pros-and-cons prompts well.

But post-training can create failure modes that base-model metrics do not reveal. In longer generations, Nesso Instruct sometimes repeated a correct sentence, sign-off, or malformed pattern many times. More available output tokens made the issue more visible rather than more capable.

This is an engineering problem with several possible layers: generation settings, repetition penalties, end-of-sequence configuration, training examples, and future preference optimization. The important point is that the model’s public evaluation names the problem clearly. A useful release tells deployers what to configure, not only what to celebrate.

## Nesso-0.4B-agentic: useful intelligence has handles

[Nesso-0.4B-agentic](https://huggingface.co/mii-llm/nesso-0.4B-agentic) is optimized for function calling, structured responses, and agentic execution patterns.

In the 20-task Italian benchmark, it scored **73/100**, the highest result among all five tested models. It produced the strongest code answer, excellent translation, reliable follow-up behavior, nuanced pros-and-cons analysis, and the best calorie estimate. Its English score of **66/100** remained competitive with the larger Qwen reference models.

![Italian benchmark radar](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/images/03_radar_it.png?raw=true)

The agentic focus suggests a particularly good role for a model this small. Facts can come from retrieval. Arithmetic can come from a calculator. Live information can come from APIs. A compact model can concentrate on recognizing intent, selecting a function, and formatting valid arguments.

That division of labor makes the system more capable and more auditable. Instead of asking 0.4B parameters to hallucinate a current answer, we can ask them to create a structured request for the system that actually knows.

Nesso Agentic is not immune to tiny-model weaknesses. It made a fundamental percentage error in one Italian math prompt, misclassified simple words, and invented Rome attractions. Those failures strengthen the case for tools and grounding rather than weakening the model’s purpose.

## Open-Zagreus-0.4B: behavior anyone can reproduce

[Open-Zagreus-0.4B](https://huggingface.co/mii-llm/open-zagreus-0.4B) is the third post-trained release and the project’s transparency experiment. Its SFT stage uses [OpenItalianData](https://huggingface.co/datasets/DeepMount00/OpenItalianData), a public dataset released by Michele Montebovi and the Italian open-source community.

On EVALITA, Open Zagreus improved the base Italian model’s overall score from **0.3226 to 0.3313**. It gained on summarization, text entailment, sentiment, and word-in-context tasks.

![EVALITA base and open-model comparison](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/images/evalita_comparison.png?raw=true)

The gains were not universal. Hate-speech detection, named-entity recognition, and relation extraction declined. Fine-tuning reallocates behavior; it does not simply add a fixed amount of intelligence to every task.

Open Zagreus matters because every behavioral ingredient can be inspected and changed. A researcher can reproduce the SFT stage, remove examples, add a domain, change the chat template, or study why one capability improved while another regressed.

## The three faces of the same foundation

Together, the post-trained models show how profoundly behavior depends on data:

- **Nesso Instruct** speaks as a general assistant.
- **Nesso Agentic** turns requests into structured, actionable work.
- **Open Zagreus** prioritizes a reproducible public-data path.

The shared foundation does not dictate one final personality. Post-training is product design performed through examples.

The next question is unavoidable: how good are these models when the prompts become messy, cultural, multilingual, or deceptively simple? Episode 6 introduces the benchmark that answered honestly.

---

**Previous:** [Episode 4 — Meet Zagreus](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/blog-series/episode-04-meet-zagreus.md)
**Next:** [Episode 6 — The Benchmark That Refused to Flatter Us](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/blog-series/episode-06-benchmarks.md)
**Series:** [View all episodes](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/blog-series/README.md)
