# Episode 7 — Small Models, Real Systems

## Routing, tools, openness, and the future of intelligence at the edge

*This is the final episode of [The Zagreus–Nesso Chronicles](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/blog-series/README.md). [Start with Episode 1](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/blog-series/episode-01-why-build-from-scratch.md).*

After roughly a trillion tokens, 64 GPUs, four base models, three post-trained models, and a benchmark full of both triumphs and spectacularly wrong recipes, one conclusion stands out:

**A small model becomes useful when the surrounding system respects what it is.**

A 0.4B-parameter network is not a compressed replacement for every AI service. It is a fast linguistic component with particular strengths, blind spots, and operational advantages. Build as if it knows everything and it will hallucinate. Build around its strengths and it can become the responsive center of a capable edge application.

## Route by language first

The bilingual benchmark makes language routing non-negotiable.

Open Zagreus produced Italian during the tested English run and should remain in Italian-only paths. Nesso Agentic was the strongest cross-language option among the project’s own post-trained models, with 73/100 in Italian and 66/100 in English. The external Qwen references were much stronger in English than Italian.

A production router can detect the prompt language before inference and choose deliberately:

- **Italian actions and structured work:** Nesso Agentic
- **Italian conversation and concise writing:** Nesso Instruct, after generation controls are tuned
- **Fully reproducible Italian experiments:** Open Zagreus
- **Spanish, Portuguese, or French specialization:** the corresponding Zagreus base followed by task-specific post-training
- **English-first rich generation:** a stronger English reference model when the deployment permits it

Language detection is cheap. Asking the wrong model to recover from a mismatched route is expensive and unreliable.

## Give arithmetic to a calculator

Nesso Agentic’s Italian benchmark lead coexisted with a fundamental percentage error. Qwen models reasoned more reliably on the tested math prompt, but thinking mode increased latency and sometimes consumed the visible-answer budget.

The production answer is not necessarily a larger model. It is a tool.

When a prompt contains arithmetic, the model can extract the expression, send it to a calculator, and verbalize the result. The same pattern extends to dates, currency conversions, database lookups, and code execution. The model supplies language understanding and orchestration; a deterministic system supplies correctness.

This is where agentic tuning becomes more than a benchmark category. Structured function calls give small models handles on capabilities they should not store in their weights.

## Ground facts instead of wishing harder

The benchmark’s hallucinations clustered around cultural and factual recall: Rome landmarks, the Sistine Chapel, Italian books, traditional recipes, and population claims.

A retrieval layer can replace invention with evidence. For a tourism application, retrieve from a curated destination database. For an internal assistant, search authorized documents. For product support, ground answers in the current manual. For culinary applications, provide a vetted recipe collection.

Retrieval does not solve every reasoning error, but it changes the model’s task from “remember the truth” to “use the supplied truth.” That is a far better fit for limited capacity.

## Treat generation settings as product code

Nesso Instruct’s repetition loops demonstrate that inference settings belong in the product’s tested configuration—not in an afterthought panel.

The repository’s analysis recommends a repetition penalty of at least 1.3 for the affected runs, along with end-of-sequence monitoring. Open Zagreus benefits from a conservative output cap in its tested Italian use. Qwen3’s thinking mode should be enabled selectively for math, logic, and code rather than spending 8–12 seconds on simple classification or email prompts.

Every deployment should test:

- repetition penalty and sampling parameters;
- maximum output length by task;
- stop tokens and chat-template alignment;
- structured-output validity;
- language compliance;
- latency on the actual target hardware;
- multi-turn behavior, not only isolated prompts.

The model weights do not define the user experience alone. Decoding is part of the application.

## Fine-tune where the failures cluster

Qualitative evaluation identifies compact, high-value post-training targets. The models do not need another random collection of general instructions. They need focused examples where multiple systems failed:

- Italian gender agreement and error explanations;
- canonical Italian literature;
- traditional Italian culinary knowledge;
- logical syllogisms;
- clean termination after a complete answer;
- grounded use of proper nouns and dates.

A small, carefully reviewed dataset can be disproportionately valuable at this stage. Post-training is less about volume than about teaching the exact behavior the product requires.

The models are currently reported at the supervised fine-tuning stage. Preference optimization is a logical next step for response quality, but it should follow—not replace—careful diagnosis of formatting, stopping, data, and routing.

## Openness compounds

The most durable contribution of Zagreus and Nesso may not be a leaderboard position. It is the release of a complete learning trail: open base-data sources, architecture configuration, Slurm scripts, Nanotron modifications, checkpoint evaluations, Axolotl recipes, public model weights, and a fully public post-training path through OpenItalianData.

That trail lowers the cost of the next experiment. A Portuguese team can begin from Zagreus Portuguese. An Italian researcher can reproduce Open Zagreus and alter one variable. An infrastructure engineer can reuse the Slurm-adapted Nanotron fork. A model evaluator can add tasks that reveal a different category of failure.

Open work becomes infrastructure when others can extend it.

## Seven models, one larger idea

The project began by asking whether a small language model could be created from first principles for European languages and edge use. The answer is now more nuanced than yes or no.

Yes, a 0.4B model can become a useful Italian assistant. Yes, targeted bilingual pretraining can compete impressively inside a language-specific benchmark. Yes, post-training can turn the same foundation toward conversation, tools, or reproducibility.

And no, small models do not escape the need for systems design. They require routing, grounding, deterministic tools, careful decoding, targeted evaluation, and honest limits.

That is not a weakness of the edge-AI vision. It is the architecture of it.

In a field that measures progress by making models larger, Zagreus and Nesso point toward another kind of scale: more languages treated as first-class, more teams able to own their stack, more devices capable of local intelligence, and more research that can be inspected from data to behavior.

The models are small. The design space they open is not.

---

**Previous:** [Episode 6 — The Benchmark That Refused to Flatter Us](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/blog-series/episode-06-benchmarks.md)
**Series:** [Return to all episodes](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/blog-series/README.md)
**Models:** [Explore mii-llm on Hugging Face](https://huggingface.co/mii-llm)
**Technical details:** [Read the complete project report](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/README.md)
