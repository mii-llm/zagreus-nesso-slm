## The Joy and Pain of Training an LLM from Scratch

### A Technical Report on the Development of the Zagreus and Nesso Model Families

---

## 1. Motivation: The Vision of Sovereign Edge Intelligence

Training a fully functional modern neural network, specifically a Large Language Model (LLM), from first principles has been a foundational ambition since the inception of our community [mii-llm](https://mii-llm.ai) that stands for Made in Italy - Large Language Model. In the current landscape, the convergence of distributed computing power and accessible knowledge has never been more potent; consequently, constructing an intelligent machine stands as one of the most exciting tasks a group of machine learning specialists can undertake.

This vision materialized when Antonio Baldassarra (CEO of Seeweb) and Marco Cristofanilli (Head of AI at Seeweb) commissioned us to develop a Small Language Model (SLM) from scratch utilizing the Seeweb infrastructure. Seeweb, a cloud provider with a strategic focus on AI, granted us access to a cluster of on-demand nodes comprising a total of 64 NVIDIA A100 GPUs.

Our primary objective was to experiment and deliver a state-of-the-art SLM with approximately 500 million parameters, built from the ground up and optimized for edge use cases within the Italian language ecosystem. We hypothesize that, in the coming years, intelligent devices—and virtually any hardware equipped with a chip—will be enhanced by neural architectures with embedded reasoning and language capabilities. Small, efficient models will be key to enabling automation at the edge. To address this need, we created Zagreus, arguably one of the few high-performing small language models dedicated to the Italian language.

We are releasing this detailed blog post, covering every step and data point required to reproduce the project, as we strongly believe in the importance of open source in reducing technological and geopolitical dependencies.

## The Joy and Pain of Training an LLM from Scratch

### A Technical Report on the Development of the Zagreus and Nesso Model Families

---

## 1. Motivation: The Vision of Sovereign Edge Intelligence

Training a fully functional modern neural network, specifically a Large Language Model (LLM), from first principles has been a foundational ambition since the inception of our community [mii-llm](https://mii-llm.ai) that stands for Made in Italy - Large Language Model. In the current landscape, the convergence of distributed computing power and accessible knowledge has never been more potent; consequently, constructing an intelligent machine stands as one of the most exciting tasks a group of machine learning specialists can undertake.

This vision materialized when Antonio Baldassarra (CEO of Seeweb) and Marco Cristofanilli (Head of AI at Seeweb) commissioned us to develop a Small Language Model (SLM) from scratch utilizing the Seeweb infrastructure. Seeweb, a cloud provider with a strategic focus on AI, granted us access to a cluster of on-demand nodes comprising a total of 64 NVIDIA A100 GPUs.

Our primary objective was to experiment and deliver a state-of-the-art SLM with approximately 500 million parameters, built from the ground up and optimized for edge use cases within the Italian language ecosystem. We hypothesize that, in the coming years, intelligent devices—and virtually any hardware equipped with a chip—will be enhanced by neural architectures with embedded reasoning and language capabilities. Small, efficient models will be key to enabling automation at the edge. To address this need, we created Zagreus, arguably one of the few high-performing small language models dedicated to the Italian language.

In the spirit of open and reproducible research, we are releasing the full Zagreus and Nesso lineup: seven models in total—four base (pretrained) checkpoints for bilingual models and three post-trained variants. Notably, our post-trained models are designed to compete on standard benchmarks with state-of-the-art models of comparable size, demonstrating that carefully engineered small models can achieve near frontier-level performance within their parameter class.

### Base models:

* [zagreus-0.4B-base-ita]() English Italian bilingual model
* [zagreus-0.4B-base-spa]() English Spanish bilingual model
* [zagreus-0.4B-base-por]() English Portuguese bilingual model
* [zagreus-0.4B-base-fra]() English French bilingual model

### Post-trained models:
* [Nesso-0.4B-instruct]() English Italian for conversational use cases
* [Nesso-0.4B-agentic]() English Italian for agentic and function calling use cases
* [Open-Zagreus-0.4B]() Fully open source data used to train this model


We are releasing this detailed blog post, covering every step and data point required to reproduce the project, as we strongly believe in the importance of open source in reducing technological and geopolitical dependencies.

---

## 2. Technology Stack: Framework Selection

There are numerous frameworks available for creating an LLM from scratch. We conducted a comparative analysis of several options. Below is a summary of our testing and the rationale behind our ultimate decision to utilize Nanotron by Hugging Face.

### Framework Comparative Analysis

[**Megatron-LM:**](https://github.com/NVIDIA/Megatron-LM) Developed by NVIDIA, this is a powerful framework designed for training large transformer models with billions of parameters. While it is likely an optimal choice for large, well-resourced teams, we found it challenging to set up and deploy effectively on our specific cluster infrastructure.

[**Llama-Factory:**](https://github.com/hiyouga/LLaMA-Factory) A versatile and user-friendly open-source framework that simplifies fine-tuning, training, and deployment of a wide range of LLMs. However, our evaluation suggests it is more specialized for fine-tuning than for pre-training from scratch.

**nanoGPT and nanochat:** Both created by Andrej Karpathy, these projects prioritize simplicity and educational value.

[**nanoGPT**](https://github.com/karpathy/nanoGPT) is a minimalist, readable codebase designed as a learning tool, though it is now considered deprecated in favor of its successor.
[**nanochat**](https://github.com/karpathy/nanochat) is the evolution of nanoGPT, offering a full-stack, end-to-end pipeline for building a complete ChatGPT-like chatbot. It covers the entire lifecycle, from tokenization and pre-training to fine-tuning and a web interface, all within a compact and hackable codebase. Although nanochat had not yet been released when we commenced this project, we believe it has a promising future, especially given its recent integration into the Transformers library.

### Our Choice: Hugging Face Nanotron

Ultimately, we selected [Hugging Face Nanotron](https://github.com/huggingface/nanotron). It is a minimalistic library focused on 3D parallelism (Data, Tensor, and Pipeline) specifically for pre-training transformer models. We value Hugging Face for its commitment to openness. We found the library well-suited for multi-node training; furthermore, it is natively integrated into the Hugging Face ecosystem (Accelerate, Datasets, hf-cli), ensuring that workflows—from data tokenization to model release—remain cohesive.

During the development cycle, we identified minor bugs and are actively contributing to the library via Pull Requests. We also established a [fork of Nanotron](https://github.com/mii-llm/nanotron) optimized to run directly on a Slurm cluster.


## 3. Data Engineering: The Tokenization Pipeline

Data is the *sine qua non* for creating an LLM. The volume of data required is contingent upon the target model size and the available compute budget. Operating as a GPU-constrained team—and thanks to the sponsorship from Seeweb—we chose to build a small language model of ~500 million parameters, trained on approximately 1 trillion tokens.

### Dataset Sources

We utilized exclusively open source datasets by the Hugging Face team for creating our four bilingual foundational model released . Below is the data distribution per model:

**mii-llm/nesso-0.4B-ita:**
* [https://huggingface.co/datasets/HuggingFaceFW/fineweb/viewer/sample-350BT](https://huggingface.co/datasets/HuggingFaceFW/fineweb/viewer/sample-350BT) (350 billion tokens)
* [https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/viewer/ita_Latn](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/viewer/ita_Latn)
* [https://huggingface.co/datasets/HuggingFaceFW/finepdfs/viewer/ita_Latn](https://huggingface.co/datasets/HuggingFaceFW/finepdfs/viewer/ita_Latn)
* [https://huggingface.co/datasets/bigcode/starcoderdata](https://huggingface.co/datasets/bigcode/starcoderdata) (250 billion tokens)

**mii-llm/nesso-0.4B-fra:**
* [https://huggingface.co/datasets/HuggingFaceFW/fineweb/viewer/sample-350BT](https://huggingface.co/datasets/HuggingFaceFW/fineweb/viewer/sample-350BT) (350 billion tokens)
* [https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/viewer/fra_Latn](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/viewer/fra_Latn)
* [https://huggingface.co/datasets/HuggingFaceFW/finepdfs/viewer/fra_Latn](https://huggingface.co/datasets/HuggingFaceFW/finepdfs/viewer/fra_Latn)
* [https://huggingface.co/datasets/bigcode/starcoderdata](https://huggingface.co/datasets/bigcode/starcoderdata) (250 billion tokens)

**mii-llm/nesso-0.4B-por:**
* [https://huggingface.co/datasets/HuggingFaceFW/fineweb/viewer/sample-350BT](https://huggingface.co/datasets/HuggingFaceFW/fineweb/viewer/sample-350BT) (350 billion tokens)
* [https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/viewer/por_Latn](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/viewer/por_Latn)
* [https://huggingface.co/datasets/HuggingFaceFW/finepdfs/viewer/por_Latn](https://huggingface.co/datasets/HuggingFaceFW/finepdfs/viewer/por_Latn)
* [https://huggingface.co/datasets/bigcode/starcoderdata](https://huggingface.co/datasets/bigcode/starcoderdata) (250 billion tokens)

**mii-llm/nesso-0.4B-spa:**
* [https://huggingface.co/datasets/HuggingFaceFW/fineweb/viewer/sample-350BT](https://huggingface.co/datasets/HuggingFaceFW/fineweb/viewer/sample-350BT) (350 billion tokens)
* [https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/viewer/spa_Latn](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/viewer/spa_Latn)
* [https://huggingface.co/datasets/HuggingFaceFW/finepdfs/viewer/spa_Latn](https://huggingface.co/datasets/HuggingFaceFW/finepdfs/viewer/spa_Latn)
* [https://huggingface.co/datasets/bigcode/starcoderdata](https://huggingface.co/datasets/bigcode/starcoderdata) (250 billion tokens)

### The Tokenization Process

Raw datasets are not ready for immediate training; they must first be tokenized. Tokenization is a CPU intensive process that transforms text strings into token sequences (numerical IDs). As a rule of thumb for storage estimation, for every 1 GB of text, approximately 3 GB of tokenized outputs are generated. For ~1 trillion tokens, one typically requires at least 3 to 5 terabytes of disk space (depending on format, sharding strategy, and compression).

We selected the Llama-3.2 tokenizer (from the Llama-3.2-1B model) because its multilingual tokenization capabilities are robust and widely adopted. Using the [datatrove](https://github.com/huggingface/datatrove) library, the process took over three weeks of continuous computation to generate ~1 trillion tokens, stratified as roughly 400B English, 400B Italian, and 200B Code.

Below is the Python script as example used for Slurm pipeline execution:

```python
import os
import sys
from pathlib import Path
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.tokens import DocumentTokenizer
from datatrove.executor import SlurmPipelineExecutor

def create_and_run_tokenization_job(input_dir, base_output_dir):
    """
    Create and execute a tokenization pipeline for a specific directory.
    """
    dir_name = os.path.basename(input_dir)
    output_dir = os.path.join(base_output_dir, f"tokenized_{dir_name}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create pipeline for tokenization
    pipeline = [
        ParquetReader(
            data_folder=input_dir,
            glob_pattern="*.parquet",
            text_key="text",
        ),
        DocumentTokenizer(
            tokenizer_name_or_path="/hub/models--meta-llama--Llama-3.2-1B",
            output_folder=output_dir,
            local_working_dir=dir_name,
            save_filename=f"{dir_name}_tokenized",
            shuffle_documents=False,
        ),
    ]
    # Configure and run the SLURM executor
    executor = SlurmPipelineExecutor(
        job_name=f"tokenize_{dir_name}",
        pipeline=pipeline,
        tasks=1,
        workers=-1,
        time="24:00:00",
        partition="boost_usr_prod",
        logging_dir=os.path.join(output_dir, f"{dir_name}_logs"),
        mem_per_cpu_gb=16,
        slurm_logs_folder=os.path.join(output_dir, f"{dir_name}_slurm_logs"),
        # also pass the SBATCH gres directive to ensure 0 GPUs allocated
        sbatch_args={
            "account": "YOUR_ACCOUNT",
            "gres": "gpu:0",
        },
    )
    
    executor.run()

def main():
    # Base paths
    base_input_path = "INPUT_DIR"
    base_output_path = "OUTPUTDIR”
    
    # List of directories to process
    # Discover directories under base_input_path instead of using a static list
    try:
        entries = os.listdir(base_input_path)
    except FileNotFoundError:
        print(f"Base input path {base_input_path} not found")
        directories = []
    else:
        directories = sorted(
            [
                name
                for name in entries
                if os.path.isdir(os.path.join(base_input_path, name)) and name.startswith("CC-MAIN")
            ]
        )
    
    # Process each directory
    # if i need other 20 directories
    # dir_name in directories[20:40]: 
    for dir_name in directories[60:]:  # Example: limit to first 20 directories
        input_dir = os.path.join(base_input_path, dir_name)
        if os.path.exists(input_dir):
            print(f"Launching tokenization job for {dir_name}")
            create_and_run_tokenization_job(input_dir, base_output_path)
        else:
            print(f"Warning: Directory {input_dir} does not exist, skipping...")

if __name__ == "__main__":
    main()
```

---

## 4. Pre-training: The Core Engine

Pre-training is the foundational step in building an LLM, transforming raw tokenized data into a model capable of context aware text completion. This is the most time consuming and GPU intensive phase. While massive models may require thousands of GPUs, our sub 1 billion parameter model was effectively trained on the 64 GPU cluster provided by Seeweb.

We utilized Nanotron, which supports multiple architectures, including Llama-3.2, Qwen-2.5, and Mixture-of-Experts (MoE) variants. For this project, we adopted a modified Llama-3.2 fully dense architecture. Our design choice was motivated by the hypothesis that, in the small-parameter regime (~500M parameters), fully dense models provide better compute utilization and more stable training dynamics than sparse architectures such as MoE. In tightly constrained capacity settings, the routing overhead and expert under-utilization typical of MoE architectures may offset their theoretical efficiency advantages.

Working with a GPU cluster is streamlined by HPC tools; we employed the Slurm scheduler. Slurm allows the cluster to be viewed as a unified Linux system where jobs can be executed across many GPUs in parallel, while handling checkpoints and logs in real time. The most challenging aspect remains ensuring the software stack—from drivers and CUDA/NCCL to Python libraries—functions harmoniously, often requiring resolution of version and ABI incompatibilities.

Successfully running a distributed training job on the tokenized data was a profound milestone. Observing the loss curve decrease from raw data after days of waiting conveys the sense of operating at the edge of scientific and engineering capability—a genuinely intense moment for a researcher.

For out-of-the-box functionality, we recommend our fork: [https://github.com/mii-llm/nanotron](https://github.com/mii-llm/nanotron) (a fork of [https://github.com/huggingface/nanotron/](https://github.com/huggingface/nanotron/)), pending the merge of our Pull Request.

### Nanotron Training Configuration

Below is the configuration used for the pre-training run:

```yaml
checkpoints:
  checkpoint_interval: 5000
  checkpoints_path: checkpoints_zagreus_ita_v2
  checkpoints_path_is_shared_file_system: false
resume_checkpoint_path: /training/pretraining/nanotron/checkpoints_zagreus_ita_v2/630000 
save_final_state: false
  save_initial_state: false
data_stages:
- data:
    dataset:
      dataset_folder:
      - /training/pretraining/fineweb-ita/tokenized
      - /training/pretraining/fineweb-edu-350BT/000_tokenized_output
      - /training/pretraining/fineweb-edu-350BT/011_tokenized_output
      - /training/pretraining/fineweb-edu-350BT/012_tokenized_output
      - /training/pretraining/fineweb-edu-350BT/013_tokenized_output
      - /training/pretraining/fineweb-edu-350BT/014_tokenized_output
      - /training/pretraining/fineweb-edu-350BT/015_tokenized_output
      - /training/pretraining/fineweb-edu-350BT/016_tokenized_output
      - /training/pretraining/finepdf-ita/000_tokenized_output
      - /training/pretraining/starcoder_tokenized/000_tokenized_output   
    num_loading_workers: 0
    seed: 8
  name: stable phase
  start_training_step: 1
general:
  benchmark_csv_path: null
  consumed_train_samples: null
  ignore_sanity_checks: true
  project: zagreus
  run: zagreus-350M
  seed: 8
  step: null
logging:
  iteration_step_info_interval: 1
  log_level: info
  log_level_replica: info
model:
  ddp_bucket_cap_mb: 100
  dtype: bfloat16
  init_method:
    std: 0.03227
  make_vocab_size_divisible_by: 1
  model_config:
    bos_token_id: 128000
    eos_token_id: 128001
    hidden_act: silu
    hidden_size: 960
    initializer_range: 0.02
    intermediate_size: 2560
    is_llama_config: true
    max_position_embeddings: 4096
    num_attention_heads: 15
    num_hidden_layers: 32
    num_key_value_heads: 5
    pad_token_id: null
    pretraining_tp: 1
    rms_norm_eps: 1.0e-05
    rope_interleaved: false
    rope_scaling: null
    rope_theta: 10000.0
    tie_word_embeddings: true
    use_cache: true
    vocab_size: 128256
optimizer:
  accumulate_grad_in_fp32: true
  clip_grad: 1.0
  learning_rate_scheduler:
    learning_rate: 0.003
    lr_decay_starting_step: 750000
    lr_decay_steps: 50000
    lr_decay_style: linear
    lr_warmup_steps: 4000
    lr_warmup_style: linear
    min_decay_lr: 1.0e-7
  optimizer_factory:
    adam_beta1: 0.9
    adam_beta2: 0.95
    adam_eps: 1.0e-08
    name: adamW
    torch_adam_is_fused: true
  weight_decay: 0.01
  zero_stage: 0        
parallelism:
  dp: 64
  expert_parallel_size: 1
  pp: 1
  pp_engine: 1f1b
  recompute_layer: false
  tp: 1
  tp_linear_async_communication: true
  tp_mode: REDUCE_SCATTER
  tp_recompute_allgather: true
profiler: null
tokenizer:
  tokenizer_max_length: null
  tokenizer_name_or_path: meta-llama/Llama-3.2-1B
  tokenizer_revision: null
tokens:
  batch_accumulation_per_replica: 1
  limit_test_batches: 0
  limit_val_batches: 0
  micro_batch_size: 4
  sequence_length: 4096
  train_steps: 2000000
  val_check_interval: 5000
```

### Slurm Execution

The command for launching Nanotron on Slurm with 64 GPUs across 8 nodes (based on the provided configuration context) is as follows:

```bash
#SBATCH --job-name=350_it
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=PARTITITION
#SBATCH --nodes=8               # 4 nodes
#SBATCH --gres=gpu:8            # 8 A100 per node
#SBATCH --cpus-per-task=32
#SBATCH --time=4-00:00:00
#SBATCH --output=slurm-%j.out

################ 0. Environment ################
module purge
module load profile/global
module load python/3.11 cuda/12.2 cudnn nccl gcc

source /path/to/venv/nanotron/bin/activate

export HF_HOME=/path/to/hf_home
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# NCCL over IB
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME="ib0,eno,eth"
export WANDB_MODE=disabled


################ 1. Distributed vars ############
GPUS_PER_NODE=4
NNODES=$SLURM_JOB_NUM_NODES           # 2
NODE_RANK=$SLURM_NODEID               # 0 or 1
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
MASTER_PORT=29400                     # free port on master
RDZV_ID=$SLURM_JOB_ID                 # unique per job

################ 2. Launch ######################
srun torchrun \
      --nnodes $NNODES \
      --nproc_per_node $GPUS_PER_NODE \
      --rdzv_id $RDZV_ID \
      --rdzv_backend c10d \
      --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
      /path/to/nanotron/run_train.py --config-file smollm2/zagreus_350M_ita.yaml
```

When successful, the training logs indicate the model convergence (“the magic happens”):

```text
…..
12/21 01:39:51 [INFO|DP=0|TP=0|lrdn0007]: iteration: 211364 / 1500000 | consumed_tokens: 480B | time_per_iteration_ms: 8.79K | tokens_per_sec: 182K | tokens_per_sec_per_gpu: 699 | global_batch_size: 1.6M | grad_norm: 0.112 | lm_loss: 2.04 | lr: 0.0001 | model_tflops_per_gpu: 15.4 | eta: 131 days, 1:45:39
12/21 01:40:00 [INFO|DP=0|TP=0|lrdn0007]: iteration: 211365 / 1500000 | consumed_tokens: 480B | time_per_iteration_ms: 8.74K | tokens_per_sec: 183K | tokens_per_sec_per_gpu: 703 | global_batch_size: 1.6M | grad_norm: 0.105 | lm_loss: 2.06 | lr: 0.0001 | model_tflops_per_gpu: 15.5 | eta: 130 days, 9:39:28
…
```

### Model Conversion

Once checkpoints are generated, they are not compatible with the Transformers library by default. Nanotron provides a script to convert the checkpoint into a fully compatible Hugging Face model:

```bash
torchrun --nproc_per_node=1 -m examples.llama.convert_nanotron_to_hf \
  --checkpoint_path=checkpoints/544000 \
  --save_path=hf_checkpoints/544000 \
  --tokenizer_name meta-llama/Llama-3.2-1B
```

# 5. Post-Training: Shaping Behavior

Creating a base model from scratch represents a major technical achievement, and we consider this work a contribution to the open community. However, a foundation model alone — even with a fully reproducible pipeline and transparent data distribution is rarely sufficient for direct real-world deployment. The post-training phase is responsible for shaping the model’s behavior toward practical usability.

This phase typically requires significantly fewer GPUs and a smaller data volume compared to pre-training. However, the *quality* and *curation strategy* of the data become substantially more important than raw scale.

We utilized **Axolotl** for post-training due to our extensive experience with the framework and its stability in multi-GPU environments. While we initially encountered configuration challenges when integrating it with our Slurm-based HPC setup, we successfully adapted the workflow to support distributed execution.

We possess extensive experience in post-training language models. Over the past several years, we have post-trained models for domain-specific applications including finance, cybersecurity, structured function calling, and agentic execution patterns. Through this work, we have curated a substantial internal dataset collection that enables controlled experimentation across varied instruction-following regimes.

This dataset collection, built with meticulous care and long-term iteration, constitutes a strategic asset for our research group. For this reason, we have decided not to publish it as open source, as we consider it a competitive advantage. Nevertheless, we believe that releasing the trained models and all evaluation results provides significant value to the broader community.

Most importantly, we demonstrate that we have been able to build and release a model that performs competitively head to head with state of the art models of similar parameter scale.

We are releasing three primary post-trained models:

* **Nesso-0.4B-instruct**: optimized for conversational and instruction-following use cases.
* **Nesso-0.4B-agentic**: optimized for function calling, structured outputs, and agentic execution patterns.

Both models utilize **Nesso-0.4B-ita** as the base and are trained on a bilingual corpus (English/Italian).

It is important to note that both models are currently at the **SFT (Supervised Fine-Tuning)** stage. In the coming weeks, we will execute the **DPO (Direct Preference Optimization)** stage and subsequently update both the models and their evaluation results.

We also released a third, fully open model: **Open-Nesso (Open-Zagreus)**.

Thanks to the work of the Italian open-source community **mii-llm**, and in particular Michele Montebovi who published the SFT dataset *OpenItalianData*—all data used and all training recipes for this model are fully open and reproducible as a full open source model from data to weights.

---

## Post-Training Slurm Script

```bash
#!/bin/bash
#SBATCH --job-name=ax_2n
#SBATCH --account=MII
#SBATCH --nodes=4               # 2 nodes
#SBATCH --gres=gpu:8            # 4 A100 per node
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out

################ 0. Environment ################
module purge
module load profile/global
module load cuda/12.2 cudnn nccl

source /training-venv/bin/activate

export HF_HOME=/
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# NCCL over IB
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME="ib0,eno,eth"

################ 1. Distributed vars ############
GPUS_PER_NODE=4
NNODES=$SLURM_JOB_NUM_NODES
NODE_RANK=$SLURM_NODEID
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
MASTER_PORT=29400
RDZV_ID=$SLURM_JOB_ID

################ 2. Launch ######################
srun torchrun \
      --nnodes $NNODES \
      --nproc_per_node $GPUS_PER_NODE \
      --rdzv_id $RDZV_ID \
      --rdzv_backend c10d \
      --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
      -m axolotl.cli.train \
      training/opendata-zagreus-350M-sft-fsdp-debug.yaml
```

---

## Axolotl Configuration

```yaml
base_model: giux78/zagreus-0.4B-ita 
strict: false
output_dir: ./ale_outputs/opendata-zagreus-sft-final
seed: 42
chat_template_jinja: "{%- for message in messages -%}\n    {{- \"<|im_start|>\" + message.role + \"\\n\" + message.content + \"<|im_end|>\" + \"\\n\" -}}\n{%- endfor -%}\n{%- if add_generation_prompt -%}\n\t{{- \"<|im_start|>assistant\\n\" -}}\n{%- endif -%}"
datasets:
  - path: /training/openitaliandata
    type: chat_template
    field_messages: conversation
    roles_to_train: ["assistant"]
    train_on_eos: turn

dataset_prepared_path: ./ale_outputs/dataset_cache/opendata-zagreus-sft

sequence_len: 4096
sample_packing: true
eval_sample_packing: true
pad_to_sequence_len: true

# Cosine schedule knobs
cosine_constant_lr_ratio: 0.8
cosine_min_lr_ratio: 0.3

optimizer: adamw_torch_fused
lr_scheduler: constant
learning_rate: 1.0e-03

max_grad_norm: 1.0
micro_batch_size: 1
gradient_accumulation_steps: 8

num_epochs: 3

bf16: auto
flash_attention: true
gradient_checkpointing: true

logging_steps: 10
eval_strategy: steps
eval_steps: 300
save_strategy: steps
save_steps: 500
save_total_limit: 3
val_set_size: 10000

fsdp_config:
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer
  fsdp_backward_prefetch_policy: BACKWARD_PRE
  fsdp_state_dict_type: FULL_STATE_DICT

special_tokens:
  pad_token: <|im_end|>
  eos_token: <|im_end|>
```

---

# 6. Pre-trained Foundational Models Evaluations

This section presents quantitative evaluations of our pre-trained foundational models. We include multiple data points to demonstrate how our data curation strategy and architectural configuration enabled the training of competitive small language model families.

These results serve both as validation and as a reproducible baseline for future experiments.

We are contributors to **lm-evaluation-harness** for multilingual benchmarks and relied extensively on this framework. For each benchmark, we provide the exact command used to ensure the evaluation reproducibility.

---

## Zagreus-0.4B-ita-base

### Evaluation Command

```bash
lm-eval --model hf --model_args pretrained=checkpoint \
  --tasks m_mmlu_it --num_fewshot 5 --device cuda:0 --batch_size 1

lm-eval --model hf --model_args pretrained=LiquidAI/LFM2-350M \
  --tasks hellaswag_it,arc_it --device cuda:0 --batch_size 1
```

Checkpoint progression:

| Checkpoint | mmlu_it (acc) | hellaswag_it (acc_norm) | arc_it (acc_norm) | Media  |
| ---------- | ------------- | ----------------------- | ----------------- | ------ |
| v2-95k     | 0.2529        | 0.3366                  | 0.2652            | 0.2849 |
| v2-205k    | 0.2628        | —                       | —                 | 0.2628 |
| v2-290k    | 0.2428        | 0.3492                  | 0.2335            | 0.2752 |
| v2-305k    | 0.2598        | 0.3562                  | 0.2652            | 0.2937 |
| v2-365k    | 0.2566        | 0.3664                  | 0.2712            | 0.2981 |
| v2-390k    | 0.2556        | 0.3438                  | 0.2498            | 0.2831 |
| v2-460k    | 0.2540        | 0.3778                  | 0.2549            | 0.2956 |
| v2-520k    | 0.2540        | 0.3778                  | 0.2549            | 0.2956 |
| v2-590k    | 0.2547        | 0.3651                  | 0.2455            | 0.2884 |
| v2-630k    | 0.2562        | 0.3632                  | 0.2643            | 0.2946 |
| v2-680k    | 0.2538        | 0.3740                  | 0.2592            | 0.2957 |
| v2-775k    | 0.2535        | 0.3750                  | 0.2583            | 0.2956 |

![zagreus-ita](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/images/zagreus-ita.png?raw=true)

---

## Zagreus-0.4B-spa-base (Spanish)

### Evaluation Command

```bash
lm-eval --model hf --model_args pretrained=checkpoint \
  --tasks m_mmlu_es --num_fewshot 5 --device cuda:0 --batch_size 1

lm-eval --model hf --model_args pretrained=LiquidAI/LFM2-350M \
  --tasks hellaswag_es,arc_es --device cuda:0 --batch_size 1
```

| Steps | mmlu_es | arc_es | hellaswag_es | Average |
| ----- | ------- | ------ | ------------ | ------- |
| 146k  | 0.254   | 0.265  | 0.409        | 0.309   |
| 216k  | 0.237   | 0.270  | 0.414        | 0.307   |
| 292k  | 0.254   | 0.262  | 0.417        | 0.311   |
| 406k  | 0.254   | 0.269  | 0.423        | 0.315   |
| 518k  | 0.255   | 0.280  | 0.429        | 0.321   |

---
![zagreus-spa](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/images/zagreus-spa.png?raw=true)

---
## Zagreus-0.4B-fra (French)

### Evaluation Command

```bash
lm-eval --model hf --model_args pretrained=checkpoint \
  --tasks m_mmlu_fr --num_fewshot 5 --device cuda:0 --batch_size 1

lm-eval --model hf --model_args pretrained=LiquidAI/LFM2-350M \
  --tasks hellaswag_fr,arc_fr --device cuda:0 --batch_size 1
```

Evaluation procedure identical to previous sections.

| Steps | m_mmlu_fr | arc_fr | hellaswag_fr | Average |
| ----- | --------- | ------ | ------------ | ------- |
| 129k  | 0.262     | —      | —            | 0.262   |
| 231k  | 0.263     | —      | —            | 0.263   |
| 365k  | 0.256     | 0.278  | 0.414        | 0.316   |
| 456k  | 0.267     | —      | —            | 0.267   |
| 603k  | 0.256     | 0.278  | 0.414        | 0.316   |
| 705k  | 0.266     | 0.281  | 0.417        | 0.321   |

---
![zagreus-fra](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/images/zagreus-fra.png?raw=true)

---
## Zagreus-0.4B-por (Portuguese)

### Evaluation Command

```bash
lm-eval --model hf --model_args pretrained=checkpoint \
  --tasks m_mmlu_pt --num_fewshot 5 --device cuda:0 --batch_size 1

lm-eval --model hf --model_args pretrained=LiquidAI/LFM2-350M \
  --tasks hellaswag_pt,arc_pt --device cuda:0 --batch_size 1
```

| Checkpoint | ARC    | HellaSwag | MMLU   | Media  |
| ---------- | ------ | --------- | ------ | ------ |
| 153k       | 0.2667 | 0.3732    | 0.2685 | 0.3028 |
| 207k       | 0.2705 | 0.3768    | 0.2671 | 0.3048 |
| 276k       | 0.2718 | 0.3789    | 0.2664 | 0.3057 |
| 345k       | 0.2564 | 0.3796    | 0.2669 | 0.3009 |
| 414k       | 0.2682 | 0.3842    | 0.2673 | 0.3066 |
| 483k       | 0.2667 | 0.3865    | 0.2658 | 0.3063 |
| 582k       | 0.2786 | 0.3865    | 0.2688 | 0.3113 |

---
---
![zagreus-por](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/images/zagreus-por.png?raw=true)

## lm-evaluation-harness-pt
For portuguese base model we also evaluate against the fantastic work of [Eduardo Garcia](https://github.com/eduagarcia) a [fork of lm-eval](https://github.com/eduagarcia/lm-evaluation-harness-pt) that has also an important [leaderboard](https://huggingface.co/spaces/eduagarcia/open_pt_llm_leaderboard) comparing many open source models. Below the results and the comparison with Qwen3-0.6B-Base.  

```bash
lm_eval     --model huggingface     --model_args "pretrained=giux78/zagreus-3B-165000,revision=main"     --tasks enem_challenge,bluex,oab_exams,assin2_rte,assin2_sts,faquad_nli,hatebr_offensive,portuguese_hate_speech,tweetsentbr     --device cuda:0     --output_path "./"
```

---
| Rank | Model / Checkpoint  | RTE    | STS    | BLUEX  | ENEM   | FAQUAD NLI | HateBR | OAB    | PT Hate | TweetSent | **Media**  |
| ---- | ------------------- | ------ | ------ | ------ | ------ | ---------- | ------ | ------ | ------- | --------- | ---------- |
| 🥇   | **zagreus 483k**    | 0.4624 | 0.1650 | 0.2434 | 0.2071 | 0.4397     | 0.3327 | 0.2528 | 0.4817  | 0.3220    | **0.3230** |
| 🥈   | **zagreus 582k**    | 0.3361 | 0.0449 | 0.2100 | 0.1903 | 0.4397     | 0.3825 | 0.2392 | 0.4444  | 0.1542    | **0.2713** |
| 🥉   | **Qwen3-0.6B-Base** | 0.3333 | 0.0726 | 0.1057 | 0.0077 | 0.4397     | 0.3333 | 0.0428 | 0.4123  | 0.5646    | **0.2569** |

---
![por-garcia](https://github.com/mii-llm/zagreus-nesso-slm/blob/main/images/por-garcia.png?raw=true)

---
# 7. Post-Trained Nesso Models Evaluations

In this section, we analyze the performance of **Nesso-0.4B-instruct** and **Nesso-0.4B-agentic** relative to comparable models. Since these models are pre-trained in English Italian we evaluate the models on english and italian benchmark as in the commands below.

```bash
lm-eval --model hf --model_args pretrained=checkpoint \
  --tasks m_mmlu_it --num_fewshot 5 --device cuda:0 --batch_size 1

lm-eval --model hf --model_args pretrained=checkpoint \
  --tasks mmlu --num_fewshot 5 --device cuda:0 --batch_size 1

lm-eval --model hf --model_args pretrained=LiquidAI/LFM2-350M \
  --tasks hellaswag_it,arc_it --device cuda:0 --batch_size 1

lm-eval --model hf --model_args pretrained=LiquidAI/LFM2-350M \
  --tasks hellaswag,arc --device cuda:0 --batch_size 1

lm-eval --model hf --model_args pretrained=LiquidAI/LFM2-350M \
  --tasks ifeval-ita --device cuda:0 --batch_size 1

lm-eval --model hf --model_args pretrained=LiquidAI/LFM2-350M \
  --tasks ifeval --device cuda:0 --batch_size 1
```

| Model               | IFEval EN | ARC_EN | HS_EN  | MMLU_EN | Media EN | IFEval IT | ARC_IT | HS_IT  | MMLU_IT | Media IT | Media Totale |
| ------------------- | --------- | ------ | ------ | ------- | -------- | --------- | ------ | ------ | ------- | -------- | ------------ |
| Qwen/Qwen3-0.6B     | 0.2758    | 0.3430 | 0.4742 | 0.4013  | 0.3736   | 0.3058    | 0.2729 | 0.3598 | 0.4025  | 0.3353   | 0.3545       |
| nesso-350M-sft-v0.6 | 0.3465    | 0.3003 | 0.4629 | 0.2871  | 0.3492   | 0.2962    | 0.2874 | 0.4076 | 0.2875  | 0.3197   | 0.3345       |
| nesso-350M-sft-v0.7 | 0.2962    | 0.2534 | 0.4062 | 0.2889  | 0.3112   | 0.2914    | 0.2541 | 0.3673 | 0.2730  | 0.2965   | 0.3039       |
| LiquidAI/LFM2-350M  | 0.1595    | 0.2457 | 0.3092 | 0.3445  | 0.2647   | 0.1427    | 0.2464 | 0.2994 | 0.3132  | 0.2504   | 0.2576       |

### Discussion

As observed, Qwen maintains a clear advantage on MMLU (both English and Italian). However, across several other benchmarks—particularly instruction-following and reasoning-oriented tasks—Nesso achieves competitive or superior performance.

Considering that MMLU is a widely used and often saturated benchmark, frequently incorporated into training corpora, we believe our results demonstrate that we have created a highly competitive small language model optimized for English/Italian edge inference scenarios.

---

## Open-Nesso-0.4B Evaluation

Open-Nesso-0.4B-ita is our fully open-source variant. It is based on Nesso-0.4B-ita and trained on the publicly available dataset published by Michele Montebovi.

Download:
[https://huggingface.co/datasets/DeepMount00/OpenItalianData](https://huggingface.co/datasets/DeepMount00/OpenItalianData)

| Model                        | mmlu_it | arc_it | hellaswag_it | Media  |
| ---------------------------- | ------- | ------ | ------------ | ------ |
| giux78/open-zagreus-350M-sft | 0.2530  | 0.3020 | 0.3608       | 0.3053 |

The model and dataset demonstrate that it is possible to build competitive English Italian language models using exclusively open-source resources.

---

# 8. Conclusion

We believe we have succeeded in creating a series of state-of-the-art small language models tailored for on-device inference. While drawing definitive conclusions in this rapidly evolving field is inherently risky, both the quantitative benchmarks and qualitative evaluation indicate that we have built models that are practically useful and scientifically robust.

We hope the community will rigorously test our models, reproduce our results, and provide constructive feedback to further advance sovereign and open AI development.

---
