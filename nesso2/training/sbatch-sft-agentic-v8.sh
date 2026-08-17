#!/bin/bash
# =============================================================================
# sbatch-sft-zagreus-04b.sh — full SFT of mii-llm/zagreus-0.4B-ita on LUMI-G
#
# Hardware : 1 node × 8 MI250x GCDs (64 GB each)
# Strategy : full fine-tune, plain DDP (accelerate_ddp_zagreus_04b.yaml) —
#            0.44B replicated is ~5 GB/GCD, FSDP would only slow it down.
# Data     : omnia_v6 filtered at score >= 0.6 (sft/prepare_dataset.py)
#
# Batch: 4 (per-device) × 4 (accum) × 8 (GCDs) = 128 seqs ≈ 512K tokens/step
# LR 1e-3, cosine_with_min_lr (floor 0.3), 3 epochs.
#
# WHY 1e-3 (v2): the first run used LR 3e-5 and the eval loss went flat at ~1.98
# from step 1000 — a 0.4B base barely moved, and it never learned to emit
# <|eot_id|>. mii-llm/nesso-0.4B-agentic (SAME base model) SFTs at LR 1e-3,
# 3 epochs, cosine held high (constant 0.8 / min 0.3). Tiny dense models need
# aggressive LRs to adapt from a raw base checkpoint. min_lr_rate 0.3 keeps LR
# high instead of decaying to 0. Watch early steps for divergence/NaN at 1e-3;
# back off to 5e-4 (--export=ALL,LR=5e-4) if unstable.
#
# NOTE on batch size: per-device batch is small NOT because of model weights
# (0.44B is tiny) but because the LM-head logits dominate memory. With this
# model's 128K-token vocab, cross-entropy materializes a [B*4096, 128256] fp32
# tensor — batch 16 needs ~50 GB there alone and OOMs. batch 4 + gradient
# checkpointing keeps it comfortable. (Same effective batch via accum.)
#
# Chat format: the model is a BASE model (no chat template). We train with the
# Llama-3 header format using special tokens already in its Llama-3.2
# tokenizer (<|start_header_id|>, <|eot_id|>), pad with
# <|finetune_right_pad_id|>, and register <|eot_id|> as a generation stop
# token. The saved tokenizer gets the inference template (no {% generation %}).
# max_seq_length must stay 4096 (model max_position_embeddings).
#
# Pre-download on a login node:
#   hf download mii-llm/zagreus-0.4B-ita --local-dir /scratch/project_465002563/training/hf_checkpoints/zagreus-350M-cpt-full-32k
#
# Smoke test first:
#   sbatch --export=ALL,MAX_SAMPLES=2000 sft/sbatch-sft-zagreus-04b.sh
# Resume:
#   sbatch --export=ALL,RESUME_OUTPUT_DIR=<existing_dir> sft/sbatch-sft-zagreus-04b.sh
# =============================================================================

#SBATCH --account=project_465002563
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-sft-agentic-v3-%j.out
#SBATCH --exclusive

# ============================================================
LUMI_PROJECT="project_465002563"
BASE="/scratch/${LUMI_PROJECT}/training"
SFT_DIR="${BASE}/sft"

MODEL_PATH="${BASE}/hf_checkpoints/zagreus-350M-cpt-full-32k"
DATASET_PATH="${BASE}/data/outputs/omnia_v6/hf_perlang_ITall_EN060/train ${BASE}/data/outputs/omnia_agentic_v8/train ${BASE}/data/outputs/omnia_agentic_v8/train ${BASE}/data/outputs/omnia_agentic_v8/train ${BASE}/data/outputs/omnia_agentic_v8/train"
EVAL_DATASET_PATH="${BASE}/data/outputs/omnia_v6/hf_perlang_ITall_EN060/eval"

if [[ -n "${RESUME_OUTPUT_DIR:-}" ]]; then
    [[ -d "${RESUME_OUTPUT_DIR}" ]] || { echo "ERROR: RESUME_OUTPUT_DIR=${RESUME_OUTPUT_DIR} does not exist." >&2; exit 1; }
    OUTPUT_DIR="${RESUME_OUTPUT_DIR}"
    echo ">>> Will resume from existing run: ${OUTPUT_DIR}"
else
    OUTPUT_DIR="${BASE}/outputs/sft-agentic-v8/$(date +%Y%m%d_%H%M%S)"
    echo ">>> Starting fresh run: ${OUTPUT_DIR}"
fi

SIF="/appl/local/laifs/containers/lumi-multitorch-latest.sif"
SQSH="${BASE}/trl-env.sqsh"
# ============================================================

################ 0. Environment ################
module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${BASE}/miopen_cache/cache" "${BASE}/miopen_cache/db" "${BASE}/triton_cache"
export SINGULARITYENV_PREPEND_PATH=/user-software/bin

################ 1. Distributed rendezvous ################
export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MASTER_PORT="2${SLURM_JOB_ID: -4}"
mkdir -p "${OUTPUT_DIR}"

################ 2. Job env file ################
JOB_ENV_FILE="${BASE}/sft_job_${SLURM_JOB_ID}.env"
cat > "${JOB_ENV_FILE}" << EOF
BASE="${BASE}"
ACCEL_CONFIG="${SFT_DIR}/accelerate_ddp_zagreus_04b.yaml"
TRAIN_SCRIPT="${SFT_DIR}/sft_train.py"
MODEL_PATH="${MODEL_PATH}"
DATASET_PATH="${DATASET_PATH}"
EVAL_DATASET_PATH="${EVAL_DATASET_PATH}"
OUTPUT_DIR="${OUTPUT_DIR}"
CHAT_TEMPLATE="${SFT_DIR}/chat_templates/llama3_training.jinja"
MESSAGES_COL="messages"
MASTER_ADDR="${MASTER_ADDR}"
MASTER_PORT="${MASTER_PORT}"
MIOPEN_CACHE="${BASE}/miopen_cache"
NUM_PROCESSES=8
MAX_SEQ_LENGTH=8192
NUM_EPOCHS=3
BATCH_SIZE=2
GRAD_ACCUM=8
LR=${LR:-1e-3}
LR_SCHEDULER=${LR_SCHEDULER:-cosine_with_min_lr}
MIN_LR_RATE=${MIN_LR_RATE:-0.3}
WARMUP_RATIO=0.03
SAVE_STEPS=${SAVE_STEPS:-1000}
EVAL_STEPS=${EVAL_STEPS:-}          # empty -> eval every SAVE_STEPS; override via --export
EVAL_SAMPLES=${EVAL_SAMPLES:-10000} # held out from training
RUN_NAME="sft-agentic-v8-${SLURM_JOB_ID}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
# Base-model chat wiring: eot as stop token, dedicated pad, inference template
# saved into the output tokenizer. Gradient checkpointing ON — the large-vocab
# logits make activation memory the constraint, not the tiny model.
EXTRA_ARGS="--gradient_checkpointing --eos_token <|eot_id|> --pad_token <|finetune_right_pad_id|> --inference_chat_template_path ${SFT_DIR}/chat_templates/llama3_inference.jinja"
EOF

################ 3. Launch ################
export JOB_ENV_FILE

srun singularity exec \
    -B "${SQSH}:/user-software:image-src=/" \
    -B "${BASE}:${BASE}" \
    "${SIF}" \
    bash "${SFT_DIR}/run_sft.sh"
