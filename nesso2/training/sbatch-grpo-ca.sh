#!/bin/bash
# =============================================================================
# sbatch-grpo-ca.sh — chat-anchored RLVR: Nesso2-0.4B-Agentic (v8) -> Instruct
#
# GRPO from v8 with a two-part reward (reward_chatanchor.py): observation
# grounding on tool inputs + token-F1-to-v8 on chat inputs, in the same batch.
#
# KEY: use a LOW KL coefficient (BETA). A large KL-to-reference anchor caps all
# policy movement — KL plateaus at ~0.005 regardless of learning rate. Drop beta
# low and let the *chat reward* (not the KL) hold conversation to v8.
#
# Run (single GPU is enough for 0.4B; scale generations for throughput):
#   MODEL_PATH=<v8 checkpoint or HF id> \
#   DATA=grpo_chatanchor.jsonl \
#   OUTPUT_DIR=outputs/nesso2-instruct \
#   sbatch sbatch-grpo-ca.sh
# =============================================================================
#SBATCH --job-name=grpo-ca
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-grpo-ca-%j.out

set -euo pipefail

# ---- hyperparameters that produced Nesso2-0.4B-Instruct ----
export MODEL_PATH="${MODEL_PATH:?set to the v8 checkpoint or HF id}"
export DATA="${DATA:-grpo_chatanchor.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/nesso2-instruct}"
export LR="${LR:-1e-5}"
export BETA="${BETA:-0.005}"          # low KL anchor — the reward holds chat, not KL
export MAX_STEPS="${MAX_STEPS:-800}"
export NUM_GEN="${NUM_GEN:-8}"        # rollouts per prompt
export PDBS="${PDBS:-8}"              # per-device completions (multiple of NUM_GEN)
export GRAD_ACC="${GRAD_ACC:-8}"
export USE_VLLM="${USE_VLLM:-1}"      # vLLM colocate for rollout throughput
export TEMP="${TEMP:-0.9}"

# TRL >= 1.9 (GRPOTrainer) + vLLM. Reward is reward_chatanchor.py in this dir.
srun python grpo_train_ca.py
