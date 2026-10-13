#!/bin/bash
# lm-evaluation-harness: https://github.com/mii-llm/lm-evaluation-harness (mii-llm fork, adds the Italian tasks)
LME=${LME:-lm-eval}
M=${1:-mii-llm/nesso3-0.4B-agentic}
OUT=${2:-./bench_out}; mkdir -p $OUT
echo "### BENCH $(date) $M ###"
# Usage: ./bench_academic.sh <model> <out_dir>
# MMLU it+en, 5-shot AND 0-shot (acc)
for T in m_mmlu_it mmlu; do
  for NS in 5 0; do
    echo ">>> $T ${NS}sh"
    $LME --model hf --model_args pretrained="$M",dtype=bfloat16 --tasks "$T" \
       --num_fewshot $NS --device cuda:0 --batch_size 16 --output_path "$OUT/${T}_${NS}sh" > "$OUT/${T}_${NS}sh.log" 2>&1
  done
done
# HS/ARC it+en, 0-shot (acc_norm)
for T in hellaswag_it arc_it hellaswag arc_challenge; do
  echo ">>> $T 0sh"
  $LME --model hf --model_args pretrained="$M",dtype=bfloat16 --tasks "$T" \
     --num_fewshot 0 --device cuda:0 --batch_size 16 --output_path "$OUT/${T}" > "$OUT/${T}.log" 2>&1
done
# IFEval it+en, generative chat template
for T in ifeval ifeval-ita; do
  echo ">>> $T generative"
  $LME --model hf --model_args pretrained="$M",dtype=bfloat16 --tasks "$T" \
     --apply_chat_template --device cuda:0 --batch_size 16 --output_path "$OUT/${T}" > "$OUT/${T}.log" 2>&1
done
echo "BENCH_V2_DONE"
