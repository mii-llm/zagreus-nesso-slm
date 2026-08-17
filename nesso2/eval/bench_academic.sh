#!/bin/bash
source ~/ai/last-lm-eval/lm-eval/bin/activate
LME=~/ai/last-lm-eval/lm-eval/bin/lm-eval
M=giux78/zagreus-0.4B-cpt-full-32k-sft-agentic-v8
OUT=~/ai/bench_v8; mkdir -p $OUT
echo "### BENCH v8 $(date) $M ###"
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
