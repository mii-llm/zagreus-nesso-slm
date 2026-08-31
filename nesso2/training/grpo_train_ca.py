"""Chat-anchored GRPO from Nesso2-0.4B-Agentic (v8) -> Nesso2-0.4B-Instruct.

obs inputs -> grounding reward (improve v8's observation handling), chat inputs
-> token-F1 to v8 (hold Italian conversation). Both types share the batch. Use a
LOW KL coefficient (beta ~0.005) so the policy can move; the CHAT REWARD, not the
KL, is what holds conversation to v8. This is the objective SFT cannot express.

Input (env DATA) is a JSONL where each row is one of:
  obs  : {"prompt": [<system>,<user>,<assistant tool_call>,<tool observation>],
          "type": "obs",  "obs_json": "<tool observation content>", "v8_response": ""}
  chat : {"prompt": [<user>],
          "type": "chat", "obs_json": "",  "v8_response": "<v8's own answer>"}
`prompt` is a conversational message list; the model generates the next turn.
The observation-grounding cases and chat prompts come from the family's private
agentic corpus (not released); this script + reward reproduce the *method*.
"""
import os, json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
import reward_chatanchor as RC

MODEL_PATH = os.environ["MODEL_PATH"]          # v8
DATA       = os.environ.get("DATA", "grpo_chatanchor.jsonl")
OUTPUT_DIR = os.environ["OUTPUT_DIR"]
MAX_STEPS  = int(os.environ.get("MAX_STEPS", "800"))
NUM_GEN    = int(os.environ.get("NUM_GEN", "8"))
PDBS       = int(os.environ.get("PDBS", "8"))
GRAD_ACC   = int(os.environ.get("GRAD_ACC", "8"))
LR         = float(os.environ.get("LR", "5e-6"))
BETA       = float(os.environ.get("BETA", "0.04"))   # KL-to-v8 anchor
TEMP       = float(os.environ.get("TEMP", "0.9"))
MAXCOMP    = int(os.environ.get("MAX_COMP_LEN", "320"))
USE_VLLM   = os.environ.get("USE_VLLM", "1") == "1"

rows = [json.loads(l) for l in open(DATA)]
ds = Dataset.from_list([{"prompt": r["prompt"], "type": r["type"],
                         "obs_json": r["obs_json"], "v8_response": r["v8_response"]} for r in rows])
print(f"[data] {len(ds)} (obs {sum(1 for r in rows if r['type']=='obs')} / chat {sum(1 for r in rows if r['type']=='chat')})", flush=True)

def _text(c):
    return "".join(m.get("content", "") for m in c) if isinstance(c, list) else c

def reward_ca(completions, obs_json, v8_response, **kw):
    types = kw["type"]
    return RC.reward_batch(types, obs_json, v8_response, [_text(c) for c in completions])

cfg = GRPOConfig(
    output_dir=OUTPUT_DIR, per_device_train_batch_size=PDBS, gradient_accumulation_steps=GRAD_ACC,
    num_generations=NUM_GEN, max_completion_length=MAXCOMP, temperature=TEMP,
    learning_rate=LR, beta=BETA, max_steps=MAX_STEPS, logging_steps=1,
    save_steps=MAX_STEPS, save_total_limit=1, bf16=True, gradient_checkpointing=True,
    log_completions=True, num_completions_to_print=2, report_to="none",
    use_vllm=USE_VLLM, **({"vllm_mode": "colocate", "vllm_gpu_memory_utilization": 0.3} if USE_VLLM else {}),
)
tok = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
trainer = GRPOTrainer(model=model, processing_class=tok, reward_funcs=reward_ca, args=cfg, train_dataset=ds)
print(f"[grpo-ca] start steps={MAX_STEPS} LR={LR} beta={BETA} from v8", flush=True)
trainer.train()
trainer.save_model(OUTPUT_DIR); tok.save_pretrained(OUTPUT_DIR)
print("GRPO_CA_DONE ->", OUTPUT_DIR, flush=True)
