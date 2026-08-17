import json, os, re, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
JD=os.path.expanduser("~/ai/translate/judge")
MODEL="giux78/zagreus-0.4B-cpt-full-32k-sft-agentic-v8"
P=json.load(open(f"{JD}/conv_prompts.json"))
R=json.load(open(f"{JD}/conv_results.json"))
tok=AutoTokenizer.from_pretrained(MODEL, use_fast=True)
model=AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
_EOS=re.compile(r"<\|eot_id\|>|<\|im_end\|>|<\|end_of_text\|>|<\|endoftext\|>|</s>")
def chat(messages):
    enc=tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    enc={k:v.to("cuda") for k,v in enc.items()}; n=enc["input_ids"].shape[1]
    t0=time.time()
    with torch.no_grad():
        out=model.generate(**enc, max_new_tokens=524, do_sample=False,
                            repetition_penalty=1.15, no_repeat_ngram_size=6,
                            pad_token_id=tok.pad_token_id or tok.eos_token_id, eos_token_id=tok.eos_token_id)
    dt=time.time()-t0
    txt=tok.decode(out[0][n:], skip_special_tokens=False)
    return _EOS.split(txt)[0].replace("<|begin_of_text|>","").strip(), dt
for lang in ("IT","EN"):
    R[lang][MODEL]={}
    for conv, msgs in P[lang].items():
        a,dt=chat(msgs); R[lang][MODEL][conv]={"answer":a,"time":dt}
        print(f"{lang} {conv} ({dt:.1f}s)", flush=True)
json.dump(R, open(f"{JD}/conv_results.json","w"), ensure_ascii=False, indent=1)
print("CONV_GEN_V8_DONE ->", MODEL)
