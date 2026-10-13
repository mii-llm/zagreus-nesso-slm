#!/usr/bin/env python3
"""Generate conversation answers for the NEW candidates (compact + v8) with the
EXACT conv_model_comparison_v3 decoding (greedy, rp1.15, no_repeat_ngram=6,
single-BOS via apply_chat_template return_dict, decode only new tokens) and
MERGE into ~/ai/translate/judge/conv_results.json (anchors preserved)."""
import json, os, re, gc, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
DEVICE="cuda"; JD=os.environ.get("JUDGE_DIR",os.path.expanduser("~/ai/translate/judge"))
PROMPTS=json.load(open(f"{JD}/conv_prompts.json"))
RES_PATH=f"{JD}/conv_results.json"
RES=json.load(open(RES_PATH))
NEW=os.environ.get("MODELS","mii-llm/nesso3-0.4B-agentic").split(",")
MAXTOK=524; RP=1.15; NRNG=6
_EOS=re.compile(r"<\|eot_id\|>|<\|im_end\|>|<\|end_of_text\|>|<\|endoftext\|>|</s>")
_THINK=re.compile(r"<think>.*?</think>\s*",re.DOTALL)
def clean(t,mid):
    t=_EOS.split(t)[0]
    if "Qwen3" in mid:
        t=_THINK.sub("",t); t=t.split("<think>")[0]
    return t.strip()
def build_inputs(tok,msgs,mid):
    kw=dict(add_generation_prompt=True,return_tensors="pt",return_dict=True)
    if "Qwen3" in mid: kw["enable_thinking"]=False
    enc=tok.apply_chat_template(msgs,**kw)
    return {k:v.to(DEVICE) for k,v in enc.items()}
def gen(tok,m,msgs,mid):
    enc=build_inputs(tok,msgs,mid); ilen=enc["input_ids"].shape[1]
    kw=dict(max_new_tokens=MAXTOK,do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
            eos_token_id=tok.eos_token_id,repetition_penalty=RP,no_repeat_ngram_size=NRNG)
    with torch.no_grad(): out=m.generate(**enc,**kw)
    return clean(tok.decode(out[0][ilen:],skip_special_tokens=True),mid)

for mid in NEW:
    print("==== ",mid,flush=True)
    tok=AutoTokenizer.from_pretrained(mid,use_fast=True)
    m=AutoModelForCausalLM.from_pretrained(mid,torch_dtype=torch.bfloat16,device_map="auto").eval()
    for lang in ("IT","EN"):
        RES.setdefault(lang,{}).setdefault(mid,{})
        for task,msgs in PROMPTS[lang].items():
            a=gen(tok,m,msgs,mid); RES[lang][mid][task]={"answer":a}
            print(f"  {lang} {task:22s} {a[:60]!r}",flush=True)
    del m; gc.collect(); torch.cuda.empty_cache()

json.dump(RES,open(RES_PATH,"w"),ensure_ascii=False,indent=1)
print("MERGED ->",RES_PATH,"| models now:",list(RES["IT"].keys()))
print("GEN_CONV_DONE")
