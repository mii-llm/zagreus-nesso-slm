#!/usr/bin/env python3
"""LLM-as-judge over conv_model_comparison results, using a local vLLM
Qwen3.6-35B-A3B OpenAI server. Pointwise absolute scoring (1-10) on
correctness / language / helpfulness, aggregated per model and per language."""
import json, os, re, sys, time, requests
from concurrent.futures import ThreadPoolExecutor, as_completed

JD   = os.path.expanduser("~/ai/translate/judge")
BASE = os.environ.get("JUDGE_URL", "http://localhost:8100/v1")
MODEL= os.environ.get("JUDGE_MODEL", "Qwen/Qwen3.6-35B-A3B")
RESULTS = json.load(open(os.path.join(JD, "conv_results.json")))
PROMPTS = json.load(open(os.path.join(JD, "conv_prompts.json")))

SYS = ("You are a strict, impartial evaluator of AI assistant answers, fluent in "
       "Italian and English. You reward factual/mathematical/logical CORRECTNESS, "
       "penalize hallucinations (e.g. wrong capital city, wrong arithmetic), penalize "
       "answers written in the WRONG language relative to the user, and heavily penalize "
       "degenerate verbatim repetition loops. Judge only the LAST assistant answer.")

TMPL = """# Conversation (context)
{ctx}

# Final user request ({lang})
{req}

# Assistant answer to grade
{ans}

# Task
Rate the assistant answer on a 1-10 integer scale for each axis:
- "correctness": factual/math/logical accuracy of the content.
- "language": fluency AND that it is written in {lang} (the user's language); a repetition loop or wrong language scores 1-3.
- "helpfulness": does it actually satisfy the request, complete and well-formed.
Then give an "overall" 1-10 holistic score.
Respond with ONLY a JSON object, no prose:
{{"correctness": <int>, "language": <int>, "helpfulness": <int>, "overall": <int>, "reason": "<=15 words"}}"""

def transcript(msgs):
    out=[]
    for m in msgs[:-1]:
        out.append(f"[{m['role']}] {m['content']}")
    return "\n".join(out) if out else "(no prior context)"

def build(lang, conv, ans):
    msgs = PROMPTS[lang][conv]
    req  = next((m["content"] for m in reversed(msgs) if m["role"]=="user"), "?")
    return TMPL.format(ctx=transcript(msgs), lang=lang, req=req, ans=ans.strip() or "(empty)")

def call(payload):
    r = requests.post(f"{BASE}/chat/completions", json=payload, timeout=180)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

_JSON = re.compile(r"\{.*\}", re.DOTALL)
def parse(txt):
    m=_JSON.search(txt)
    if not m: return None
    try:
        d=json.loads(m.group(0))
        return {k:int(d.get(k,0)) for k in ("correctness","language","helpfulness","overall")} | {"reason":str(d.get("reason",""))[:120]}
    except Exception:
        return None

def judge_one(lang, model, conv, ans):
    payload=dict(model=MODEL, temperature=0, max_tokens=220,
                 messages=[{"role":"system","content":SYS},
                           {"role":"user","content":build(lang,conv,ans)}])
    for att in range(3):
        try:
            d=parse(call(payload))
            if d: return d
        except Exception as e:
            if att==2: return {"correctness":0,"language":0,"helpfulness":0,"overall":0,"reason":f"ERR {e}"[:80]}
            time.sleep(2)
    return {"correctness":0,"language":0,"helpfulness":0,"overall":0,"reason":"parse-fail"}

# wait for server
for _ in range(120):
    try:
        requests.get(f"{BASE}/models", timeout=5); break
    except Exception:
        time.sleep(5)
else:
    print("server never came up"); sys.exit(1)

jobs=[]
for lang, per in RESULTS.items():
    for model, convs in per.items():
        for conv, rec in convs.items():
            jobs.append((lang, model, conv, rec.get("answer","")))
print(f"scoring {len(jobs)} answers via {MODEL} ...", flush=True)

scores={}  # scores[model][lang][conv]=dict
t0=time.time()
with ThreadPoolExecutor(max_workers=24) as ex:
    futs={ex.submit(judge_one,l,m,c,a):(l,m,c) for (l,m,c,a) in jobs}
    done=0
    for f in as_completed(futs):
        l,m,c=futs[f]; scores.setdefault(m,{}).setdefault(l,{})[c]=f.result()
        done+=1
        if done%40==0: print(f"  {done}/{len(jobs)}  ({time.time()-t0:.0f}s)", flush=True)

json.dump(scores, open(os.path.join(JD,"judge_scores.json"),"w"), ensure_ascii=False, indent=1)

def mean(vals): return sum(vals)/len(vals) if vals else 0.0
models=list(RESULTS[list(RESULTS)[0]].keys())
def overall_for(m,lang):
    return [scores[m][lang][c]["overall"] for c in scores.get(m,{}).get(lang,{})]

print("\n"+"="*92)
print("LLM-AS-JUDGE (Qwen3.6-35B-A3B) — mean OVERALL /10, greedy answers")
print("="*92)
print(f"{'Model':<44}{'IT':>7}{'EN':>7}{'BOTH':>8}{'corr':>7}{'lang':>7}{'help':>7}")
print("-"*92)
rows=[]
for m in models:
    it=overall_for(m,'IT'); en=overall_for(m,'EN')
    both=it+en
    allc=[scores[m][l][c] for l in scores.get(m,{}) for c in scores[m][l]]
    corr=mean([x['correctness'] for x in allc]); langq=mean([x['language'] for x in allc]); helpc=mean([x['helpfulness'] for x in allc])
    rows.append((mean(both), m, mean(it), mean(en), mean(both), corr, langq, helpc))
for b,m,it,en,bo,corr,lq,hp in sorted(rows, reverse=True):
    print(f"{m.split('/')[-1]:<44}{it:>7.2f}{en:>7.2f}{bo:>8.2f}{corr:>7.2f}{lq:>7.2f}{hp:>7.2f}")
print("-"*92)
print("saved ->", os.path.join(JD,"judge_scores.json"))
