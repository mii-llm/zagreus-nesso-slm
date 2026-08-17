from datasets import load_from_disk, Dataset
import os, random
random.seed(42)
SRC="/scratch/project_465002563/training/data/outputs/omnia_v6/hf_dataset_score_0_00/train"
OUT="/scratch/project_465002563/training/data/outputs/omnia_v6/hf_perlang_ITall_EN060"
ds=load_from_disk(SRC)
print("cols:", ds.column_names, "n:", len(ds))
assert "score" in ds.column_names, "no score column — need parquet rebuild"
ITW=set("che di il la per non sono come è una un dei delle gli mi ti ci vorrei puoi grazie perché così molto anche quando dove cosa questo essere fare più senza tra ho hai".split())
def is_it(msgs):
    t=" ".join(m.get("content","") for m in msgs[:3]).lower()
    w=[x.strip(".,!?;:()") for x in t.split()[:60]]
    return sum(1 for x in w if x in ITW)>=4
def keep(ex):
    it=is_it(ex["messages"])
    return it or (ex.get("score") is not None and float(ex["score"])>=0.6)
ds2=ds.filter(keep, num_proc=16)
# language tally
n_it=sum(1 for r in ds2.select(range(min(20000,len(ds2)))) if is_it(r["messages"]))
print(f"kept {len(ds2)}/{len(ds)}  (~{100*len(ds2)/len(ds):.0f}%)  IT~{100*n_it/min(20000,len(ds2)):.0f}% (sampled)")
ds2=ds2.shuffle(seed=42)
ev=ds2.select(range(10000)); tr=ds2.select(range(10000,len(ds2)))
os.makedirs(OUT,exist_ok=True)
tr.save_to_disk(OUT+"/train"); ev.save_to_disk(OUT+"/eval")
print("SAVED", OUT, "train", len(tr), "eval", len(ev))
