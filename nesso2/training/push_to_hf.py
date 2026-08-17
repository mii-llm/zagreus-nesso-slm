import json, os, shutil
from huggingface_hub import HfApi
MD="/scratch/project_465002563/training/outputs/sft-agentic-v8/20260816_135656"
TMPL="/scratch/project_465002563/training/zagreus_tools_template.jinja"
REPO="giux78/zagreus-0.4B-cpt-full-32k-sft-agentic-v8"
shutil.copy(TMPL, os.path.join(MD,"chat_template.jinja"))
tc=os.path.join(MD,"tokenizer_config.json"); d=json.load(open(tc)); d.pop("chat_template",None)
json.dump(d, open(tc,"w"), ensure_ascii=False, indent=2)
c=json.load(open(os.path.join(MD,"config.json"))); assert c.get("eos_token_id")==128009
api=HfApi(); api.create_repo(REPO, private=True, exist_ok=True, repo_type="model")
for a in range(5):
    try:
        api.upload_folder(folder_path=MD, repo_id=REPO, repo_type="model", commit_message="v8 final: balanced omnia>=0.6 + agentic-v8 (notool-rich abstention) + par_diff/multiarg/multistep)")
        print("PUSHED", REPO); break
    except Exception as e: print("retry",a,str(e)[:140])
print("PUSH_V8_DONE")
