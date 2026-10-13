#!/usr/bin/env python3
"""Rebuild every headline number of the Nesso3 report from the raw result files in ../results.

    python3 nesso3/eval/summarize_results.py            # prints markdown tables

Nothing in the README / model card is typed by hand without a counterpart here: the 100-case grades,
the robot / healthcare benchmark aggregates, the lm-eval academics and the LLM-judge runs are all read
from the JSON files that were produced by the evaluation scripts in this folder.
"""
import collections, glob, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results")
CASES = json.load(open(os.path.join(HERE, "..", "..", "nesso2", "eval", "agentic_eval_cases_100.json")))["cases"]
CAT = {c["id"]: (c["lang"], c["cat"]) for c in CASES}
CATS = ["single", "par_same", "par_diff", "multiarg", "disamb", "missing", "unavail", "notool", "obs", "multistep"]


def short(m):
    return m.split("/")[-1].replace("zagreus-0.4B-cpt-full-32k-", "")


def agentic():
    print("### 100-case bilingual function calling (graded from the raw model outputs)\n")
    print("| result file | model | total | IT | EN | " + " | ".join(CATS) + " |")
    print("|---|---|---|---|---|" + "---|" * len(CATS))
    for f in sorted(glob.glob(os.path.join(R, "agentic_100", "*.json"))):
        d = json.load(open(f))
        tag = os.path.basename(f).replace("agentic_eval_100_results_", "").replace(".json", "")
        for m, g in d["grades"].items():
            tot = sum(bool(v) for v in g.values())
            it = sum(bool(v) for k, v in g.items() if CAT[k][0] == "IT")
            pc = [sum(bool(v) for k, v in g.items() if CAT[k][1] == c) for c in CATS]
            print(f"| {tag} | {short(m)} | {tot} | {it} | {tot - it} | " + " | ".join(map(str, pc)) + " |")


def bench(kind):
    print(f"\n### {kind} benchmark (score %, per result directory)\n")
    print("| run | model | overall | function calling | FC IT | FC EN | conversation IT | conversation EN |")
    print("|---|---|---|---|---|---|---|---|")
    for d in sorted(glob.glob(os.path.join(R, kind, "*"))):
        r = json.load(open(os.path.join(d, "results.json")))
        for m, x in r["models"].items():
            a = x["aggregate"]
            g = lambda k: (f"{a[k]['score'] * 100:.1f}" if isinstance(a.get(k), dict) and "score" in a[k] else "-")
            print(f"| {os.path.basename(d)} | {short(m)} | {g('overall')} | {g('function_calling')} | {g('function_calling_it')} | "
                  f"{g('function_calling_en')} | {g('conversation_it')} | {g('conversation_en')} |")


ACAD = {"bench_v5b": "Nesso3-0.4B-Agentic (= SFT v5b)", "bench_v8": "Nesso2-0.4B-Agentic (v1, = SFT v8)", "bench_nesso_ag": "nesso-0.4B-agentic",
        "bench_qwen_same": "Qwen3-0.6B", "bench_opd": "OPD-1 (not released)", "bench_opd2": "OPD-2 (not released)"}
COLS = [("MMLU IT (5-shot)", "m_mmlu_it_5sh", "m_mmlu_it"), ("MMLU EN (5-shot)", "mmlu_5sh", "mmlu"),
        ("HellaSwag IT", "hellaswag_it", "hellaswag_it"), ("HellaSwag EN", "hellaswag", "hellaswag"),
        ("ARC IT", "arc_it", "arc_it"), ("ARC EN", "arc_challenge", "arc_challenge"),
        ("IFEval IT", "ifeval-ita", "ifeval-ita"), ("IFEval EN", "ifeval", "ifeval")]


def acad_value(d, task):
    r = d["results"][task]
    for k in ("acc_norm,none", "inst_level_loose_acc,none", "acc,none"):
        if k in r:
            return r[k]


def academics():
    print("\n### academics (lm-eval; MMLU 5-shot acc, HellaSwag/ARC 0-shot acc_norm, IFEval inst_level_loose_acc with chat template)\n")
    print("| model | " + " | ".join(c[0] for c in COLS) + " | avg IT | avg EN | overall | MMLU IT 0-shot | MMLU EN 0-shot |")
    print("|---|" + "---|" * (len(COLS) + 5))
    for b, label in ACAD.items():
        vals = {}
        for name, sub, task in COLS + [("m0", "m_mmlu_it_0sh", "m_mmlu_it"), ("e0", "mmlu_0sh", "mmlu")]:
            fs = glob.glob(os.path.join(R, "academics", b, sub, "*", "results_*.json"))
            vals[name] = acad_value(json.load(open(sorted(fs)[-1])), task) if fs else None
        it = [vals[c[0]] for c in COLS if c[0].endswith("IT") or "IT " in c[0]]
        en = [vals[c[0]] for c in COLS if c[0].endswith("EN") or "EN " in c[0]]
        f = lambda x: "-" if x is None else f"{x:.4f}"
        ait = sum(it) / 4 if None not in it else None
        aen = sum(en) / 4 if None not in en else None
        ov = (ait + aen) / 2 if ait is not None and aen is not None else None
        print(f"| {label} | " + " | ".join(f(vals[c[0]]) for c in COLS) + f" | {f(ait)} | {f(aen)} | {f(ov)} | {f(vals['m0'])} | {f(vals['e0'])} |")


JUDGE_MODELS = {
    "Nesso3-0.4B-Agentic (= SFT v5b)": "giux78/nesso2-0.4B-agentic-v2",
    "Nesso2-0.4B-Agentic (v1)": "giux78/zagreus-0.4B-cpt-full-32k-sft-agentic-v8",
    "nesso-0.4B-agentic": "mii-llm/nesso-0.4B-agentic",
    "Qwen3-0.6B": "Qwen/Qwen3-0.6B",
    "OPD-1 (not released)": "giux78/nesso2-0.4B-agentic-v2-opd",
    "OPD-2 (not released)": "giux78/nesso2-0.4B-agentic-v2-opd2",
    "chat-anchored RL v3 (not released)": "giux78/zagreus-0.4B-cpt-full-32k-grpo-chatanchor-v3",
    "compact SFT v3": "giux78/zagreus-0.4B-cpt-full-32k-sft-compact-v3",
    "nesso-3B (OPD teacher)": "giux78/nesso-3B",
}


def judge():
    files = sorted(glob.glob(os.path.join(R, "judge", "judge_run*.json")))
    runs = [json.load(open(f)) for f in files]
    print("\n### LLM-as-judge (Qwen3.6-35B-A3B): per run, mean 'overall' /10 (IT / EN / both)\n")
    print("| model | " + " | ".join(os.path.basename(f)[:-5] for f in files) + " |")
    print("|---|" + "---|" * len(files))
    for label, k in JUDGE_MODELS.items():
        cells = []
        for r in runs:
            if k not in r:
                cells.append("-")
                continue
            it = [x["overall"] for x in r[k]["IT"].values()]
            en = [x["overall"] for x in r[k]["EN"].values()]
            cells.append(f"{sum(it) / len(it):.2f} / {sum(en) / len(en):.2f} / {(sum(it) + sum(en)) / (len(it) + len(en)):.2f}")
        print(f"| {label} | " + " | ".join(cells) + " |")
    print(f"\n### LLM-as-judge, mean over the {len(runs)} retained runs\n")
    print("| model | runs | IT | EN | both | correctness | language | helpfulness |")
    print("|---|---|---|---|---|---|---|---|")
    for label, k in JUDGE_MODELS.items():
        sub = collections.defaultdict(list)
        its, ens = [], []
        for r in runs:
            if k not in r:
                continue
            it = [x["overall"] for x in r[k]["IT"].values()]
            en = [x["overall"] for x in r[k]["EN"].values()]
            its.append(sum(it) / len(it))
            ens.append(sum(en) / len(en))
            for x in list(r[k]["IT"].values()) + list(r[k]["EN"].values()):
                for a in ("correctness", "language", "helpfulness"):
                    sub[a].append(x[a])
        if not its:
            continue
        m = lambda l: sum(l) / len(l)
        print(f"| {label} | {len(its)} | {m(its):.2f} | {m(ens):.2f} | {(m(its) + m(ens)) / 2:.2f} | "
              f"{m(sub['correctness']):.2f} | {m(sub['language']):.2f} | {m(sub['helpfulness']):.2f} |")


if __name__ == "__main__":
    agentic()
    bench("robot")
    bench("healthcare")
    academics()
    judge()
