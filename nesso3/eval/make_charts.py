#!/usr/bin/env python3
"""Draw the two figures used in the Nesso3 README as dependency-free SVG.
All values are copied from the output of `summarize_results.py` (see ../results for the raw files).

    python3 nesso3/eval/make_charts.py     # writes ../images/*.svg
"""
import os
from xml.sax.saxutils import escape

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")
os.makedirs(OUT, exist_ok=True)
FONT = "font-family='-apple-system,Segoe UI,Helvetica,Arial,sans-serif'"
COL = {"base": "#8b95a5", "good": "#2e9e5b", "mixed": "#e0a030", "bad": "#d1493f", "release": "#2563eb", "ref": "#555555"}


def journey():
    rows = [
        ("compact SFT v1: one compact contract, 2,531 tools", 61, "base"),
        ("v2: procedural abstention data (over-abstains)", 60, "bad"),
        ("v3: fewer refusals + 'answer-me' data", 64, "good"),
        ("v4: name-grounding / crowded-disambiguation patches", 56, "bad"),
        ("v3 + chat-anchored RL", 68, "good"),
        ("  + extra RL rewards (par-diff, multi-step, missing)", 70, "mixed"),
        ("v5a: + public FC data incl. ToolMind", 49, "bad"),
        ("v5b: + public FC data, no ToolMind  =  NESSO3", 74, "release"),
        ("v6: v5b + Nemotron-SFT-Agentic-v2 slice", 68, "bad"),
        ("OPD-1 on v5b (gentle)", 72, "mixed"),
        ("OPD-2 on v5b (strong)", 75, "mixed"),
    ]
    W, LBL, X0, XW, RH, TOP = 900, 400, 410, 460, 30, 70
    H = TOP + RH * len(rows) + 70
    sx = lambda v: X0 + XW * v / 80.0
    s = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {W} {H}' width='{W}' {FONT}>",
         f"<rect width='{W}' height='{H}' fill='#ffffff'/>",
         f"<text x='20' y='28' font-size='18' font-weight='700' fill='#111'>100-case bilingual function-calling suite: every experiment on the road to Nesso3</text>",
         f"<text x='20' y='48' font-size='12' fill='#555'>total correct out of 100 (50 IT + 50 EN), greedy decoding, each model on its own native contract</text>"]
    for v in range(0, 81, 20):
        s.append(f"<line x1='{sx(v)}' y1='{TOP - 8}' x2='{sx(v)}' y2='{TOP + RH * len(rows)}' stroke='#e5e7eb'/>")
        s.append(f"<text x='{sx(v)}' y='{TOP + RH * len(rows) + 16}' font-size='11' fill='#777' text-anchor='middle'>{v}</text>")
    for i, (lab, v, k) in enumerate(rows):
        y = TOP + i * RH
        bold = " font-weight='700'" if k == "release" else ""
        s.append(f"<text x='{LBL}' y='{y + 19}' font-size='12.5' fill='#222' text-anchor='end'{bold}>{escape(lab)}</text>")
        s.append(f"<rect x='{X0}' y='{y + 5}' width='{sx(v) - X0:.1f}' height='{RH - 10}' rx='3' fill='{COL[k]}'/>")
        s.append(f"<text x='{sx(v) + 6:.1f}' y='{y + 19}' font-size='12.5' fill='#111' font-weight='700'>{v}</text>")
    for v, lab, dy in [(68, "Nesso2 v1 68", 0), (67, "Qwen3-0.6B 67", 14), (76, "Qwen3-0.6B thinking 76", 0)]:
        yb = TOP + RH * len(rows)
        s.append(f"<line x1='{sx(v)}' y1='{TOP - 8}' x2='{sx(v)}' y2='{yb}' stroke='{COL['ref']}' stroke-dasharray='4 3'/>")
    s.append(f"<text x='{sx(67.5)}' y='{TOP - 14}' font-size='10.5' fill='#555' text-anchor='end'>Nesso2 v1 68 · Qwen3-0.6B 67</text>")
    s.append(f"<text x='{sx(76) + 4}' y='{TOP - 14}' font-size='10.5' fill='#555'>Qwen thinking 76</text>")
    lx = 20
    for k, lab in [("release", "released (Nesso3)"), ("good", "worked, kept"), ("mixed", "mixed / not released"), ("bad", "regression"), ("base", "baseline")]:
        yb = H - 22
        s.append(f"<rect x='{lx}' y='{yb - 10}' width='12' height='12' rx='2' fill='{COL[k]}'/><text x='{lx + 18}' y='{yb}' font-size='12' fill='#333'>{lab}</text>")
        lx += 40 + len(lab) * 6.6
    s.append("</svg>")
    open(os.path.join(OUT, "journey_100case.svg"), "w").write("\n".join(s))


def compare():
    # (title, [(label, value)], max, fmt)
    panels = [
        ("100-case function calling (/100)", [("Nesso3", 74, "release"), ("Nesso2 v1", 68, "base"), ("Qwen3-0.6B", 67, "mixed")], 100, "{:.0f}"),
        ("Italian half of the 100 (/50)", [("Nesso3", 38, "release"), ("Nesso2 v1", 35, "base"), ("Qwen3-0.6B", 29, "mixed")], 50, "{:.0f}"),
        ("Held-out Italian healthcare (%)", [("Nesso3", 68.5, "release"), ("Nesso2 v1", 65.3, "base"), ("Qwen3-0.6B", 63.9, "mixed")], 100, "{:.1f}"),
        ("Robot benchmark, novel tools (%)", [("Nesso3", 63.6, "release"), ("Nesso2 v1", 53.3, "base"), ("Qwen3-0.6B", 71.1, "mixed")], 100, "{:.1f}"),
        ("Generic English tools (%)", [("Nesso3", 48, "release"), ("Qwen3-0.6B", 84, "mixed")], 100, "{:.0f}"),
        ("Academic average, IT+EN (x100)", [("Nesso3", 35.6, "release"), ("Nesso2 v1", 35.3, "base"), ("Qwen3-0.6B", 39.6, "mixed")], 50, "{:.1f}"),
        ("Chat, LLM judge, Italian (/10)", [("Nesso3", 3.65, "release"), ("Nesso2 v1", 4.35, "base"), ("Qwen3-0.6B", 2.73, "mixed")], 10, "{:.2f}"),
        ("Chat, LLM judge, English (/10)", [("Nesso3", 4.23, "release"), ("Nesso2 v1", 4.00, "base"), ("Qwen3-0.6B", 5.78, "mixed")], 10, "{:.2f}"),
    ]
    cols, PW, PH = 4, 215, 150
    W = cols * PW + 20
    rows_n = (len(panels) + cols - 1) // cols
    H = 70 + rows_n * PH + 30
    s = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {W} {H}' width='{W}' {FONT}>",
         f"<rect width='{W}' height='{H}' fill='#ffffff'/>",
         "<text x='20' y='28' font-size='18' font-weight='700' fill='#111'>Nesso3-0.4B-Agentic vs Nesso2 v1 vs Qwen3-0.6B</text>",
         "<text x='20' y='48' font-size='12' fill='#555'>higher is better in every panel; Qwen3-0.6B is ahead on 4 of the 8 (robot, generic tools, academics, English chat) and we publish those too</text>"]
    for i, (title, bars, mx, fmt) in enumerate(panels):
        px, py = 10 + (i % cols) * PW, 70 + (i // cols) * PH
        s.append(f"<text x='{px + 10}' y='{py + 14}' font-size='11.5' font-weight='700' fill='#222'>{escape(title)}</text>")
        bw = 40
        for j, (lab, v, k) in enumerate(bars):
            bx = px + 20 + j * (bw + 18)
            h = 85 * v / mx
            by = py + 28 + 85 - h
            s.append(f"<rect x='{bx}' y='{by:.1f}' width='{bw}' height='{h:.1f}' rx='3' fill='{COL[k]}'/>")
            s.append(f"<text x='{bx + bw / 2}' y='{by - 4:.1f}' font-size='11' font-weight='700' fill='#111' text-anchor='middle'>{fmt.format(v)}</text>")
            s.append(f"<text x='{bx + bw / 2}' y='{py + 28 + 85 + 14}' font-size='9.5' fill='#444' text-anchor='middle'>{lab}</text>")
    s.append("</svg>")
    open(os.path.join(OUT, "nesso3_vs_baselines.svg"), "w").write("\n".join(s))


if __name__ == "__main__":
    journey()
    compare()
    print("wrote", sorted(os.listdir(OUT)))
