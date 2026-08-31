"""Chat-anchored RLVR reward — the last 0.4B card.
Two input types share the batch so RL optimizes obs improvement WHILE anchoring chat to v8:
  - type 'obs'  : reward grounding on the tool observation (no re-call, uses the obs value).
  - type 'chat' : reward staying v8-like (token-F1 to v8's self-distilled response).
Both in [0,1]. GRPO's per-prompt advantage normalizes across the group."""
import re, json

_TC = re.compile(r"<tool_call>", re.I)
_WORD = re.compile(r"[a-zàèéìòùA-Z0-9]+")

def _obs_values(obs_json):
    """Salient value strings from the tool observation JSON (recursive)."""
    try: o = json.loads(obs_json)
    except Exception: o = obs_json
    vals = []
    def walk(x):
        if isinstance(x, dict):
            for v in x.values(): walk(v)
        elif isinstance(x, list):
            for v in x: walk(v)
        elif isinstance(x, (str, int, float)):
            vals.append(str(x))
    walk(o)
    return vals

_STOP = set("the a an of to in on is are and or for with la le il di da un una che per".split())
def _content_tokens(s):
    return [w.lower() for w in _WORD.findall(s) if w.lower() not in _STOP and len(w) > 2]

def obs_reward(obs_json, completion):
    if _TC.search(completion):            # re-called a tool instead of grounding -> the failure mode
        return 0.0
    vals = _obs_values(obs_json)
    toks = set(_content_tokens(completion))
    hit = 0; tot = 0
    for v in vals:
        vt = _content_tokens(v)
        if not vt: continue
        tot += 1
        if sum(1 for t in vt if t in toks) >= max(1, len(vt)//2): hit += 1
    if tot == 0: return 0.5                # nothing checkable -> at least it didn't re-call
    return 0.3 + 0.7 * (hit / tot)         # grounded on the observation value(s)

def chat_reward(v8_response, completion):
    a = set(_content_tokens(v8_response)); b = set(_content_tokens(completion))
    if not a or not b: return 0.0
    inter = len(a & b)
    p = inter / len(b); r = inter / len(a)
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)   # token-F1 to v8

def reward_one(typ, obs_json, v8_response, completion):
    return obs_reward(obs_json, completion) if typ == "obs" else chat_reward(v8_response, completion)

def reward_batch(types, obs_jsons, v8_responses, completions):
    return [reward_one(t, o, v, c) for t, o, v, c in zip(types, obs_jsons, v8_responses, completions)]

if __name__ == "__main__":
    # self-test: grounded answer scores high, re-call scores 0, off-topic low
    oj = '{"headlines": ["the election results"]}'
    print("grounded", round(obs_reward(oj, "The top headline is about the election results."), 3))
    print("recall  ", round(obs_reward(oj, '<tool_call>\n{"name":"get_news"}\n</tool_call>'), 3))
    print("offtopic", round(obs_reward(oj, "I am not sure about that."), 3))
    print("chat-hi ", round(chat_reward("Java is object oriented and compiled.", "Java is an object oriented compiled language."), 3))
    print("chat-lo ", round(chat_reward("Java is object oriented and compiled.", "The weather is nice today."), 3))
    print("REWARD_CA_SELFTEST_DONE")
