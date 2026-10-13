import re, sys
p = sys.argv[1]
s = open(p).read()
a = s.index("def encode_prompt(")
b = s.index("def extract_number")
new = '''def encode_prompt(tok, messages: list[dict[str, str]], chat_template_kwargs: dict | None = None, tools: list | None = None) -> list[int]:
    """Token ids of `messages` rendered with the tokenizer's chat template, ready for
    the assistant's turn (BOS prepended when the template leaves it out)."""
    kw = dict(chat_template_kwargs or {})
    if tools:
        kw["tools"] = tools
    text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **kw)
    ids = tok.encode(text, add_special_tokens=False)
    bos = tok.bos_token_id
    if bos is not None and (not ids or ids[0] != bos):
        ids = [bos] + ids
    return ids


'''
open(p, "w").write(s[:a] + new + s[b:])
print("patched", p)
