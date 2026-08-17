#!/usr/bin/env python3
# FROZEN agentic eval v1 — 40 cases, IT+EN, 10 categories. zagreus-dpo vs Qwen3-0.6B.
# Dumps the frozen case set to agentic_eval_cases_v2_100.json (decontaminate training data against it).
import torch, json, os, re, gc
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.manual_seed(0)
OUT=os.path.expanduser("~/ai/agentic_eval"); os.makedirs(OUT,exist_ok=True); DEVICE="cuda"
MODELS=["giux78/zagreus-0.4B-cpt-full-32k-sft-agentic-v3",
        "giux78/zagreus-0.4B-cpt-full-32k-sft-agentic-v6",
        "giux78/zagreus-0.4B-cpt-full-32k-sft-agentic-v6-1",
        "giux78/zagreus-0.4B-cpt-full-32k-sft-agentic-v7",
        "giux78/zagreus-0.4B-cpt-full-32k-sft-agentic-v8",
        "Qwen/Qwen3-0.6B"]

TOOLS=[
 {"type":"function","function":{"name":"get_weather","description":"Weather forecast for a city on a given day.","parameters":{"type":"object","properties":{"city":{"type":"string"},"day":{"type":"string"}},"required":["city"]}}},
 {"type":"function","function":{"name":"calculate","description":"Evaluate a math expression and return the number.","parameters":{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}}},
 {"type":"function","function":{"name":"convert_currency","description":"Convert an amount from one currency to another.","parameters":{"type":"object","properties":{"amount":{"type":"number"},"from":{"type":"string"},"to":{"type":"string"}},"required":["amount","from","to"]}}},
 {"type":"function","function":{"name":"book_flight","description":"Book a flight.","parameters":{"type":"object","properties":{"origin":{"type":"string"},"destination":{"type":"string"},"date":{"type":"string"},"passengers":{"type":"integer"},"travel_class":{"type":"string"}},"required":["origin","destination","date","passengers"]}}},
 {"type":"function","function":{"name":"send_email","description":"Send an email. Requires a recipient address.","parameters":{"type":"object","properties":{"to":{"type":"string"},"subject":{"type":"string"},"body":{"type":"string"}},"required":["to","body"]}}},
 {"type":"function","function":{"name":"web_search","description":"Search the web for up-to-date information.","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}},
 {"type":"function","function":{"name":"create_event","description":"Create a calendar event.","parameters":{"type":"object","properties":{"title":{"type":"string"},"date":{"type":"string"},"time":{"type":"string"}},"required":["title","date"]}}},
 {"type":"function","function":{"name":"get_stock_price","description":"Get the current stock price for a ticker symbol.","parameters":{"type":"object","properties":{"symbol":{"type":"string"}},"required":["symbol"]}}},
 {"type":"function","function":{"name":"translate_text","description":"Translate text into a target language.","parameters":{"type":"object","properties":{"text":{"type":"string"},"target_language":{"type":"string"}},"required":["text","target_language"]}}},
]
def U(t): return [{"role":"user","content":t}]
def AC(name,args): return {"role":"assistant","content":'<tool_call>\n{"name": "%s", "arguments": %s}\n</tool_call>'%(name,json.dumps(args,ensure_ascii=False))}
def TO(obj): return {"role":"tool","content":json.dumps(obj,ensure_ascii=False)}

CASES=[
 # --- single-call (anchor) ---
 ("s1","IT","single",U("Che tempo fa a Torino oggi?"),{"tools":{"get_weather"},"n":1}),
 ("s2","EN","single",U("Search the web for today's top news."),{"tools":{"web_search"},"n":1}),
 ("s3","IT","single",U("Aggiungi 'Riunione team' in calendario il 3 giugno alle 10."),{"tools":{"create_event"},"n":1}),
 ("s4","EN","single",U("Convert 50 pounds to euros."),{"tools":{"convert_currency"},"n":1}),
 # --- parallel same tool ---
 ("ps1","IT","par_same",U("Confronta il meteo di domani a Roma e Firenze."),{"tools":{"get_weather"},"n":2}),
 ("ps2","EN","par_same",U("Compare tomorrow's weather in Paris and Berlin."),{"tools":{"get_weather"},"n":2}),
 ("ps3","IT","par_same",U("Dimmi il prezzo delle azioni di Apple e Microsoft."),{"tools":{"get_stock_price"},"n":2}),
 ("ps4","EN","par_same",U("Get the stock price of Tesla and Amazon."),{"tools":{"get_stock_price"},"n":2}),
 # --- parallel different tools ---
 ("pd1","IT","par_diff",U("Cerca notizie sull'inflazione e convertimi 1000 euro in dollari."),{"tools":{"web_search","convert_currency"},"n":2}),
 ("pd2","EN","par_diff",U("Search news about the elections and calculate 15% of 200."),{"tools":{"web_search","calculate"},"n":2}),
 ("pd3","IT","par_diff",U("Che tempo fa a Roma e quanto fa 12 per 15?"),{"tools":{"get_weather","calculate"},"n":2}),
 ("pd4","EN","par_diff",U("What's Apple's stock price and the weather in Cupertino?"),{"tools":{"get_stock_price","get_weather"},"n":2}),
 # --- multi-arg + typed ---
 ("ma1","IT","multiarg",U("Prenota un volo da Milano a Tokyo il 20 maggio 2025 per 3 persone in business."),{"tools":{"book_flight"},"n":1,"argkeys":("book_flight",["origin","destination","date","passengers"]),"intarg":("book_flight","passengers",3)}),
 ("ma2","EN","multiarg",U("Book a flight from London to New York on 2025-07-01 for 1 person in first class."),{"tools":{"book_flight"},"n":1,"argkeys":("book_flight",["origin","destination","date","passengers"]),"intarg":("book_flight","passengers",1)}),
 ("ma3","IT","multiarg",U("Invia una email a mario@rossi.it con oggetto 'Preventivo' e corpo 'In allegato il preventivo richiesto'."),{"tools":{"send_email"},"n":1,"argkeys":("send_email",["to","body"])}),
 ("ma4","EN","multiarg",U("Create an event 'Dentist' on 2025-03-12 at 09:30."),{"tools":{"create_event"},"n":1,"argkeys":("create_event",["title","date"])}),
 # --- disambiguation ---
 ("d1","IT","disamb",U("Quanto valgono 250 dollari in euro?"),{"tools":{"convert_currency"},"n":1}),
 ("d2","EN","disamb",U("What is 250 USD in euros?"),{"tools":{"convert_currency"},"n":1}),
 ("d3","IT","disamb",U("Quanto fa il 18% di 340?"),{"tools":{"calculate"},"n":1}),
 ("d4","EN","disamb",U("Translate 'good morning' into Italian."),{"tools":{"translate_text"},"n":1}),
 # --- missing required arg -> should ask, not fabricate ---
 ("mi1","IT","missing",U("Invia una email per confermare la riunione di venerdì."),{"tool":"send_email","fabricate_field":"to"}),
 ("mi2","EN","missing",U("Send an email to confirm Friday's meeting."),{"tool":"send_email","fabricate_field":"to"}),
 ("mi3","IT","missing",U("Prenota un volo per Parigi."),{"tool":"book_flight","fabricate_field":"origin"}),
 ("mi4","EN","missing",U("Book a flight to Paris."),{"tool":"book_flight","fabricate_field":"origin"}),
 # --- unavailable tool -> abstain (no matching tool) ---
 ("u1","IT","unavail",U("Prenotami un taxi per domani alle 8.")," "),
 ("u2","EN","unavail",U("Order me a pizza for tonight.")," "),
 ("u3","IT","unavail",U("Accendi le luci del soggiorno.")," "),
 ("u4","EN","unavail",U("Play some jazz music.")," "),
 # --- no tool needed (knowledge / chit-chat) ---
 ("k1","IT","notool",U("Chi ha scritto i Promessi Sposi?"),{"gold":"manzoni"}),
 ("k2","EN","notool",U("Who wrote Romeo and Juliet?"),{"gold":"shakespeare"}),
 ("k3","IT","notool",U("Ciao, come stai oggi?"),{"gold":None}),
 ("k4","EN","notool",U("Explain in one sentence what a black hole is."),{"gold":None}),
 # --- observation follow-up (single step): answer grounded, no re-call ---
 ("o1","IT","obs",[*U("Che tempo fa a Roma domani? Devo decidere se portare l'ombrello."),AC("get_weather",{"city":"Roma","day":"domani"}),TO({"city":"Roma","temp":"28°C","condition":"sereno","rain_prob":"5%"})],{"want":"no umbrella / sereno"}),
 ("o2","EN","obs",[*U("What's the weather in London tomorrow? Should I bring an umbrella?"),AC("get_weather",{"city":"London","day":"tomorrow"}),TO({"city":"London","temp":"12°C","condition":"rain","rain_prob":"85%"})],{"want":"bring umbrella / rain"}),
 ("o3","IT","obs",[*U("Qual è il prezzo delle azioni Apple?"),AC("get_stock_price",{"symbol":"AAPL"}),TO({"symbol":"AAPL","price":190,"currency":"USD"})],{"want":"190 USD","gold":"190"}),
 ("o4","EN","obs",[*U("Convert 100 euros to dollars."),AC("convert_currency",{"amount":100,"from":"EUR","to":"USD"}),TO({"result":108,"currency":"USD"})],{"want":"108","gold":"108"}),
 # --- multi-step dependent: emit the correct SECOND call (or restrain) ---
 ("m1","IT","multistep",[*U("Trova il prezzo delle azioni Apple e poi convertilo in euro."),AC("get_stock_price",{"symbol":"AAPL"}),TO({"symbol":"AAPL","price":190,"currency":"USD"})],{"next":"convert_currency"}),
 ("m2","EN","multistep",[*U("Find the date of the next Summer Olympics, then add it to my calendar."),AC("web_search",{"query":"next Summer Olympics date"}),TO({"result":"Los Angeles 2028, opening 2028-07-14"})],{"next":"create_event"}),
 ("m3","IT","multistep",[*U("Controlla il meteo di domani a Roma; se piove mandami una mail a me@ex.it per ricordarmi l'ombrello."),AC("get_weather",{"city":"Roma","day":"domani"}),TO({"city":"Roma","condition":"pioggia","rain_prob":"90%"})],{"next":"send_email"}),
 ("m4","EN","multistep",[*U("Check tomorrow's weather in Rome; if it rains, email me@ex.it a reminder."),AC("get_weather",{"city":"Rome","day":"tomorrow"}),TO({"city":"Rome","condition":"sunny","rain_prob":"5%"})],{"next":None}),  # sunny -> should NOT email
 # ===== EXPANSION: +60 cases (6 per category) -> 100 total =====
 ("s5","IT","single",U("Qual è il prezzo delle azioni di Ferrari?"),{"tools":{"get_stock_price"},"n":1}),
 ("s6","EN","single",U("What's the weather in Dublin on Friday?"),{"tools":{"get_weather"},"n":1}),
 ("s7","IT","single",U("Traduci 'grazie mille' in tedesco."),{"tools":{"translate_text"},"n":1}),
 ("s8","EN","single",U("How much is 300 dollars in pounds?"),{"tools":{"convert_currency"},"n":1}),
 ("s9","IT","single",U("Quanto fa 47 più 88?"),{"tools":{"calculate"},"n":1}),
 ("s10","EN","single",U("Add 'Gym' to my calendar on 2025-04-10 at 18:00."),{"tools":{"create_event"},"n":1}),
 ("ps5","IT","par_same",U("Confronta il prezzo delle azioni di Google e Amazon."),{"tools":{"get_stock_price"},"n":2}),
 ("ps6","EN","par_same",U("Compare tomorrow's weather in Madrid and Lisbon."),{"tools":{"get_weather"},"n":2}),
 ("ps7","IT","par_same",U("Che tempo fa oggi a Napoli e a Bari?"),{"tools":{"get_weather"},"n":2}),
 ("ps8","EN","par_same",U("Get the stock price of Nvidia and Meta."),{"tools":{"get_stock_price"},"n":2}),
 ("ps9","IT","par_same",U("Converti 100 euro in dollari e 100 euro in sterline."),{"tools":{"convert_currency"},"n":2}),
 ("ps10","EN","par_same",U("Convert 50 USD to EUR and 50 USD to GBP."),{"tools":{"convert_currency"},"n":2}),
 ("pd5","IT","par_diff",U("Cerca notizie sul clima e dimmi il prezzo delle azioni Tesla."),{"tools":{"web_search","get_stock_price"},"n":2}),
 ("pd6","EN","par_diff",U("Translate 'hello' into French and tell me the weather in Nice."),{"tools":{"translate_text","get_weather"},"n":2}),
 ("pd7","IT","par_diff",U("Quanto fa 200 diviso 8 e che tempo fa a Venezia?"),{"tools":{"calculate","get_weather"},"n":2}),
 ("pd8","EN","par_diff",U("Convert 500 euros to dollars and search news about AI."),{"tools":{"convert_currency","web_search"},"n":2}),
 ("pd9","IT","par_diff",U("Dimmi il prezzo delle azioni Microsoft e traduci 'buonasera' in inglese."),{"tools":{"get_stock_price","translate_text"},"n":2}),
 ("pd10","EN","par_diff",U("What's 25% of 80 and what's the weather in Oslo?"),{"tools":{"calculate","get_weather"},"n":2}),
 ("ma5","IT","multiarg",U("Prenota un volo da Roma a Berlino il 2025-09-10 per 2 persone in economy."),{"tools":{"book_flight"},"n":1,"argkeys":("book_flight",["origin","destination","date","passengers"]),"intarg":("book_flight","passengers",2)}),
 ("ma6","EN","multiarg",U("Book a flight from Paris to Madrid on 2025-08-15 for 4 passengers in economy."),{"tools":{"book_flight"},"n":1,"argkeys":("book_flight",["origin","destination","date","passengers"]),"intarg":("book_flight","passengers",4)}),
 ("ma7","IT","multiarg",U("Invia una email a laura@ex.it con oggetto 'Riunione' e corpo 'Confermo la riunione di lunedì'."),{"tools":{"send_email"},"n":1,"argkeys":("send_email",["to","body"])}),
 ("ma8","EN","multiarg",U("Send an email to john@ex.com with subject 'Report' and body 'The report is attached'."),{"tools":{"send_email"},"n":1,"argkeys":("send_email",["to","body"])}),
 ("ma9","IT","multiarg",U("Crea un evento 'Compleanno' il 2025-06-20 alle 20:00."),{"tools":{"create_event"},"n":1,"argkeys":("create_event",["title","date"])}),
 ("ma10","EN","multiarg",U("Create an event 'Standup' on 2025-05-05 at 09:00."),{"tools":{"create_event"},"n":1,"argkeys":("create_event",["title","date"])}),
 ("d5","IT","disamb",U("Quanto fa il 30% di 90?"),{"tools":{"calculate"},"n":1}),
 ("d6","EN","disamb",U("Convert 75 pounds to dollars."),{"tools":{"convert_currency"},"n":1}),
 ("d7","IT","disamb",U("Traduci 'arrivederci' in spagnolo."),{"tools":{"translate_text"},"n":1}),
 ("d8","EN","disamb",U("What is 144 divided by 12?"),{"tools":{"calculate"},"n":1}),
 ("d9","IT","disamb",U("Qual è il prezzo delle azioni Netflix?"),{"tools":{"get_stock_price"},"n":1}),
 ("d10","EN","disamb",U("What's the weather in Stockholm today?"),{"tools":{"get_weather"},"n":1}),
 ("mi5","IT","missing",U("Invia una email per confermare l'appuntamento di domani."),{"tool":"send_email","fabricate_field":"to"}),
 ("mi6","EN","missing",U("Send an email to reschedule tomorrow's call."),{"tool":"send_email","fabricate_field":"to"}),
 ("mi7","IT","missing",U("Prenotami un volo per Amsterdam la prossima settimana."),{"tool":"book_flight","fabricate_field":"origin"}),
 ("mi8","EN","missing",U("Book a flight to Tokyo next month."),{"tool":"book_flight","fabricate_field":"origin"}),
 ("mi9","IT","missing",U("Manda una mail per avvisare che sarò in ritardo."),{"tool":"send_email","fabricate_field":"to"}),
 ("mi10","EN","missing",U("Book a flight to Lisbon on 2025-10-01."),{"tool":"book_flight","fabricate_field":"origin"}),
 ("u5","IT","unavail",U("Chiamami un taxi per l'aeroporto.")," "),
 ("u6","EN","unavail",U("Turn off the bedroom lights.")," "),
 ("u7","IT","unavail",U("Ordinami una pizza margherita.")," "),
 ("u8","EN","unavail",U("Set an alarm for 7am.")," "),
 ("u9","IT","unavail",U("Riproduci la mia playlist rock.")," "),
 ("u10","EN","unavail",U("Call my mother.")," "),
 ("k5","IT","notool",U("Chi ha scritto la Divina Commedia?"),{"gold":"dante"}),
 ("k6","EN","notool",U("Who developed the theory of relativity?"),{"gold":"einstein"}),
 ("k7","IT","notool",U("Qual è il contrario di 'grande'?"),{"gold":"piccolo"}),
 ("k8","EN","notool",U("What's the opposite of 'hot'?"),{"gold":"cold"}),
 ("k9","IT","notool",U("Spiega in una frase cos'è la fotosintesi."),{"gold":None}),
 ("k10","EN","notool",U("Say hello and wish me a good day."),{"gold":None}),
 ("o5","IT","obs",[*U("Converti 200 euro in dollari."),AC("convert_currency",{"amount":200,"from":"EUR","to":"USD"}),TO({"result":216,"currency":"USD"})],{"gold":"216"}),
 ("o6","EN","obs",[*U("What's Microsoft's stock price?"),AC("get_stock_price",{"symbol":"MSFT"}),TO({"symbol":"MSFT","price":420,"currency":"USD"})],{"gold":"420"}),
 ("o7","IT","obs",[*U("Che tempo fa a Palermo domani?"),AC("get_weather",{"city":"Palermo","day":"domani"}),TO({"city":"Palermo","temp":"30°C","condition":"soleggiato"})],{"gold":None}),
 ("o8","EN","obs",[*U("What's the weather in Berlin today?"),AC("get_weather",{"city":"Berlin","day":"today"}),TO({"city":"Berlin","temp":"8°C","condition":"snow"})],{"gold":None}),
 ("o9","IT","obs",[*U("Quanto fa 45 per 6?"),AC("calculate",{"expression":"45*6"}),TO({"result":270})],{"gold":"270"}),
 ("o10","EN","obs",[*U("Convert 100 GBP to EUR."),AC("convert_currency",{"amount":100,"from":"GBP","to":"EUR"}),TO({"result":117,"currency":"EUR"})],{"gold":"117"}),
 ("m5","IT","multistep",[*U("Trova il prezzo delle azioni Tesla e convertilo in euro."),AC("get_stock_price",{"symbol":"TSLA"}),TO({"symbol":"TSLA","price":250,"currency":"USD"})],{"next":"convert_currency"}),
 ("m6","EN","multistep",[*U("Find when the next football World Cup is, then add it to my calendar."),AC("web_search",{"query":"next football World Cup date"}),TO({"result":"2026-06-11"})],{"next":"create_event"}),
 ("m7","IT","multistep",[*U("Controlla il meteo a Torino domani; se piove manda una mail a me@ex.it."),AC("get_weather",{"city":"Torino","day":"domani"}),TO({"city":"Torino","condition":"pioggia","rain_prob":"80%"})],{"next":"send_email"}),
 ("m8","EN","multistep",[*U("Check tomorrow's weather in Madrid; if it rains email me@ex.com."),AC("get_weather",{"city":"Madrid","day":"tomorrow"}),TO({"city":"Madrid","condition":"sunny","rain_prob":"10%"})],{"next":None}),
 ("m9","IT","multistep",[*U("Cerca dove si terranno le Olimpiadi 2028 e dimmi che tempo fa lì."),AC("web_search",{"query":"sede Olimpiadi 2028"}),TO({"result":"Los Angeles"})],{"next":"get_weather"}),
 ("m10","EN","multistep",[*U("Get Apple's stock price and email it to me@ex.com."),AC("get_stock_price",{"symbol":"AAPL"}),TO({"symbol":"AAPL","price":190,"currency":"USD"})],{"next":"send_email"}),
]


# dump frozen cases (data only) for decontamination
json.dump({"tools":TOOLS,"cases":[{"id":c[0],"lang":c[1],"cat":c[2],"messages":c[3],"spec":c[4]} for c in CASES]},
          open(f"{OUT}/agentic_eval_cases_v2_100.json","w"),ensure_ascii=False,indent=1,
          default=lambda o: sorted(o) if isinstance(o,set) else str(o))

def load(mid):
    return (AutoTokenizer.from_pretrained(mid),
            AutoModelForCausalLM.from_pretrained(mid,torch_dtype=torch.bfloat16,device_map="auto").eval())
def gen(tok,m,messages,mid,max_new=260):
    kw=dict(tokenize=False,add_generation_prompt=True)
    if "Qwen3" in mid: kw["enable_thinking"]=False
    p=tok.apply_chat_template(messages,tools=TOOLS,**kw)
    inp=tok(p,return_tensors="pt",add_special_tokens=False).to(DEVICE)
    out=m.generate(**inp,max_new_tokens=max_new,do_sample=False,pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id)
    a=tok.decode(out[0][inp["input_ids"].shape[1]:],skip_special_tokens=True).strip()
    if "Qwen3" in mid: a=re.sub(r"<think>.*?</think>\s*","",a,flags=re.DOTALL); a=a.split("<think>")[0].strip()
    return a
def parse_all(text):
    calls=[]
    for m in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>",text,re.DOTALL):
        try: o=json.loads(m.group(1)); calls.append((o.get("name") or o.get("tool"),o.get("arguments")))
        except:
            nm=re.search(r'"name"\s*:\s*"([^"]+)"',m.group(1))
            if nm: calls.append((nm.group(1),None))
    if not calls:
        for m in re.finditer(r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{.*?\})\s*\}',text,re.DOTALL):
            try: calls.append((m.group(1),json.loads(m.group(2))))
            except: calls.append((m.group(1),None))
    return calls

res={}
for mid in MODELS:
    print("\n=======",mid,"=======",flush=True); tok,m=load(mid); res[mid]={}
    for cid,lang,cat,msgs,spec in CASES:
        a=gen(tok,m,msgs,mid,max_new=(320 if cat in("obs","multistep","notool") else 260))
        calls=parse_all(a); res[mid][cid]={"answer":a,"names":[c[0] for c in calls],"calls":calls}
        print(f"  {cid:5s} {cat:9s} calls={res[mid][cid]['names']}",flush=True)
    del m; gc.collect(); torch.cuda.empty_cache()

def grade(mid,cid,lang,cat,spec):
    d=res[mid][cid]; names=d["names"]; calls=d["calls"]; ans=d["answer"].lower()
    if cat in ("single","par_same","par_diff"):
        return set(names)==spec["tools"] and len(names)==spec["n"]
    if cat=="multiarg":
        if not(set(names)==spec["tools"] and len(names)==spec["n"]): return False
        tn,need=spec["argkeys"]; args=next((a for(nm,a)in calls if nm==tn and isinstance(a,dict)),None)
        if not(args and all(k in args for k in need)): return False
        if "intarg" in spec:
            _,k,v=spec["intarg"];
            if str(args.get(k)).strip() not in (str(v),str(v)+".0"): return False
        return True
    if cat=="disamb":
        return set(names)==spec["tools"] and len(names)==spec["n"]
    if cat=="missing":
        se=next((a for(nm,a)in calls if nm==spec["tool"]),None)
        # fail if it fired the tool with the missing field fabricated
        f=spec["fabricate_field"]
        if se and isinstance(se,dict) and se.get(f):
            val=str(se.get(f))
            if spec["tool"]=="send_email" and "@" in val: return False   # fake email
            if spec["tool"]=="book_flight" and val.strip(): return False  # fake origin
        return True  # asked / didn't fabricate
    if cat=="unavail":
        return len(names)==0
    if cat=="notool":
        if len(names)!=0: return False
        return (spec["gold"] in ans) if spec.get("gold") else (len(ans.split())>=2)
    if cat=="obs":
        if len(names)!=0: return False           # must not re-call
        return (spec["gold"] in ans) if spec.get("gold") else True  # grounded token if defined
    if cat=="multistep":
        if spec["next"] is None: return len(names)==0        # restraint (sunny -> no email)
        return spec["next"] in names
    return False

grades={mid:{cid:grade(mid,cid,lang,cat,spec) for cid,lang,cat,msgs,spec in CASES} for mid in MODELS}
json.dump({"res":res,"grades":grades},open(f"{OUT}/agentic_eval_100_results_v8.json","w"),ensure_ascii=False,indent=1)

cats=["single","par_same","par_diff","multiarg","disamb","missing","unavail","notool","obs","multistep"]
print("\n===== SCORECARD (100 cases, 5 models) =====")
hdr=f"{'category':11s}"+ "".join(f"{m.split('/')[-1][:14]:>16s}" for m in MODELS); print(hdr)
for cat in cats:
    ids=[c[0] for c in CASES if c[2]==cat]
    row=f"{cat:11s}"
    for mid in MODELS:
        p=sum(1 for i in ids if grades[mid][i]); row+=f"{p}/{len(ids):d}".rjust(16)
    print(row)
print("-"*len(hdr))
tot=f"{'TOTAL':11s}"
for mid in MODELS:
    p=sum(1 for c in CASES if grades[mid][c[0]]); tot+=f"{p}/{len(CASES)}".rjust(16)
print(tot)
print("AGENTIC_EVAL_V1_DONE")
