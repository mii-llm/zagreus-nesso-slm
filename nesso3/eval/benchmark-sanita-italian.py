#!/usr/bin/env python3
"""NEW held-out Italian agentic benchmark — CUP / Sanita digitale (healthcare
booking). A completely novel domain (no weather/order/robot/currency overlap)
with Italian tool names + Italian schemas, to pick the best model on unseen
Italian tool-use. Reuses the fair engine (per-model native contract: compact
<tools> injection, v8 verbose template, Qwen native). Function tests only."""
from __future__ import annotations
import importlib.util, sys
from pathlib import Path
from typing import Any

ENGINE_PATH = Path(__file__).with_name("test-fc-conv-sol-5.6.py")
SPEC = importlib.util.spec_from_file_location("bilingual_eval_engine", ENGINE_PATH)
engine = importlib.util.module_from_spec(SPEC); sys.modules[SPEC.name] = engine
SPEC.loader.exec_module(engine)

def T(name, desc, props, req):
    return {"type":"function","function":{"name":name,"description":desc,
            "parameters":{"type":"object","properties":props,"required":req}}}
S=lambda d=None: {"type":"string"} if d is None else {"type":"string","description":d}

TOOLS=[
 T("cerca_prestazione","Cerca una prestazione sanitaria (visita o esame) per nome.",
   {"nome":S("nome della prestazione, es. 'visita cardiologica'")},["nome"]),
 T("elenco_strutture","Elenca le strutture sanitarie di una citta che erogano una prestazione.",
   {"citta":S(),"prestazione":S()},["citta","prestazione"]),
 T("verifica_disponibilita","Verifica gli orari disponibili per una prestazione in una struttura in una data. NON prenota.",
   {"prestazione":S(),"struttura":S(),"data":S("data ISO AAAA-MM-GG")},["prestazione","struttura","data"]),
 T("prenota_appuntamento","Prenota (conferma) un appuntamento per una prestazione.",
   {"prestazione":S(),"struttura":S(),"data":S("data ISO AAAA-MM-GG"),"ora":S("HH:MM"),
    "codice_fiscale":S()},["prestazione","struttura","data","ora","codice_fiscale"]),
 T("annulla_prenotazione","Annulla una prenotazione esistente dato il suo codice.",
   {"codice_prenotazione":S()},["codice_prenotazione"]),
 T("verifica_esenzione","Verifica se un'esenzione ticket e valida per un assistito.",
   {"codice_fiscale":S(),"codice_esenzione":S("es. E01, E02, E30")},["codice_fiscale","codice_esenzione"]),
 T("calcola_ticket","Calcola l'importo del ticket per una prestazione, considerando l'esenzione.",
   {"prestazione":S(),"esente":{"type":"boolean"}},["prestazione","esente"]),
 T("consulta_ricetta","Consulta una ricetta dematerializzata dato il numero ricetta elettronica (NRE).",
   {"numero_ricetta":S()},["numero_ricetta"]),
 T("recupera_referto","Recupera un referto medico dato codice fiscale e identificativo referto.",
   {"codice_fiscale":S(),"id_referto":S()},["codice_fiscale","id_referto"]),
 T("invia_promemoria","Invia un promemoria per una prenotazione su un canale (sms o email).",
   {"codice_prenotazione":S(),"canale":{"type":"string","enum":["sms","email"]}},["codice_prenotazione","canale"]),
]

def call(name,**args): return {"name":name,"arguments":args}
FT=[]
def fc(id_,sub,prompt,expected,**extra):
    FT.append({"id":id_,"subtype":sub,"language":"it","prompt":prompt,"expected":expected,**extra})

# --- single_call ---
fc("san_it_cerca","single_call","Cerca la prestazione 'risonanza magnetica ginocchio'.",
   [call("cerca_prestazione",nome="risonanza magnetica ginocchio")])
fc("san_it_annulla","single_call","Annulla la prenotazione con codice PRN-4471.",
   [call("annulla_prenotazione",codice_prenotazione="PRN-4471")])
fc("san_it_ricetta","single_call","Consulta la ricetta dematerializzata numero RCE-2026-88213.",
   [call("consulta_ricetta",numero_ricetta="RCE-2026-88213")])
fc("san_it_strutture","single_call","Mostrami le strutture a Bologna per una visita cardiologica.",
   [call("elenco_strutture",citta="Bologna",prestazione="visita cardiologica")])

# --- multi_argument ---
fc("san_it_prenota","multi_argument","Prenota una visita dermatologica alla struttura Ospedale San Raffaele il 2026-10-05 alle 11:20 per il codice fiscale RSSMRA85M01H501Z.",
   [call("prenota_appuntamento",prestazione="visita dermatologica",struttura="Ospedale San Raffaele",data="2026-10-05",ora="11:20",codice_fiscale="RSSMRA85M01H501Z")])
fc("san_it_disp","multi_argument","Verifica la disponibilita per una ecografia addome alla struttura Poliambulatorio Centro il 2026-09-30.",
   [call("verifica_disponibilita",prestazione="ecografia addome",struttura="Poliambulatorio Centro",data="2026-09-30")])
fc("san_it_esenz","multi_argument","Verifica l'esenzione E02 per il codice fiscale VRDLGI90A41F205K.",
   [call("verifica_esenzione",codice_fiscale="VRDLGI90A41F205K",codice_esenzione="E02")])
fc("san_it_promemoria","multi_argument","Invia un promemoria via sms per la prenotazione PRN-9080.",
   [call("invia_promemoria",codice_prenotazione="PRN-9080",canale="sms")])

# --- parallel_same_tool ---
fc("san_it_par_annulla","parallel_same_tool","Annulla le prenotazioni PRN-1001 e PRN-1002.",
   [call("annulla_prenotazione",codice_prenotazione="PRN-1001"),call("annulla_prenotazione",codice_prenotazione="PRN-1002")])
fc("san_it_par_ricetta","parallel_same_tool","Consulta le ricette RCE-100 e RCE-200.",
   [call("consulta_ricetta",numero_ricetta="RCE-100"),call("consulta_ricetta",numero_ricetta="RCE-200")])
fc("san_it_par_cerca","parallel_same_tool","Cerca le prestazioni 'visita oculistica' e 'campo visivo'.",
   [call("cerca_prestazione",nome="visita oculistica"),call("cerca_prestazione",nome="campo visivo")])
fc("san_it_par_referto","parallel_same_tool","Recupera i referti REF-55 e REF-77 per il codice fiscale BNCLRA92T50L219X.",
   [call("recupera_referto",codice_fiscale="BNCLRA92T50L219X",id_referto="REF-55"),call("recupera_referto",codice_fiscale="BNCLRA92T50L219X",id_referto="REF-77")])

# --- parallel_diff_tools ---
fc("san_it_pd_esenz_ticket","parallel_diff_tools","Verifica l'esenzione E01 per BRSNNA80A01F839K e calcola il ticket per una visita ortopedica da non esente.",
   [call("verifica_esenzione",codice_fiscale="BRSNNA80A01F839K",codice_esenzione="E01"),call("calcola_ticket",prestazione="visita ortopedica",esente=False)])
fc("san_it_pd_annulla_ricetta","parallel_diff_tools","Annulla la prenotazione PRN-3003 e consulta la ricetta RCE-3003.",
   [call("annulla_prenotazione",codice_prenotazione="PRN-3003"),call("consulta_ricetta",numero_ricetta="RCE-3003")])
fc("san_it_pd_cerca_strutture","parallel_diff_tools","Cerca la prestazione 'TAC torace' e mostrami le strutture a Torino per TAC torace.",
   [call("cerca_prestazione",nome="TAC torace"),call("elenco_strutture",citta="Torino",prestazione="TAC torace")])
fc("san_it_pd_prom_disp","parallel_diff_tools","Invia un promemoria email per PRN-7007 e verifica la disponibilita di una visita neurologica alla struttura Clinica Villa Serena il 2026-11-02.",
   [call("invia_promemoria",codice_prenotazione="PRN-7007",canale="email"),call("verifica_disponibilita",prestazione="visita neurologica",struttura="Clinica Villa Serena",data="2026-11-02")])

# --- disambiguation ---
fc("san_it_dis_solo_verifica","disambiguation","Voglio solo verificare se ci sono posti per una visita cardiologica alla struttura Ospedale Molinette il 2026-10-10, non prenotare ancora.",
   [call("verifica_disponibilita",prestazione="visita cardiologica",struttura="Ospedale Molinette",data="2026-10-10")])
fc("san_it_dis_conferma","disambiguation","Conferma e prenota davvero la visita ortopedica alla struttura Ospedale Careggi il 2026-10-12 alle 09:00 per RSSMRA85M01H501Z.",
   [call("prenota_appuntamento",prestazione="visita ortopedica",struttura="Ospedale Careggi",data="2026-10-12",ora="09:00",codice_fiscale="RSSMRA85M01H501Z")])
fc("san_it_dis_referto","disambiguation","Recupera il referto dell'esame, non la ricetta: identificativo REF-909 per il cf MRTLNZ88D12G273P.",
   [call("recupera_referto",codice_fiscale="MRTLNZ88D12G273P",id_referto="REF-909")])
fc("san_it_dis_ricetta","disambiguation","Voglio la ricetta dematerializzata numero RCE-4567, non il referto.",
   [call("consulta_ricetta",numero_ricetta="RCE-4567")])

# --- missing_argument (deve chiedere) ---
fc("san_it_miss_prenota","missing_argument","Prenota una visita dermatologica.",[],
   clarification=["struttura","data","ora","codice fiscale","quale","?"])
fc("san_it_miss_esenz","missing_argument","Verifica un'esenzione per il mio codice fiscale RSSMRA85M01H501Z.",[],
   clarification=["codice esenzione","quale esenzione","codice","?"])
fc("san_it_miss_prom","missing_argument","Invia un promemoria per la prenotazione PRN-5005.",[],
   clarification=["canale","sms","email","come","?"])
fc("san_it_miss_referto","missing_argument","Recupera il mio referto.",[],
   clarification=["codice fiscale","identificativo","id","quale","?"])

# --- unavailable_tool (nessuno strumento adatto -> rifiuta) ---
fc("san_it_unav_taxi","unavailable_tool","Prenotami un taxi per andare in ospedale domani alle 8.",[],
   clarification=["non posso","non disponibile","nessuno strumento","non ho","non e disponibile"])
fc("san_it_unav_farmaco","unavailable_tool","Ordina per me un antibiotico su una farmacia online.",[],
   clarification=["non posso","non disponibile","nessuno strumento","non ho","non e disponibile"])
fc("san_it_unav_chiama","unavailable_tool","Chiama al telefono il mio medico di base adesso.",[],
   clarification=["non posso","non disponibile","nessuno strumento","non ho","non e disponibile"])
fc("san_it_unav_paga","unavailable_tool","Paga subito il ticket con la mia carta di credito 4539 1488 0343 6467.",[],
   clarification=["non posso","non disponibile","nessuno strumento","non ho","non e disponibile"])

# --- no_tool_discrimination (domanda generale -> risposta testuale) ---
fc("san_it_nt_ticket","no_tool_discrimination","In una frase, spiega cos'e il ticket sanitario. Non serve usare strumenti.",[],
   answer_terms=["ticket","sanitar","quota","paga","prestazione"],answer_terms_min=2)
fc("san_it_nt_esenzione","no_tool_discrimination","Spiega in breve cosa significa 'esenzione per reddito'. Nessuno strumento.",[],
   answer_terms=["esenzione","reddito","paga","ticket"],answer_terms_min=2)
fc("san_it_nt_prelievo","no_tool_discrimination","Dammi un consiglio generale su come prepararsi a un prelievo del sangue. Senza strumenti.",[],
   answer_terms=["digiuno","acqua","mattina","prelievo","cibo"],answer_terms_min=2)
fc("san_it_nt_nre","no_tool_discrimination","Cosa vuol dire 'ricetta dematerializzata'? Rispondi a parole.",[],
   answer_terms=["ricetta","elettronic","digital","carta","numero"],answer_terms_min=2)

# --- observation_grounding (una chiamata, poi risposta ancorata all'osservazione) ---
fc("san_it_obs_disp","observation_grounding","Verifica la disponibilita per una visita cardiologica alla struttura Ospedale Niguarda il 2026-10-08 e poi dimmi il primo orario libero e il costo del ticket.",
   [call("verifica_disponibilita",prestazione="visita cardiologica",struttura="Ospedale Niguarda",data="2026-10-08")],
   strict_call_count=True,
   observation={"content":"{\"orari_liberi\": [\"09:30\", \"14:00\"], \"ticket_eur\": 23}",
     "criteria":[{"kind":"contains","values":["09:30"]},{"kind":"contains","values":["23"]},
       {"kind":"not_contains","values":["14:00","10:30"],"critical":True},{"kind":"max_words","value":70}]})
fc("san_it_obs_ricetta","observation_grounding","Consulta la ricetta RCE-7788 e dimmi il farmaco e il numero di confezioni.",
   [call("consulta_ricetta",numero_ricetta="RCE-7788")],strict_call_count=True,
   observation={"content":"{\"numero_ricetta\": \"RCE-7788\", \"farmaco\": \"Amoxicillina 1g\", \"confezioni\": 2}",
     "criteria":[{"kind":"contains","values":["Amoxicillina"]},{"kind":"contains","values":["2"]},
       {"kind":"not_contains","values":["Ibuprofene","3 confezioni"],"critical":True},{"kind":"max_words","value":60}]})
fc("san_it_obs_referto","observation_grounding","Recupera il referto REF-321 per il cf FRRGPP70A01H501Z e dimmi l'esame e l'esito.",
   [call("recupera_referto",codice_fiscale="FRRGPP70A01H501Z",id_referto="REF-321")],strict_call_count=True,
   observation={"content":"{\"id_referto\": \"REF-321\", \"esame\": \"Emocromo\", \"esito\": \"nella norma\"}",
     "criteria":[{"kind":"contains","values":["Emocromo"]},{"kind":"contains","values":["norma"]},
       {"kind":"not_contains","values":["alterato","patologico"],"critical":True},{"kind":"max_words","value":60}]})
fc("san_it_obs_esenz","observation_grounding","Verifica l'esenzione E30 per il cf CNTMRC85T10G224L e dimmi se e valida e fino a quando.",
   [call("verifica_esenzione",codice_fiscale="CNTMRC85T10G224L",codice_esenzione="E30")],strict_call_count=True,
   observation={"content":"{\"codice_esenzione\": \"E30\", \"valida\": true, \"scadenza\": \"2027-03-31\"}",
     "criteria":[{"kind":"contains_any","values":["valida","si","sì"]},{"kind":"contains","values":["2027-03-31"]},
       {"kind":"not_contains","values":["non valida","scaduta"],"critical":True},{"kind":"max_words","value":60}]})

# --- multi_step (chiamata1 -> osservazione -> chiamata2 ancorata) ---
fc("san_it_ms_disp_prenota","multi_step","Verifica la disponibilita di una visita oculistica alla struttura Ospedale Sacco il 2026-10-15; dopo il risultato, prenota il primo orario libero per il cf DLLNNA83S45F205W.",
   [call("verifica_disponibilita",prestazione="visita oculistica",struttura="Ospedale Sacco",data="2026-10-15")],
   strict_call_count=True,
   observation={"content":"{\"orari_liberi\": [\"10:00\", \"16:30\"]}",
     "expected":[call("prenota_appuntamento",prestazione="visita oculistica",struttura="Ospedale Sacco",data="2026-10-15",ora="10:00",codice_fiscale="DLLNNA83S45F205W")]})
fc("san_it_ms_esenz_ticket","multi_step","Verifica l'esenzione E01 per il cf RSSMRA85M01H501Z; se risulta valida, calcola di conseguenza il ticket per una visita fisiatrica.",
   [call("verifica_esenzione",codice_fiscale="RSSMRA85M01H501Z",codice_esenzione="E01")],strict_call_count=True,
   observation={"content":"{\"valida\": true, \"codice_esenzione\": \"E01\"}",
     "expected":[call("calcola_ticket",prestazione="visita fisiatrica",esente=True)]})
fc("san_it_ms_disp_prom","multi_step","Verifica la disponibilita di una ecografia tiroide alla struttura Poliambulatorio Nord il 2026-10-20; poi invia un promemoria via email per la prenotazione che ti verra indicata.",
   [call("verifica_disponibilita",prestazione="ecografia tiroide",struttura="Poliambulatorio Nord",data="2026-10-20")],
   strict_call_count=True,
   observation={"content":"{\"orari_liberi\": [\"08:15\"], \"codice_prenotazione\": \"PRN-6161\"}",
     "expected":[call("invia_promemoria",codice_prenotazione="PRN-6161",canale="email")]})
fc("san_it_ms_ricetta_prenota","multi_step","Consulta la ricetta RCE-9012; dopo il risultato, prenota la prestazione indicata nella ricetta alla struttura Ospedale Maggiore il 2026-11-10 alle 12:00 per il cf VRDLGI90A41F205K.",
   [call("consulta_ricetta",numero_ricetta="RCE-9012")],strict_call_count=True,
   observation={"content":"{\"numero_ricetta\": \"RCE-9012\", \"prestazione\": \"visita pneumologica\"}",
     "expected":[call("prenota_appuntamento",prestazione="visita pneumologica",struttura="Ospedale Maggiore",data="2026-11-10",ora="12:00",codice_fiscale="VRDLGI90A41F205K")]})

engine.FUNCTION_SYSTEM_PROMPT=("Sei un assistente per la prenotazione di prestazioni sanitarie (CUP). "
    "Usa uno strumento solo quando serve. Non inventare mai argomenti obbligatori mancanti: "
    "se manca un dato richiesto, chiedi. Se non esiste uno strumento adatto, dillo chiaramente. "
    "Fonda ogni risposta solo sull'osservazione restituita dallo strumento.")
engine.TOOLS=TOOLS
engine.FUNCTION_TESTS=FT
engine.CONVERSATION_TESTS=[]
engine.DEFAULT_MODELS=[
    "giux78/zagreus-0.4B-cpt-full-32k-sft-compact",
    "giux78/zagreus-0.4B-cpt-full-32k-sft-agentic-v8",
    "Qwen/Qwen3-0.6B",
]
if __name__=="__main__":
    engine.main()
