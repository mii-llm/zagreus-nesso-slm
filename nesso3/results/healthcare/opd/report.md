# Bilingual Conversation and Function-Calling Evaluation

Generated: 2026-10-10T21:58:36.781761+00:00

Greedy decoding, bfloat16, native chat templates, Qwen thinking disabled. 
Scores are deterministic checks, not an LLM judge. Inspect raw outputs below.

## Summary

| Model | Overall | Conv IT | Conv EN | FC IT | FC EN | Median latency |
|---|---:|---:|---:|---:|---:|---:|
| giux78/nesso2-0.4B-agentic-v2-opd | 72.7% | N/A | N/A | 72.7% | N/A | 1.26s |

## Function-calling capabilities

Each cell reports mean deterministic score and exact passes (100% score).

| Capability | giux78/nesso2-0.4B-agentic-v2-opd |
|---|---:|
| single_call | 100.0% (4/4) |
| multi_argument | 84.7% (0/4) |
| parallel_same_tool | 95.0% (3/4) |
| parallel_diff_tools | 85.0% (2/4) |
| disambiguation | 89.7% (1/4) |
| missing_argument | 50.0% (2/4) |
| unavailable_tool | 100.0% (4/4) |
| no_tool_discrimination | 67.5% (2/4) |
| observation_grounding | 24.1% (0/4) |
| multi_step | 31.0% (0/4) |

## Per-test scores

| Test | Category | Language | giux78/nesso2-0.4B-agentic-v2-opd |
|---|---|---:|---:|
| san_it_cerca | single_call | it | 100.0% |
| san_it_annulla | single_call | it | 100.0% |
| san_it_ricetta | single_call | it | 100.0% |
| san_it_strutture | single_call | it | 100.0% |
| san_it_prenota | multi_argument | it | 92.0% |
| san_it_disp | multi_argument | it | 86.7% |
| san_it_esenz | multi_argument | it | 80.0% |
| san_it_promemoria | multi_argument | it | 80.0% |
| san_it_par_annulla | parallel_same_tool | it | 100.0% |
| san_it_par_ricetta | parallel_same_tool | it | 100.0% |
| san_it_par_cerca | parallel_same_tool | it | 100.0% |
| san_it_par_referto | parallel_same_tool | it | 80.0% |
| san_it_pd_esenz_ticket | parallel_diff_tools | it | 80.0% |
| san_it_pd_annulla_ricetta | parallel_diff_tools | it | 100.0% |
| san_it_pd_cerca_strutture | parallel_diff_tools | it | 100.0% |
| san_it_pd_prom_disp | parallel_diff_tools | it | 60.0% |
| san_it_dis_solo_verifica | disambiguation | it | 86.7% |
| san_it_dis_conferma | disambiguation | it | 92.0% |
| san_it_dis_referto | disambiguation | it | 80.0% |
| san_it_dis_ricetta | disambiguation | it | 100.0% |
| san_it_miss_prenota | missing_argument | it | 100.0% |
| san_it_miss_esenz | missing_argument | it | 0.0% |
| san_it_miss_prom | missing_argument | it | 0.0% |
| san_it_miss_referto | missing_argument | it | 100.0% |
| san_it_unav_taxi | unavailable_tool | it | 100.0% |
| san_it_unav_farmaco | unavailable_tool | it | 100.0% |
| san_it_unav_chiama | unavailable_tool | it | 100.0% |
| san_it_unav_paga | unavailable_tool | it | 100.0% |
| san_it_nt_ticket | no_tool_discrimination | it | 100.0% |
| san_it_nt_esenzione | no_tool_discrimination | it | 100.0% |
| san_it_nt_prelievo | no_tool_discrimination | it | 70.0% |
| san_it_nt_nre | no_tool_discrimination | it | 0.0% |
| san_it_obs_disp | observation_grounding | it | 35.0% |
| san_it_obs_ricetta | observation_grounding | it | 17.5% |
| san_it_obs_referto | observation_grounding | it | 17.5% |
| san_it_obs_esenz | observation_grounding | it | 26.2% |
| san_it_ms_disp_prenota | multi_step | it | 50.0% |
| san_it_ms_esenz_ticket | multi_step | it | 50.0% |
| san_it_ms_disp_prom | multi_step | it | 10.0% |
| san_it_ms_ricetta_prenota | multi_step | it | 14.0% |

## Raw outputs

### giux78/nesso2-0.4B-agentic-v2-opd

#### san_it_cerca — 100.0%

Category: `single_call` · Language: `it` · Latency: `0.749s` · Output tokens: `35`

**Prompt**

```text
Cerca la prestazione 'risonanza magnetica ginocchio'.
```

**Model output**

```text
<tool_call>
{"name": "cerca_prestazione", "arguments": {"nome": "risonanza magnetica ginocchio"}}
</tool_call>
```

**Checks**

- expected cerca_prestazione, actual cerca_prestazione: name_score=1.00
- arg nome='risonanza magnetica ginocchio', expected 'risonanza magnetica ginocchio': True
- call count expected=1 actual=1 score=1.00

#### san_it_annulla — 100.0%

Category: `single_call` · Language: `it` · Latency: `0.792s` · Output tokens: `38`

**Prompt**

```text
Annulla la prenotazione con codice PRN-4471.
```

**Model output**

```text
<tool_call>
{"name": "annulla_prenotazione", "arguments": {"codice_prenotazione": "PRN-4471"}}
</tool_call>
```

**Checks**

- expected annulla_prenotazione, actual annulla_prenotazione: name_score=1.00
- arg codice_prenotazione='PRN-4471', expected 'PRN-4471': True
- call count expected=1 actual=1 score=1.00

#### san_it_ricetta — 100.0%

Category: `single_call` · Language: `it` · Latency: `0.766s` · Output tokens: `37`

**Prompt**

```text
Consulta la ricetta dematerializzata numero RCE-2026-88213.
```

**Model output**

```text
<tool_call>
{"name": "consulta_ricetta", "arguments": {"numero_ricetta": "RCE-2026-88213"}}
</tool_call>
```

**Checks**

- expected consulta_ricetta, actual consulta_ricetta: name_score=1.00
- arg numero_ricetta='RCE-2026-88213', expected 'RCE-2026-88213': True
- call count expected=1 actual=1 score=1.00

#### san_it_strutture — 100.0%

Category: `single_call` · Language: `it` · Latency: `0.861s` · Output tokens: `42`

**Prompt**

```text
Mostrami le strutture a Bologna per una visita cardiologica.
```

**Model output**

```text
<tool_call>
{"name": "elenco_strutture", "arguments": {"citta": "Bologna", "prestazione": "visita cardiologica"}}
</tool_call>
```

**Checks**

- expected elenco_strutture, actual elenco_strutture: name_score=1.00
- arg citta='Bologna', expected 'Bologna': True
- arg prestazione='visita cardiologica', expected 'visita cardiologica': True
- call count expected=1 actual=1 score=1.00

#### san_it_prenota — 92.0%

Category: `multi_argument` · Language: `it` · Latency: `1.453s` · Output tokens: `70`

**Prompt**

```text
Prenota una visita dermatologica alla struttura Ospedale San Raffaele il 2026-10-05 alle 11:20 per il codice fiscale RSSMRA85M01H501Z.
```

**Model output**

```text
<tool_call>
{"name": "prenota_appuntamento", "arguments": {"prestazione": "visita dermatologica", "struttura": "Ospedale San Raffaele", "data": "2026-10-05", "ora": "11:20"}}
</tool_call>
```

**Checks**

- expected prenota_appuntamento, actual prenota_appuntamento: name_score=1.00
- arg prestazione='visita dermatologica', expected 'visita dermatologica': True
- arg struttura='Ospedale San Raffaele', expected 'Ospedale San Raffaele': True
- arg data='2026-10-05', expected '2026-10-05': True
- arg ora='11:20', expected '11:20': True
- arg codice_fiscale=None, expected 'RSSMRA85M01H501Z': False
- call count expected=1 actual=1 score=1.00

#### san_it_disp — 86.7%

Category: `multi_argument` · Language: `it` · Latency: `0.947s` · Output tokens: `45`

**Prompt**

```text
Verifica la disponibilita per una ecografia addome alla struttura Poliambulatorio Centro il 2026-09-30.
```

**Model output**

```text
<tool_call>
{"name": "verifica_disponibilita", "arguments": {"prestazione": "ecografia addome", "data": "2026-09-30"}}
</tool_call>
```

**Checks**

- expected verifica_disponibilita, actual verifica_disponibilita: name_score=1.00
- arg prestazione='ecografia addome', expected 'ecografia addome': True
- arg struttura=None, expected 'Poliambulatorio Centro': False
- arg data='2026-09-30', expected '2026-09-30': True
- call count expected=1 actual=1 score=1.00

#### san_it_esenz — 80.0%

Category: `multi_argument` · Language: `it` · Latency: `0.824s` · Output tokens: `40`

**Prompt**

```text
Verifica l'esenzione E02 per il codice fiscale VRDLGI90A41F205K.
```

**Model output**

```text
<tool_call>
{"name": "verifica_esenzione", "arguments": {"codice_fiscale": "VRDLGI90A41F205K"}}
</tool_call>
```

**Checks**

- expected verifica_esenzione, actual verifica_esenzione: name_score=1.00
- arg codice_fiscale='VRDLGI90A41F205K', expected 'VRDLGI90A41F205K': True
- arg codice_esenzione=None, expected 'E02': False
- call count expected=1 actual=1 score=1.00

#### san_it_promemoria — 80.0%

Category: `multi_argument` · Language: `it` · Latency: `0.747s` · Output tokens: `37`

**Prompt**

```text
Invia un promemoria via sms per la prenotazione PRN-9080.
```

**Model output**

```text
<tool_call>
{"name": "invia_promemoria", "arguments": {"codice_prenotazione": "PRN-9080"}}
</tool_call>
```

**Checks**

- expected invia_promemoria, actual invia_promemoria: name_score=1.00
- arg codice_prenotazione='PRN-9080', expected 'PRN-9080': True
- arg canale=None, expected 'sms': False
- call count expected=1 actual=1 score=1.00

#### san_it_par_annulla — 100.0%

Category: `parallel_same_tool` · Language: `it` · Latency: `1.512s` · Output tokens: `75`

**Prompt**

```text
Annulla le prenotazioni PRN-1001 e PRN-1002.
```

**Model output**

```text
<tool_call>
{"name": "annulla_prenotazione", "arguments": {"codice_prenotazione": "PRN-1001"}}
</tool_call>
<tool_call>
{"name": "annulla_prenotazione", "arguments": {"codice_prenotazione": "PRN-1002"}}
</tool_call>
```

**Checks**

- expected annulla_prenotazione, actual annulla_prenotazione: name_score=1.00
- arg codice_prenotazione='PRN-1001', expected 'PRN-1001': True
- expected annulla_prenotazione, actual annulla_prenotazione: name_score=1.00
- arg codice_prenotazione='PRN-1002', expected 'PRN-1002': True
- call count expected=2 actual=2 score=1.00

#### san_it_par_ricetta — 100.0%

Category: `parallel_same_tool` · Language: `it` · Latency: `1.333s` · Output tokens: `65`

**Prompt**

```text
Consulta le ricette RCE-100 e RCE-200.
```

**Model output**

```text
<tool_call>
{"name": "consulta_ricetta", "arguments": {"numero_ricetta": "RCE-100"}}
</tool_call>
<tool_call>
{"name": "consulta_ricetta", "arguments": {"numero_ricetta": "RCE-200"}}
</tool_call>
```

**Checks**

- expected consulta_ricetta, actual consulta_ricetta: name_score=1.00
- arg numero_ricetta='RCE-100', expected 'RCE-100': True
- expected consulta_ricetta, actual consulta_ricetta: name_score=1.00
- arg numero_ricetta='RCE-200', expected 'RCE-200': True
- call count expected=2 actual=2 score=1.00

#### san_it_par_cerca — 100.0%

Category: `parallel_same_tool` · Language: `it` · Latency: `1.382s` · Output tokens: `61`

**Prompt**

```text
Cerca le prestazioni 'visita oculistica' e 'campo visivo'.
```

**Model output**

```text
<tool_call>
{"name": "cerca_prestazione", "arguments": {"nome": "visita oculistica"}}
</tool_call>
<tool_call>
{"name": "cerca_prestazione", "arguments": {"nome": "campo visivo"}}
</tool_call>
```

**Checks**

- expected cerca_prestazione, actual cerca_prestazione: name_score=1.00
- arg nome='visita oculistica', expected 'visita oculistica': True
- expected cerca_prestazione, actual cerca_prestazione: name_score=1.00
- arg nome='campo visivo', expected 'campo visivo': True
- call count expected=2 actual=2 score=1.00

#### san_it_par_referto — 80.0%

Category: `parallel_same_tool` · Language: `it` · Latency: `1.863s` · Output tokens: `79`

**Prompt**

```text
Recupera i referti REF-55 e REF-77 per il codice fiscale BNCLRA92T50L219X.
```

**Model output**

```text
<tool_call>
{"name": "recupera_referto", "arguments": {"codice_fiscale": "BNCLRA92T50L219X"}}
</tool_call>
<tool_call>
{"name": "recupera_referto", "arguments": {"codice_fiscale": "BNCLRA92T50L219X"}}
</tool_call>
```

**Checks**

- expected recupera_referto, actual recupera_referto: name_score=1.00
- arg codice_fiscale='BNCLRA92T50L219X', expected 'BNCLRA92T50L219X': True
- arg id_referto=None, expected 'REF-55': False
- expected recupera_referto, actual recupera_referto: name_score=1.00
- arg codice_fiscale='BNCLRA92T50L219X', expected 'BNCLRA92T50L219X': True
- arg id_referto=None, expected 'REF-77': False
- call count expected=2 actual=2 score=1.00

#### san_it_pd_esenz_ticket — 80.0%

Category: `parallel_diff_tools` · Language: `it` · Latency: `1.888s` · Output tokens: `80`

**Prompt**

```text
Verifica l'esenzione E01 per BRSNNA80A01F839K e calcola il ticket per una visita ortopedica da non esente.
```

**Model output**

```text
<tool_call>
{"name": "verifica_esenzione", "arguments": {"codice_fiscale": "BRSNNA80A01F839K"}}
</tool_call>
<tool_call>
{"name": "calcola_ticket", "arguments": {"prestazione": "visita ortopedica", "esente": "E01"}}
</tool_call>
```

**Checks**

- expected verifica_esenzione, actual verifica_esenzione: name_score=1.00
- arg codice_fiscale='BRSNNA80A01F839K', expected 'BRSNNA80A01F839K': True
- arg codice_esenzione=None, expected 'E01': False
- expected calcola_ticket, actual calcola_ticket: name_score=1.00
- arg prestazione='visita ortopedica', expected 'visita ortopedica': True
- arg esente='E01', expected False: False
- call count expected=2 actual=2 score=1.00

#### san_it_pd_annulla_ricetta — 100.0%

Category: `parallel_diff_tools` · Language: `it` · Latency: `1.735s` · Output tokens: `71`

**Prompt**

```text
Annulla la prenotazione PRN-3003 e consulta la ricetta RCE-3003.
```

**Model output**

```text
<tool_call>
{"name": "annulla_prenotazione", "arguments": {"codice_prenotazione": "PRN-3003"}}
</tool_call>
<tool_call>
{"name": "consulta_ricetta", "arguments": {"numero_ricetta": "RCE-3003"}}
</tool_call>
```

**Checks**

- expected annulla_prenotazione, actual annulla_prenotazione: name_score=1.00
- arg codice_prenotazione='PRN-3003', expected 'PRN-3003': True
- expected consulta_ricetta, actual consulta_ricetta: name_score=1.00
- arg numero_ricetta='RCE-3003', expected 'RCE-3003': True
- call count expected=2 actual=2 score=1.00

#### san_it_pd_cerca_strutture — 100.0%

Category: `parallel_diff_tools` · Language: `it` · Latency: `1.700s` · Output tokens: `71`

**Prompt**

```text
Cerca la prestazione 'TAC torace' e mostrami le strutture a Torino per TAC torace.
```

**Model output**

```text
<tool_call>
{"name": "cerca_prestazione", "arguments": {"nome": "TAC torace"}}
</tool_call>
<tool_call>
{"name": "elenco_strutture", "arguments": {"citta": "Torino", "prestazione": "TAC torace"}}
</tool_call>
```

**Checks**

- expected cerca_prestazione, actual cerca_prestazione: name_score=1.00
- arg nome='TAC torace', expected 'TAC torace': True
- expected elenco_strutture, actual elenco_strutture: name_score=1.00
- arg citta='Torino', expected 'Torino': True
- arg prestazione='TAC torace', expected 'TAC torace': True
- call count expected=2 actual=2 score=1.00

#### san_it_pd_prom_disp — 60.0%

Category: `parallel_diff_tools` · Language: `it` · Latency: `2.034s` · Output tokens: `84`

**Prompt**

```text
Invia un promemoria email per PRN-7007 e verifica la disponibilita di una visita neurologica alla struttura Clinica Villa Serena il 2026-11-02.
```

**Model output**

```text
<tool_call>
{"name": "invia_promemoria", "arguments": {"codice_prenotazione": "PRN-7007", "canale": "email"}}
</tool_call>
<tool_call>
{"name": "cerca_prestazione", "arguments": {"nome": "visita neurologica", "prestazione": "visita"}}
</tool_call>
```

**Checks**

- expected invia_promemoria, actual invia_promemoria: name_score=1.00
- arg codice_prenotazione='PRN-7007', expected 'PRN-7007': True
- arg canale='email', expected 'email': True
- expected verifica_disponibilita, actual cerca_prestazione: name_score=0.00
- arg prestazione='visita', expected 'visita neurologica': False
- arg struttura=None, expected 'Clinica Villa Serena': False
- arg data=None, expected '2026-11-02': False
- extra arguments (not penalized): ['nome']
- call count expected=2 actual=2 score=1.00

#### san_it_dis_solo_verifica — 86.7%

Category: `disambiguation` · Language: `it` · Latency: `1.131s` · Output tokens: `46`

**Prompt**

```text
Voglio solo verificare se ci sono posti per una visita cardiologica alla struttura Ospedale Molinette il 2026-10-10, non prenotare ancora.
```

**Model output**

```text
<tool_call>
{"name": "verifica_disponibilita", "arguments": {"prestazione": "visita cardiologica", "data": "2026-10-10"}}
</tool_call>
```

**Checks**

- expected verifica_disponibilita, actual verifica_disponibilita: name_score=1.00
- arg prestazione='visita cardiologica', expected 'visita cardiologica': True
- arg struttura=None, expected 'Ospedale Molinette': False
- arg data='2026-10-10', expected '2026-10-10': True
- call count expected=1 actual=1 score=1.00

#### san_it_dis_conferma — 92.0%

Category: `disambiguation` · Language: `it` · Latency: `1.618s` · Output tokens: `68`

**Prompt**

```text
Conferma e prenota davvero la visita ortopedica alla struttura Ospedale Careggi il 2026-10-12 alle 09:00 per RSSMRA85M01H501Z.
```

**Model output**

```text
<tool_call>
{"name": "prenota_appuntamento", "arguments": {"prestazione": "visita ortopedica", "struttura": "Ospedale Careggi", "data": "2026-10-12", "ora": "09:00"}}
</tool_call>
```

**Checks**

- expected prenota_appuntamento, actual prenota_appuntamento: name_score=1.00
- arg prestazione='visita ortopedica', expected 'visita ortopedica': True
- arg struttura='Ospedale Careggi', expected 'Ospedale Careggi': True
- arg data='2026-10-12', expected '2026-10-12': True
- arg ora='09:00', expected '09:00': True
- arg codice_fiscale=None, expected 'RSSMRA85M01H501Z': False
- call count expected=1 actual=1 score=1.00

#### san_it_dis_referto — 80.0%

Category: `disambiguation` · Language: `it` · Latency: `0.965s` · Output tokens: `41`

**Prompt**

```text
Recupera il referto dell'esame, non la ricetta: identificativo REF-909 per il cf MRTLNZ88D12G273P.
```

**Model output**

```text
<tool_call>
{"name": "recupera_referto", "arguments": {"codice_fiscale": "MRTLNZ88D12G273P"}}
</tool_call>
```

**Checks**

- expected recupera_referto, actual recupera_referto: name_score=1.00
- arg codice_fiscale='MRTLNZ88D12G273P', expected 'MRTLNZ88D12G273P': True
- arg id_referto=None, expected 'REF-909': False
- call count expected=1 actual=1 score=1.00

#### san_it_dis_ricetta — 100.0%

Category: `disambiguation` · Language: `it` · Latency: `0.826s` · Output tokens: `34`

**Prompt**

```text
Voglio la ricetta dematerializzata numero RCE-4567, non il referto.
```

**Model output**

```text
<tool_call>
{"name": "consulta_ricetta", "arguments": {"numero_ricetta": "RCE-4567"}}
</tool_call>
```

**Checks**

- expected consulta_ricetta, actual consulta_ricetta: name_score=1.00
- arg numero_ricetta='RCE-4567', expected 'RCE-4567': True
- call count expected=1 actual=1 score=1.00

#### san_it_miss_prenota — 100.0%

Category: `missing_argument` · Language: `it` · Latency: `0.539s` · Output tokens: `22`

**Prompt**

```text
Prenota una visita dermatologica.
```

**Model output**

```text
Per procedere, ho bisogno di conoscere il codice fiscale dell'assistito.
```

**Checks**

- expected no calls; got 0
- non-tool response matched ['codice fiscale']; required at least 1 from ['struttura', 'data', 'ora', 'codice fiscale', 'quale', '?']: True

#### san_it_miss_esenz — 0.0%

Category: `missing_argument` · Language: `it` · Latency: `1.042s` · Output tokens: `42`

**Prompt**

```text
Verifica un'esenzione per il mio codice fiscale RSSMRA85M01H501Z.
```

**Model output**

```text
<tool_call>
{"name": "annulla_prenotazione", "arguments": {"codice_prenotazione": "RSSMRA85M01H501Z"}}
</tool_call>
```

**Checks**

- expected no calls; got 1
- non-tool response matched ['codice']; required at least 1 from ['codice esenzione', 'quale esenzione', 'codice', '?']: False

#### san_it_miss_prom — 0.0%

Category: `missing_argument` · Language: `it` · Latency: `0.902s` · Output tokens: `37`

**Prompt**

```text
Invia un promemoria per la prenotazione PRN-5005.
```

**Model output**

```text
<tool_call>
{"name": "invia_promemoria", "arguments": {"codice_prenotazione": "PRN-5005"}}
</tool_call>
```

**Checks**

- expected no calls; got 1
- non-tool response matched []; required at least 1 from ['canale', 'sms', 'email', 'come', '?']: False

#### san_it_miss_referto — 100.0%

Category: `missing_argument` · Language: `it` · Latency: `0.537s` · Output tokens: `22`

**Prompt**

```text
Recupera il mio referto.
```

**Model output**

```text
Per procedere, ho bisogno del codice fiscale. Potresti fornirmelo?
```

**Checks**

- expected no calls; got 0
- non-tool response matched ['codice fiscale', '?']; required at least 1 from ['codice fiscale', 'identificativo', 'id', 'quale', '?']: True

#### san_it_unav_taxi — 100.0%

Category: `unavailable_tool` · Language: `it` · Latency: `1.252s` · Output tokens: `52`

**Prompt**

```text
Prenotami un taxi per andare in ospedale domani alle 8.
```

**Model output**

```text
Mi dispiace, non posso prenotare un taxi per te. Posso aiutarti a trovare un'opzione di trasporto alternativa, come un'auto a noleggio o un servizio di trasporto pubblico.
```

**Checks**

- expected no calls; got 0
- non-tool response matched ['non posso']; required at least 1 from ['non posso', 'non disponibile', 'nessuno strumento', 'non ho', 'non e disponibile']: True

#### san_it_unav_farmaco — 100.0%

Category: `unavailable_tool` · Language: `it` · Latency: `1.112s` · Output tokens: `45`

**Prompt**

```text
Ordina per me un antibiotico su una farmacia online.
```

**Model output**

```text
Mi dispiace, non posso ordinare per me un antibiotico online. Posso aiutarti a trovare un antibiotico in farmacia, ma non posso effettuare ordini per te.
```

**Checks**

- expected no calls; got 0
- non-tool response matched ['non posso']; required at least 1 from ['non posso', 'non disponibile', 'nessuno strumento', 'non ho', 'non e disponibile']: True

#### san_it_unav_chiama — 100.0%

Category: `unavailable_tool` · Language: `it` · Latency: `0.849s` · Output tokens: `35`

**Prompt**

```text
Chiama al telefono il mio medico di base adesso.
```

**Model output**

```text
Mi dispiace, non posso fare che. Posso aiutarti a trovare un medico di base, ma non posso chiamare il telefono.
```

**Checks**

- expected no calls; got 0
- non-tool response matched ['non posso']; required at least 1 from ['non posso', 'non disponibile', 'nessuno strumento', 'non ho', 'non e disponibile']: True

#### san_it_unav_paga — 100.0%

Category: `unavailable_tool` · Language: `it` · Latency: `1.236s` · Output tokens: `51`

**Prompt**

```text
Paga subito il ticket con la mia carta di credito 4539 1488 0343 6467.
```

**Model output**

```text
Mi dispiace, ma non posso effettuare pagamenti. Posso aiutarti a trovare un modo per pagare il ticket, come un'offerta di prenotazione o un'offerta di ricarica.
```

**Checks**

- expected no calls; got 0
- non-tool response matched ['non posso']; required at least 1 from ['non posso', 'non disponibile', 'nessuno strumento', 'non ho', 'non e disponibile']: True

#### san_it_nt_ticket — 100.0%

Category: `no_tool_discrimination` · Language: `it` · Latency: `0.962s` · Output tokens: `40`

**Prompt**

```text
In una frase, spiega cos'e il ticket sanitario. Non serve usare strumenti.
```

**Model output**

```text
Il ticket sanitario è un documento che permette di pagare per prestazioni di assistenza sanitaria, come visite o esami, e viene emesso dal sistema sanitario nazionale.
```

**Checks**

- expected no calls; got 0
- non-tool response matched ['ticket', 'sanitar', 'paga']; required at least 2 from ['ticket', 'sanitar', 'quota', 'paga', 'prestazione']: True

#### san_it_nt_esenzione — 100.0%

Category: `no_tool_discrimination` · Language: `it` · Latency: `1.263s` · Output tokens: `53`

**Prompt**

```text
Spiega in breve cosa significa 'esenzione per reddito'. Nessuno strumento.
```

**Model output**

```text
L'esenzione per reddito è un beneficio che consente a un assistito di usufruire di una prestazione sanitaria senza pagare un ticket, a condizione che il suo reddito sia inferiore a una certa soglia.
```

**Checks**

- expected no calls; got 0
- non-tool response matched ['esenzione', 'reddito', 'paga', 'ticket']; required at least 2 from ['esenzione', 'reddito', 'paga', 'ticket']: True

#### san_it_nt_prelievo — 70.0%

Category: `no_tool_discrimination` · Language: `it` · Latency: `1.290s` · Output tokens: `54`

**Prompt**

```text
Dammi un consiglio generale su come prepararsi a un prelievo del sangue. Senza strumenti.
```

**Model output**

```text
Non ho accesso a informazioni in tempo reale o strumenti per fornire consigli specifici su come prepararsi a un prelievo del sangue. Posso solo fornire informazioni generali su come prepararsi per questo esame.
```

**Checks**

- expected no calls; got 0
- non-tool response matched ['prelievo']; required at least 2 from ['digiuno', 'acqua', 'mattina', 'prelievo', 'cibo']: False

#### san_it_nt_nre — 0.0%

Category: `no_tool_discrimination` · Language: `it` · Latency: `0.747s` · Output tokens: `31`

**Prompt**

```text
Cosa vuol dire 'ricetta dematerializzata'? Rispondi a parole.
```

**Model output**

```text
<tool_call>
{"name": "consulta_ricetta", "arguments": {"numero_ricetta": "NRE"}}
</tool_call>
```

**Checks**

- expected no calls; got 1
- non-tool response matched ['ricetta', 'numero']; required at least 2 from ['ricetta', 'elettronic', 'digital', 'carta', 'numero']: False

#### san_it_obs_disp — 35.0%

Category: `observation_grounding` · Language: `it` · Latency: `4.466s` · Output tokens: `188`

**Prompt**

```text
Verifica la disponibilita per una visita cardiologica alla struttura Ospedale Niguarda il 2026-10-08 e poi dimmi il primo orario libero e il costo del ticket.
```

**Model output**

```text
STEP 1 — MODEL TOOL DECISION
<tool_call>
{"name": "verifica_disponibilita", "arguments": {"prestazione": "visita cardiologica", "data": "2026-10-08"}}
</tool_call>
<tool_call>
{"name": "prenota_appuntamento", "arguments": {"prestazione": "visita cardiologica", "struttura": "Ospedale Niguarda", "data": "2026-10-08"}}
</tool_call>

TOOL OBSERVATION
{"orari_liberi": ["09:30", "14:00"], "ticket_eur": 23}

STEP 2 — MODEL RESPONSE
<tool_call>
{"name": "prenota_appuntamento", "arguments": {"prestazione": "visita cardiologica", "struttura": "Ospedale Niguarda", "data": "2026-10-08", "ora": "09:30", "codice_fiscale": "1234567890"}}
</tool_call>
```

**Checks**

- step 1: expected verifica_disponibilita, actual verifica_disponibilita: name_score=1.00
- step 1: arg prestazione='visita cardiologica', expected 'visita cardiologica': True
- step 1: arg struttura=None, expected 'Ospedale Niguarda': False
- step 1: arg data='2026-10-08', expected '2026-10-08': True
- step 1: call count expected=1 actual=2 score=0.50
- step 1: strict call count failed: expected 1, got 2; stage score forced to 0
- step 2: contains one of ['09:30']: True
- step 2: contains one of ['23']: True
- step 2: avoids ['14:00', '10:30']: True
- step 2: words 18 <= 70: True
- step 2: does not call another tool after receiving the observation: False
- combined two-step score: (0.000 + 0.700) / 2

#### san_it_obs_ricetta — 17.5%

Category: `observation_grounding` · Language: `it` · Latency: `2.692s` · Output tokens: `110`

**Prompt**

```text
Consulta la ricetta RCE-7788 e dimmi il farmaco e il numero di confezioni.
```

**Model output**

```text
STEP 1 — MODEL TOOL DECISION
<tool_call>
{"name": "consulta_ricetta", "arguments": {"numero_ricetta": "RCE-7788"}}
</tool_call>
<tool_call>
{"name": "ricetta", "arguments": {"numero_fiscale": "RCE-7788"}}
</tool_call>

TOOL OBSERVATION
{"numero_ricetta": "RCE-7788", "farmaco": "Amoxicillina 1g", "confezioni": 2}

STEP 2 — MODEL RESPONSE
<tool_call>
{"name": "ricetta", "arguments": {"numero_fiscale": "RCE-7788", "id_ricetta": "RCE-7788"}}
</tool_call>
```

**Checks**

- step 1: expected consulta_ricetta, actual consulta_ricetta: name_score=1.00
- step 1: arg numero_ricetta='RCE-7788', expected 'RCE-7788': True
- step 1: call count expected=1 actual=2 score=0.50
- step 1: strict call count failed: expected 1, got 2; stage score forced to 0
- step 2: contains one of ['Amoxicillina']: False
- step 2: contains one of ['2']: False
- step 2: avoids ['Ibuprofene', '3 confezioni']: True
- step 2: words 9 <= 60: True
- step 2: does not call another tool after receiving the observation: False
- combined two-step score: (0.000 + 0.350) / 2

#### san_it_obs_referto — 17.5%

Category: `observation_grounding` · Language: `it` · Latency: `2.510s` · Output tokens: `104`

**Prompt**

```text
Recupera il referto REF-321 per il cf FRRGPP70A01H501Z e dimmi l'esame e l'esito.
```

**Model output**

```text
STEP 1 — MODEL TOOL DECISION
<tool_call>
{"name": "recupera_referto", "arguments": {"codice_fiscale": "FRRGPP70A01H501Z"}}
</tool_call>
<tool_call>
{"name": "cerca_prestazione", "arguments": {"nome": "esame e esito"}}
</tool_call>

TOOL OBSERVATION
{"id_referto": "REF-321", "esame": "Emocromo", "esito": "nella norma"}

STEP 2 — MODEL RESPONSE
<tool_call>
{"name": "cerca_prestazione", "arguments": {"nome": "esame e esito"}}
</tool_call>
```

**Checks**

- step 1: expected recupera_referto, actual recupera_referto: name_score=1.00
- step 1: arg codice_fiscale='FRRGPP70A01H501Z', expected 'FRRGPP70A01H501Z': True
- step 1: arg id_referto=None, expected 'REF-321': False
- step 1: call count expected=1 actual=2 score=0.50
- step 1: strict call count failed: expected 1, got 2; stage score forced to 0
- step 2: contains one of ['Emocromo']: False
- step 2: contains one of ['norma']: False
- step 2: avoids ['alterato', 'patologico']: True
- step 2: words 9 <= 60: True
- step 2: does not call another tool after receiving the observation: False
- combined two-step score: (0.000 + 0.350) / 2

#### san_it_obs_esenz — 26.2%

Category: `observation_grounding` · Language: `it` · Latency: `3.098s` · Output tokens: `129`

**Prompt**

```text
Verifica l'esenzione E30 per il cf CNTMRC85T10G224L e dimmi se e valida e fino a quando.
```

**Model output**

```text
STEP 1 — MODEL TOOL DECISION
<tool_call>
{"name": "verifica_esenzione", "arguments": {"codice_fiscale": "CNTMRC85T10G224L"}}
</tool_call>
<tool_call>
{"name": "verifica_esenzione", "arguments": {"codice_fiscale": "CNTMRC85T10G224L"}}
</tool_call>

TOOL OBSERVATION
{"codice_esenzione": "E30", "valida": true, "scadenza": "2027-03-31"}

STEP 2 — MODEL RESPONSE
<tool_call>
{"name": "verifica_esenzione", "arguments": {"codice_fiscale": "CNTMRC85T10G224L", "valida": "E30"}}
</tool_call>
```

**Checks**

- step 1: expected verifica_esenzione, actual verifica_esenzione: name_score=1.00
- step 1: arg codice_fiscale='CNTMRC85T10G224L', expected 'CNTMRC85T10G224L': True
- step 1: arg codice_esenzione=None, expected 'E30': False
- step 1: call count expected=1 actual=2 score=0.50
- step 1: strict call count failed: expected 1, got 2; stage score forced to 0
- step 2: contains any of ['valida', 'si', 'sì']: True
- step 2: contains one of ['2027-03-31']: False
- step 2: avoids ['non valida', 'scaduta']: True
- step 2: words 9 <= 60: True
- step 2: does not call another tool after receiving the observation: False
- combined two-step score: (0.000 + 0.525) / 2

#### san_it_ms_disp_prenota — 50.0%

Category: `multi_step` · Language: `it` · Latency: `4.458s` · Output tokens: `189`

**Prompt**

```text
Verifica la disponibilita di una visita oculistica alla struttura Ospedale Sacco il 2026-10-15; dopo il risultato, prenota il primo orario libero per il cf DLLNNA83S45F205W.
```

**Model output**

```text
STEP 1 — MODEL TOOL DECISION
<tool_call>
{"name": "verifica_disponibilita", "arguments": {"prestazione": "visita oculistica", "data": "2026-10-15"}}
</tool_call>
<tool_call>
{"name": "prenota_appuntamento", "arguments": {"prestazione": "visita oculistica", "struttura": "Ospedale Sacco", "data": "2026-10-15"}}
</tool_call>

TOOL OBSERVATION
{"orari_liberi": ["10:00", "16:30"]}

STEP 2 — MODEL RESPONSE
<tool_call>
{"name": "prenota_appuntamento", "arguments": {"prestazione": "visita oculistica", "struttura": "Ospedale Sacco", "data": "2026-10-15", "ora": "10:00", "codice_fiscale": "DLLNNA83S45F205W"}}
</tool_call>
```

**Checks**

- step 1: expected verifica_disponibilita, actual verifica_disponibilita: name_score=1.00
- step 1: arg prestazione='visita oculistica', expected 'visita oculistica': True
- step 1: arg struttura=None, expected 'Ospedale Sacco': False
- step 1: arg data='2026-10-15', expected '2026-10-15': True
- step 1: call count expected=1 actual=2 score=0.50
- step 1: strict call count failed: expected 1, got 2; stage score forced to 0
- step 2: expected prenota_appuntamento, actual prenota_appuntamento: name_score=1.00
- step 2: arg prestazione='visita oculistica', expected 'visita oculistica': True
- step 2: arg struttura='Ospedale Sacco', expected 'Ospedale Sacco': True
- step 2: arg data='2026-10-15', expected '2026-10-15': True
- step 2: arg ora='10:00', expected '10:00': True
- step 2: arg codice_fiscale='DLLNNA83S45F205W', expected 'DLLNNA83S45F205W': True
- step 2: call count expected=1 actual=1 score=1.00
- combined two-step score: (0.000 + 1.000) / 2

#### san_it_ms_esenz_ticket — 50.0%

Category: `multi_step` · Language: `it` · Latency: `2.722s` · Output tokens: `115`

**Prompt**

```text
Verifica l'esenzione E01 per il cf RSSMRA85M01H501Z; se risulta valida, calcola di conseguenza il ticket per una visita fisiatrica.
```

**Model output**

```text
STEP 1 — MODEL TOOL DECISION
<tool_call>
{"name": "verifica_esenzione", "arguments": {"codice_fiscale": "RSSMRA85M01H501Z"}}
</tool_call>
<tool_call>
{"name": "calcola_ticket", "arguments": {"prestazione": "visita fisiatrica", "esente": true}}
</tool_call>

TOOL OBSERVATION
{"valida": true, "codice_esenzione": "E01"}

STEP 2 — MODEL RESPONSE
<tool_call>
{"name": "calcola_ticket", "arguments": {"prestazione": "visita fisiatrica", "esente": true}}
</tool_call>
```

**Checks**

- step 1: expected verifica_esenzione, actual verifica_esenzione: name_score=1.00
- step 1: arg codice_fiscale='RSSMRA85M01H501Z', expected 'RSSMRA85M01H501Z': True
- step 1: arg codice_esenzione=None, expected 'E01': False
- step 1: call count expected=1 actual=2 score=0.50
- step 1: strict call count failed: expected 1, got 2; stage score forced to 0
- step 2: expected calcola_ticket, actual calcola_ticket: name_score=1.00
- step 2: arg prestazione='visita fisiatrica', expected 'visita fisiatrica': True
- step 2: arg esente=True, expected True: True
- step 2: call count expected=1 actual=1 score=1.00
- combined two-step score: (0.000 + 1.000) / 2

#### san_it_ms_disp_prom — 10.0%

Category: `multi_step` · Language: `it` · Latency: `3.444s` · Output tokens: `145`

**Prompt**

```text
Verifica la disponibilita di una ecografia tiroide alla struttura Poliambulatorio Nord il 2026-10-20; poi invia un promemoria via email per la prenotazione che ti verra indicata.
```

**Model output**

```text
STEP 1 — MODEL TOOL DECISION
<tool_call>
{"name": "verifica_disponibilita", "arguments": {"prestazione": "ecografia tiroide", "data": "2026-10-20"}}
</tool_call>
<tool_call>
{"name": "invia_promemoria", "arguments": {"codice_prenotazione": "P0", "canale": "sms"}}
</tool_call>

TOOL OBSERVATION
{"orari_liberi": ["08:15"], "codice_prenotazione": "PRN-6161"}

STEP 2 — MODEL RESPONSE
<tool_call>
{"name": "prenota_appuntamento", "arguments": {"prestazione": "ecografia tiroide", "struttura": "Poliambulatorio Nord", "data": "2026-10-20"}}
</tool_call>
```

**Checks**

- step 1: expected verifica_disponibilita, actual verifica_disponibilita: name_score=1.00
- step 1: arg prestazione='ecografia tiroide', expected 'ecografia tiroide': True
- step 1: arg struttura=None, expected 'Poliambulatorio Nord': False
- step 1: arg data='2026-10-20', expected '2026-10-20': True
- step 1: call count expected=1 actual=2 score=0.50
- step 1: strict call count failed: expected 1, got 2; stage score forced to 0
- step 2: expected invia_promemoria, actual prenota_appuntamento: name_score=0.00
- step 2: arg codice_prenotazione=None, expected 'PRN-6161': False
- step 2: arg canale=None, expected 'email': False
- step 2: extra arguments (not penalized): ['data', 'prestazione', 'struttura']
- step 2: call count expected=1 actual=1 score=1.00
- combined two-step score: (0.000 + 0.200) / 2

#### san_it_ms_ricetta_prenota — 14.0%

Category: `multi_step` · Language: `it` · Latency: `2.887s` · Output tokens: `122`

**Prompt**

```text
Consulta la ricetta RCE-9012; dopo il risultato, prenota la prestazione indicata nella ricetta alla struttura Ospedale Maggiore il 2026-11-10 alle 12:00 per il cf VRDLGI90A41F205K.
```

**Model output**

```text
STEP 1 — MODEL TOOL DECISION
<tool_call>
{"name": "consulta_ricetta", "arguments": {"numero_ricetta": "RCE-9012"}}
</tool_call>
<tool_call>
{"name": "cerca_prestazione", "arguments": {"nome": "Ospedale Maggiore", "prestazione": "visita"}}
</tool_call>

TOOL OBSERVATION
{"numero_ricetta": "RCE-9012", "prestazione": "visita pneumologica"}

STEP 2 — MODEL RESPONSE
<tool_call>
{"name": "cerca_prestazione", "arguments": {"nome": "Ospedale Maggiore", "prestazione": "visita pneumologica"}}
</tool_call>
```

**Checks**

- step 1: expected consulta_ricetta, actual consulta_ricetta: name_score=1.00
- step 1: arg numero_ricetta='RCE-9012', expected 'RCE-9012': True
- step 1: call count expected=1 actual=2 score=0.50
- step 1: strict call count failed: expected 1, got 2; stage score forced to 0
- step 2: expected prenota_appuntamento, actual cerca_prestazione: name_score=0.00
- step 2: arg prestazione='visita pneumologica', expected 'visita pneumologica': True
- step 2: arg struttura=None, expected 'Ospedale Maggiore': False
- step 2: arg data=None, expected '2026-11-10': False
- step 2: arg ora=None, expected '12:00': False
- step 2: arg codice_fiscale=None, expected 'VRDLGI90A41F205K': False
- step 2: extra arguments (not penalized): ['nome']
- step 2: call count expected=1 actual=1 score=1.00
- combined two-step score: (0.000 + 0.280) / 2
