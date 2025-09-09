# %% [markdown]
# # **Capitolo 4**

# %%
# STEP 0 — Fix parsing: rileva separatore e promuovi header se necessario
import re, io, csv
import pandas as pd
from pathlib import Path

# === Percorsi (aggiorna la cartella)
PATH_IN  = "dataset.csv"   # se stai eseguendo nella stessa cartella
OUT_DIR  = Path(r"C:\Users\Jacopo\Tesi_Fintech\dataset_pulito")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PATH_OUT = OUT_DIR / "_dataset_semicolon.csv"

# === 1) Tenta di rilevare il separatore
delims = [",",";","\t","|"]
sample = ""
with open(PATH_IN, "r", encoding="utf-8", errors="ignore") as f:
    for _ in range(10):
        try:
            sample += next(f)
        except StopIteration:
            break

# Conta i separatori più probabili nella prima manciata di righe
counts = {d: sample.count(d) for d in delims}
best_sep = max(counts, key=counts.get)
if counts[best_sep] == 0:
    # fallback tipico per Google Forms in italiano: ';'
    best_sep = ";"

print("Separator candidate counts:", counts, "-> chosen:", best_sep)

# === 2) Leggi una prima volta senza header (per capire se la prima riga è header testuale)
df_try = pd.read_csv(PATH_IN, sep=best_sep, engine="python", header=None)
print("Initial shape (no header):", df_try.shape)

# euristica: se la prima riga contiene tante stringhe lunghe e parole chiave del tuo questionario, è un header
KEYWORDS = [
    "fascia d'età","occupazione","titolo di studio","fonte di reddito",
    "principale","fornitore","conti correnti","IBAN",
    "banche TRADIZIONALI","Fiducia","Sicurezza","Costi","Commissioni",
    "Innovazione","Tecnologia","Facilità d'uso","Servizio Clienti",
    "FINTECH","servizi/app Fintech","Importanti","importanza",
    "Assenza di costi fissi","reputazione","apertura",
    "filiale fisica","app mobile","servizi innovativi","Promozioni","cashback",
    "prossimi 2-3 anni","gestire la maggior parte"
]
kw_regex = re.compile("|".join([re.escape(k) for k in KEYWORDS]), re.IGNORECASE)

def looks_like_header(row_vals):
    vals = [str(x or "").strip() for x in row_vals]
    nonnum = sum(1 for v in vals if not re.fullmatch(r"-?\d+(?:[.,]\d+)?", v))
    hits = sum(1 for v in vals if kw_regex.search(v))
    len_ok = sum(1 for v in vals if 1 <= len(v) <= 120)
    score = hits*2 + nonnum + len_ok*0.1
    return score >= max(6, df_try.shape[1]*0.5)

use_header = looks_like_header(df_try.iloc[0].tolist())
print("Promote first row as header?", use_header)

if use_header:
    cols = [str(x).strip() for x in df_try.iloc[0]]
    df = df_try.iloc[1:].reset_index(drop=True)
    df.columns = cols
else:
    df = pd.read_csv(PATH_IN, sep=best_sep, engine="python", header=0)
    if df.shape[1] == 1 and best_sep != ";":
        df = pd.read_csv(PATH_IN, sep=";", engine="python", header=0)
    if df.shape[1] == 1:
        df = pd.read_csv(PATH_IN, sep=best_sep, engine="python", header=None)
        cols = [f"col_{i+1}" for i in range(df.shape[1])]
        df.columns = cols

print("Final shape:", df.shape)
print("Column sample:", list(df.columns)[:8])

df.to_csv(PATH_OUT, index=False, encoding="utf-8-sig")
print("Saved parsed:", PATH_OUT)

# %% [markdown]
# ## STEP 1 — Setup & Profilo dati

# %%
import re, numpy as np, pandas as pd
from pathlib import Path

OUT_DIR = Path(r"C:\Users\Jacopo\Tesi_Fintech\dataset_pulito")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PATH_IN = OUT_DIR / "_dataset_semicolon.csv"   # <— usa il file riparato
df0 = pd.read_csv(PATH_IN)
print("Loaded:", PATH_IN, "shape:", df0.shape)

# === 1) Trova la riga che contiene i VERI header (basato sui testi del questionario)
KEYWORDS = [
    "fascia d'età", "occupazione", "titolo di studio", "fonte di reddito",
    "principale fornitore", "conti correnti", "IBAN",
    "banche TRADIZIONALI", "Fiducia e Sicurezza", "Costi e Commissioni",
    "Innovazione e Tecnologia", "Facilità d'uso", "Qualità del Servizio Clienti",
    "FINTECH", "servizi/app Fintech utilizzi", "Importanti", "importanza",
    "Assenza di costi fissi", "Solidità e reputazione", "velocità di apertura",
    "filiale fisica", "app mobile", "servizi innovativi", "Promozioni e cashback",
    "prossimi 2-3 anni", "gestire la maggior parte delle tue finanze"
]
kw_regex = re.compile("|".join([re.escape(k) for k in KEYWORDS]), re.IGNORECASE)

def header_score(row_vals):
    vals = [str(x) for x in row_vals]
    hits = sum(1 for v in vals if kw_regex.search(v or ""))
    nonnum = sum(1 for v in vals if not re.fullmatch(r"-?\d+(?:\.\d+)?", v or ""))
    return hits*2 + nonnum

candidate_rows = min(5, len(df0))
scores = [(i, header_score(df0.iloc[i].tolist())) for i in range(candidate_rows)]
scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
best_row, best_score = scores_sorted[0]
print("Header candidate row:", best_row, "score:", best_score, "scores:", scores_sorted)

# accetta sia "Column\d+" sia "col_\d+"
generic_cols = all(re.fullmatch(r"(?:Column\d+|col_\d+)", str(c)) for c in df0.columns)
if generic_cols and best_score >= max(6, len(df0.columns)*0.15):
    new_cols = [str(x).strip() for x in df0.iloc[best_row]]
    seen, safe_cols = set(), []
    for c in new_cols:
        base = re.sub(r"\s+", " ", c or "Colonna")
        base = base.replace(":", "").replace(";", "").replace(",", "")
        key = base
        k = 1
        while key in seen:
            k += 1
            key = f"{base} ({k})"
        seen.add(key)
        safe_cols.append(key)
    df = df0.iloc[best_row+1:].reset_index(drop=True)
    df.columns = safe_cols
    print("Promossa riga", best_row, "a header. Colonne:", len(df.columns))
else:
    df = df0.copy()
    print("Mantengo header esistenti.")

# === 2) Normalizza stringhe
for c in df.columns:
    if df[c].dtype == object:
        df[c] = df[c].astype(str).str.strip()

# === 3) Mappa colonne -> etichette canoniche
PAT = {
    # Demografia
    "age_band": r"fascia.*et[àa]|18\s*[-–]\s*21|22\s*[-–]\s*25|26\s*[-–]\s*30",
    "occupation": r"\boccupazione\b|lavoratore|studente",
    "education": r"(titolo.*studio|laurea|diploma|master)",
    "income_source": r"(fonte.*reddito|sostentamento|supporto.*familiare|part-?time|full-?time|borsa.*studio)",
    # Struttura conto/fornitore
    "primary_provider": r"(principale).*(fornitore|istituto|banc[ao])|dove accrediti lo stipendio",
    "accounts_count": r"(conti).*(correnti|iban)|quanti conti",
    # Valutazioni TRAD
    "trad_trust": r"(tradizional).*(fiducia|sicurezza)",
    "trad_fees": r"(tradizional).*(costi|commissioni)",
    "trad_innov": r"(tradizional).*(innovazione|tecnologia)",
    "trad_ux": r"(tradizional).*(facilit[aà].*uso|app|sito|usabilit)",
    "trad_service": r"(tradizional).*(servizio|assistenza|filiale|telefonico)",
    # Uso Fintech (tipi di servizio) — multirisposta
    "fintech_services_multi": r"(quali).*(servizi|app).*(fintech).*(utilizzi)",
    # Valutazioni FINTECH
    "fin_trust": r"(fintech).*(fiducia|sicurezza)",
    "fin_fees": r"(fintech).*(costi|commissioni)",
    "fin_ux": r"(fintech).*(facilit[aà].*uso|app|sito|usabilit)",
    "fin_service": r"(fintech).*(servizio|assistenza|chat|email)",
    # Importanza fattori
    "imp_no_fees": r"(assenza).*(costi).*fissi|canone.*zero",
    "imp_brand": r"(solidit[aà]|reputazione).*(brand)",
    "imp_onboarding": r"(facilit[aà]|velocit[aà]).*(apertura)",
    "imp_branch": r"(filiale).*fisica",
    "imp_app": r"(app).*(qualit[aà]|velocit[aà]|design)",
    "imp_innovation": r"(servizi).*(innovativi|criptovalute|trading|basso.*costo)",
    "imp_cashback": r"(promozioni|cashback)",
    # Prospettiva 2–3 anni
    "future_pref": r"(prossimi).*(2-?3).*(anni)|maggior.*finanze|prevedi.*gestire"
}

def first_match(colnames, pattern):
    for c in colnames:
        if re.search(pattern, c, flags=re.IGNORECASE):
            return c
    return None

rename_map = {}
for key, pat in PAT.items():
    c = first_match(df.columns, pat)
    if c: rename_map[c] = key

# Se Column26/col_26 è nel df ed è la lista provider-app, assegniamola se non già mappata
if ("Column26" in df.columns or "col_26" in df.columns) and "fintech_services_multi" not in rename_map.values():
    if "Column26" in df.columns:
        rename_map["Column26"] = "fintech_services_multi"
    else:
        rename_map["col_26"] = "fintech_services_multi"

df = df.rename(columns=rename_map)

print("\nMappatura colonne (canonico <= originale):")
for orig, new in rename_map.items():
    print(f"- {new:>22} <= {orig}")

# === 4) Salva
OUT_PARSED = OUT_DIR / "_dataset_parsed_labeled.csv"
df.to_csv(OUT_PARSED, index=False, encoding="utf-8-sig")
print("\nSalvato:", OUT_PARSED)

# %% [markdown]
# ## STEP 2 — Pulizia & Normalizzazione

# %%
# STEP 2 — FIX + normalizzazione Likert + indici (senza explode in-place)
import re, numpy as np, pandas as pd
from pathlib import Path

BASE = Path(r"C:\Users\Jacopo\Tesi_Fintech\dataset_pulito")
LAB  = BASE / "_dataset_parsed_labeled.csv"    # risultato Step 1
SEMICSV = BASE / "_dataset_semicolon.csv"      # file con header originali (ci serve per cercare testi)
OUT  = BASE / "_dataset_clean.csv"

df = pd.read_csv(LAB)
print("Loaded labeled:", LAB, "shape:", df.shape)

# ---------- 1) FIX ETICHETTE ----------
if "fin_service" not in df.columns:
    df_raw = pd.read_csv(SEMICSV)
    fin_serv_col = None
    for c in df_raw.columns:
        s = str(c).lower()
        if ("fintech" in s) and ("qualità" in s or "qualita" in s) and ("servizio clienti" in s or "assistenza" in s or "chat" in s or "email" in s):
            fin_serv_col = c
            break
    if fin_serv_col and fin_serv_col in df.columns and "imp_app" in df.columns and fin_serv_col == "imp_app":
        df = df.rename(columns={"imp_app": "fin_service"})
        print("Rinominata 'imp_app' -> 'fin_service' (era la qualità del servizio clienti FINTECH).")
    elif "imp_app" in df.columns:
        df["fin_service"] = df["imp_app"]
        print("Creato 'fin_service' copiando da 'imp_app' (fallback).")

if "imp_app" not in df.columns:
    df_raw = pd.read_csv(SEMICSV)
    imp_app_guess = None
    for c in df_raw.columns:
        s = str(c).lower()
        if ("importante" in s or "importanza" in s) and ("app" in s) and ("mobile" in s):
            imp_app_guess = c
            break
    if imp_app_guess and imp_app_guess in df.columns:
        df = df.rename(columns={imp_app_guess: "imp_app"})
        print("Rinominata colonna IMPORTANZA app -> 'imp_app'.")
    else:
        print("ATTENZIONE: non ho trovato la colonna 'imp_app' (IMPORTANZA app mobile). Proseguo senza di essa.")

# ---------- 2) Normalizza testi / spazi ----------
for c in df.columns:
    if df[c].dtype == object:
        df[c] = df[c].astype(str).str.strip()

# ---------- 3) Likert 1–5 ----------
LIKERT_MAP = [
    (r"^(?:1)(?:\D|$)|molto\s*neg", 1),
    (r"^(?:2)(?:\D|$)|poco\s*(?:pos|imp)", 2),
    (r"^(?:3)(?:\D|$)|neutro|n[ée]", 3),
    (r"^(?:4)(?:\D|$)|abbastanza|piuttosto", 4),
    (r"^(?:5)(?:\D|$)|molto\s*pos|estremamente\s*imp|molto\s*imp", 5),
    (r"strongly\s*disagree", 1),
    (r"\bdisagree\b", 2),
    (r"\bneutral\b", 3),
    (r"\bagree\b", 4),
    (r"strongly\s*agree", 5),
]
def to_likert_safe(s):
    s = s.astype(str).str.lower().str.strip()
    out = pd.to_numeric(s, errors="coerce")
    if out.isna().all():
        out = pd.Series([np.nan]*len(s))
    for pat, val in LIKERT_MAP:
        mask = s.str.contains(pat, regex=True, na=False)
        out = out.where(~mask, val)
    return pd.to_numeric(out, errors="coerce").clip(1,5)

LIKERT_COLS = [
    "trad_trust","trad_fees","trad_innov","trad_ux","trad_service",
    "fin_trust","fin_fees","fin_ux","fin_service",
    "imp_no_fees","imp_brand","imp_onboarding","imp_branch","imp_app","imp_innovation","imp_cashback"
]
for c in [x for x in LIKERT_COLS if x in df.columns]:
    df[c] = to_likert_safe(df[c])

# ---------- 4) Fasce d’età → coorte ----------
def normalize_age_band(x:str):
    x = str(x).lower()
    if re.search(r"18\s*[-–]\s*21", x): return "18-21"
    if re.search(r"22\s*[-–]\s*25", x): return "22-25"
    if re.search(r"26\s*[-–]\s*30", x): return "26-30"
    return "NA"
if "age_band" in df.columns:
    df["age_band_norm"] = df["age_band"].map(normalize_age_band)
else:
    df["age_band_norm"] = "NA"

def band_to_cohort(b):
    if b in ["18-21","22-25"]: return "Gen Z 18-25"
    if b == "26-30": return "26-30 (Z/Mix)"
    return "Altro/NA"
df["coorte"] = df["age_band_norm"].map(band_to_cohort)

# ---------- 5) (Niente explode qui) ----------
# Le dummies dei servizi Fintech verranno create in 4A in modo robusto.

# ---------- 6) Indici compositi (medie 1–5) ----------
def mean_if_any(row, cols):
    vals = [row[c] for c in cols if c in row.index]
    vals = [v for v in vals if pd.notna(v)]
    return np.mean(vals) if len(vals) else np.nan

df["IDX_TRAD"] = df.apply(lambda r: mean_if_any(r, ["trad_trust","trad_fees","trad_innov","trad_ux","trad_service"]), axis=1)
df["IDX_FIN"]  = df.apply(lambda r: mean_if_any(r, ["fin_trust","fin_fees","fin_ux","fin_service"]), axis=1)
df["IDX_IMPORTANCE"] = df.apply(lambda r: mean_if_any(
    r, ["imp_no_fees","imp_brand","imp_onboarding","imp_branch","imp_app","imp_innovation","imp_cashback"]), axis=1)

print("\nIndici (media 1–5) — head():")
print(df[["IDX_TRAD","IDX_FIN","IDX_IMPORTANCE"]].head())
print("\nCoverage indici (non-NA counts):")
print(df[["IDX_TRAD","IDX_FIN","IDX_IMPORTANCE"]].notna().sum())

# ---------- 7) Preferenze future ----------
if "future_pref" in df.columns:
    s = df["future_pref"].astype(str).str.lower()
    def map_future(x):
        if "banca tradizionale" in x: return "Tradizionale"
        if "neobanca" in x or "fintech" in x: return "Fintech"
        if "mix" in x: return "Mix"
        if "non saprei" in x: return "NS/NA"
        return "Altro/NA"
    df["future_pref_norm"] = s.map(map_future)
    print("\nDistribuzione 'future_pref_norm' (FIX):")
    print(df["future_pref_norm"].value_counts(dropna=False))

# ---------- 8) Salva ----------
df.to_csv(OUT, index=False, encoding="utf-8-sig")
print("\nSalvato CLEAN (FIX):", OUT)

# %%
# STEP 2B — correzione imp_app e ricalcolo indici (ridotto; mantiene approccio senza explode)
import re, numpy as np, pandas as pd
from pathlib import Path

BASE = Path(r"C:\Users\Jacopo\Tesi_Fintech\dataset_pulito")
LAB  = BASE / "_dataset_parsed_labeled.csv"
SEMICSV = BASE / "_dataset_semicolon.csv"
OUT  = BASE / "_dataset_clean.csv"

df = pd.read_csv(LAB)
df_raw = pd.read_csv(SEMICSV)
print("Loaded:", LAB, df.shape, "| Raw headers:", df_raw.shape)

# 1) Heuristic imp_app dai raw headers
imp_app_guess = None
for c in df_raw.columns:
    s = str(c).lower()
    if ("quanto sono importanti" in s or "importante" in s or "importanza" in s) and ("app" in s):
        imp_app_guess = c
        break
print("Heuristic match for imp_app column in raw headers:", imp_app_guess)

if imp_app_guess and imp_app_guess in df.columns:
    df["imp_app"] = df[imp_app_guess]
else:
    print("ATTENZIONE: non ho trovato una colonna IMPORTANZA 'app mobile' nei raw headers; mantengo la precedente 'imp_app'.")

# 2) fin_service dai raw headers se necessario
if "fin_service" not in df.columns or df["fin_service"].isna().all():
    fin_serv_col = None
    for c in df_raw.columns:
        s = str(c).lower()
        if ("fintech" in s) and ("servizio clienti" in s or "assistenza" in s or "chat" in s or "email" in s):
            fin_serv_col = c
            break
    if fin_serv_col and fin_serv_col in df.columns:
        df["fin_service"] = df[fin_serv_col]
        print("Creato 'fin_service' dai raw headers:", fin_serv_col)
    else:
        print("ATTENZIONE: non trovo una colonna valutazione FINTECH 'servizio clienti' nei raw headers; tengo la versione esistente.")

# 3) Normalizza Likert + indici
LIKERT_MAP = [
    (r"^(?:1)(?:\D|$)|molto\s*neg", 1),
    (r"^(?:2)(?:\D|$)|poco\s*(?:pos|imp)", 2),
    (r"^(?:3)(?:\D|$)|neutro|n[ée]", 3),
    (r"^(?:4)(?:\D|$)|abbastanza|piuttosto", 4),
    (r"^(?:5)(?:\D|$)|molto\s*pos|estremamente\s*imp|molto\s*imp", 5),
    (r"strongly\s*disagree", 1),
    (r"\bdisagree\b", 2),
    (r"\bneutral\b", 3),
    (r"\bagree\b", 4),
    (r"strongly\s*agree", 5),
]
def to_likert_safe(s):
    s = s.astype(str).str.lower().str.strip()
    out = pd.to_numeric(s, errors="coerce")
    if out.isna().all():
        out = pd.Series([np.nan]*len(s))
    for pat, val in LIKERT_MAP:
        mask = s.str.contains(pat, regex=True, na=False)
        out = out.where(~mask, val)
    return pd.to_numeric(out, errors="coerce").clip(1,5)

LIKERT_COLS = [
    "trad_trust","trad_fees","trad_innov","trad_ux","trad_service",
    "fin_trust","fin_fees","fin_ux","fin_service",
    "imp_no_fees","imp_brand","imp_onboarding","imp_branch","imp_app","imp_innovation","imp_cashback"
]
for c in [x for x in LIKERT_COLS if x in df.columns]:
    df[c] = to_likert_safe(df[c])

def mean_if_any(row, cols):
    vals = [row[c] for c in cols if c in row.index]
    vals = [v for v in vals if pd.notna(v)]
    return np.mean(vals) if len(vals) else np.nan

df["IDX_TRAD"] = df.apply(lambda r: mean_if_any(r, ["trad_trust","trad_fees","trad_innov","trad_ux","trad_service"]), axis=1)
df["IDX_FIN"]  = df.apply(lambda r: mean_if_any(r, ["fin_trust","fin_fees","fin_ux","fin_service"]), axis=1)
df["IDX_IMPORTANCE"] = df.apply(lambda r: mean_if_any(
    r, ["imp_no_fees","imp_brand","imp_onboarding","imp_branch","imp_app","imp_innovation","imp_cashback"]), axis=1)

# 4) Ricostruisci fasce/coorti se mancano
if "age_band_norm" not in df.columns and "age_band" in df.columns:
    def normalize_age_band(x:str):
        x = str(x).lower()
        if re.search(r"18\s*[-–]\s*21", x): return "18-21"
        if re.search(r"22\s*[-–]\s*25", x): return "22-25"
        if re.search(r"26\s*[-–]\s*30", x): return "26-30"
        return "NA"
    df["age_band_norm"] = df["age_band"].map(normalize_age_band)

if "coorte" not in df.columns and "age_band_norm" in df.columns:
    def band_to_cohort(b):
        if b in ["18-21","22-25"]: return "Gen Z 18-25"
        if b == "26-30": return "26-30 (Z/Mix)"
        return "Altro/NA"
    df["coorte"] = df["age_band_norm"].map(band_to_cohort)

# 5) Salva pulito
df.to_csv(OUT, index=False, encoding="utf-8-sig")
print("\nSalvato CLEAN (aggiornato):", OUT)

# %%
# STEP 3 (fix) — KPI descrittivi + per fascia/coorte (senza errori di pivot)
import numpy as np, pandas as pd
from pathlib import Path

BASE = Path(r"C:\Users\Jacopo\Tesi_Fintech\dataset_pulito")
IN   = BASE / "_dataset_clean.csv"

df = pd.read_csv(IN)
print("Loaded:", IN, "shape:", df.shape)

# 0) Dedup nomi colonna (tieni la prima occorrenza)
dup_mask = pd.Index(df.columns).duplicated(keep='first')
if dup_mask.any():
    dups = list(pd.Index(df.columns)[dup_mask])
    print("ATTENZIONE: colonne duplicate rimosse:", dups)
    df = df.loc[:, ~pd.Index(df.columns).duplicated(keep='first')]

def save_table(df_, name):
    path = BASE / f"{name}.csv"
    df_.to_csv(path, index=True if isinstance(df_, pd.DataFrame) else False, encoding="utf-8-sig")
    print("Saved:", path)

# A) Distribuzioni base
age_dist = df["age_band_norm"].value_counts(dropna=False).rename("count")
cohort_dist = df["coorte"].value_counts(dropna=False).rename("count")
save_table(age_dist, "kpi_age_dist")
save_table(cohort_dist, "kpi_cohort_dist")
print("\nDistribuzione fasce d'età:\n", age_dist)
print("\nDistribuzione coorti:\n", cohort_dist)

# B) Provider principale (normalizzato)
if "primary_provider" in df.columns:
    s = df["primary_provider"].astype(str).str.lower()
    def map_provider(x):
        if "tradizionale" in x: return "Banca tradizionale"
        if "neobanca" in x or "fintech" in x: return "Neobanca/Fintech"
        if "online" in x: return "Banca online (gruppo trad.)"
        if "posta" in x: return "Poste"
        return "Altro/NA"
    df["primary_provider_norm"] = s.map(map_provider)

    prov_overall = df["primary_provider_norm"].value_counts().to_frame("count")
    prov_overall["share_%"] = (prov_overall["count"]/prov_overall["count"].sum()*100).round(1)
    save_table(prov_overall, "kpi_primary_provider_overall")
    print("\nPrimary provider — overall:\n", prov_overall)

    prov_by_age = (df.pivot_table(index="age_band_norm",
                                  columns="primary_provider_norm",
                                  aggfunc="size", fill_value=0))
    prov_by_age_share = (prov_by_age.div(prov_by_age.sum(axis=1), axis=0)*100).round(1)
    save_table(prov_by_age, "kpi_primary_provider_by_age_counts")
    save_table(prov_by_age_share, "kpi_primary_provider_by_age_share")
    print("\nPrimary provider — per fascia (quote %):\n", prov_by_age_share)

# C) Tipi Fintech — verranno gestiti in 4A (qui opzionale)

# D) Medie valutazioni (1–5) per fascia
trad_cols = [c for c in ["trad_trust","trad_fees","trad_innov","trad_ux","trad_service"] if c in df.columns]
fin_cols  = [c for c in ["fin_trust","fin_fees","fin_ux","fin_service"] if c in df.columns]
imp_cols  = [c for c in ["imp_no_fees","imp_brand","imp_onboarding","imp_branch","imp_app","imp_innovation","imp_cashback"] if c in df.columns]

def mean_sd_table(cols, by=None):
    if not cols: return pd.DataFrame()
    if by:
        m = df.groupby(by)[cols].mean().round(2)
        s = df.groupby(by)[cols].std().round(2)
        return m.add_suffix(" (mean)").join(s.add_suffix(" (sd)"))
    else:
        m = df[cols].mean().to_frame("mean").round(2)
        s = df[cols].std().to_frame("sd").round(2)
        return m.join(s)

trad_by_age = mean_sd_table(trad_cols, by="age_band_norm")
fin_by_age  = mean_sd_table(fin_cols,  by="age_band_norm")
imp_by_age  = mean_sd_table(imp_cols,  by="age_band_norm")
save_table(trad_by_age, "kpi_trad_ratings_by_age")
save_table(fin_by_age,  "kpi_fin_ratings_by_age")
save_table(imp_by_age,  "kpi_importance_by_age")
print("\nMedie TRAD per fascia:\n", trad_by_age)
print("\nMedie FINTECH per fascia:\n", fin_by_age)
print("\nImportanza fattori per fascia:\n", imp_by_age)

# Indici sintetici
idx_by_age = df.groupby("age_band_norm")[["IDX_TRAD","IDX_FIN","IDX_IMPORTANCE"]].mean().round(2)
save_table(idx_by_age, "kpi_indices_by_age")
print("\nIndici sintetici (1–5) per fascia:\n", idx_by_age)

# E) Preferenza futura
if "future_pref" in df.columns:
    s = df["future_pref"].astype(str).str.lower()
    def map_future(x):
        if "banca tradizionale" in x: return "Tradizionale"
        if "neobanca" in x or "fintech" in x: return "Fintech"
        if "mix" in x: return "Mix"
        if "non saprei" in x: return "NS/NA"
        return "Altro/NA"
    df["future_pref_norm"] = s.map(map_future)

    fp_overall = df["future_pref_norm"].value_counts().to_frame("count")
    fp_overall["share_%"] = (fp_overall["count"]/fp_overall["count"].sum()*100).round(1)
    save_table(fp_overall, "kpi_future_pref_overall")
    print("\nPreferenza futura — overall:\n", fp_overall)

    fp_by_age = (df.pivot_table(index="age_band_norm",
                                columns="future_pref_norm",
                                aggfunc="size", fill_value=0))
    fp_by_age_share = (fp_by_age.div(fp_by_age.sum(axis=1), axis=0)*100).round(1)
    save_table(fp_by_age, "kpi_future_pref_by_age_counts")
    save_table(fp_by_age_share, "kpi_future_pref_by_age_share")
    print("\nPreferenza futura — per fascia (quote %):\n", fp_by_age_share)

# F) Indici per preferenza futura
if "future_pref_norm" in df.columns:
    idx_by_future = df.groupby("future_pref_norm")[["IDX_TRAD","IDX_FIN","IDX_IMPORTANCE"]].mean().round(2)
    save_table(idx_by_future, "kpi_indices_by_future_pref")
    print("\nIndici per preferenza futura:\n", idx_by_future)

print("\n[Step 3 OK] CSV salvati in:", BASE)

# %%
# STEP 4A — derivate robuste + grafici (con pattern non-catturanti)
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re

BASE = Path(r"C:\Users\Jacopo\Tesi_Fintech\dataset_pulito")
IN   = BASE / "_dataset_clean.csv"
df   = pd.read_csv(IN)
print("Loaded:", IN, "shape:", df.shape)

# 0) Derivate "norm"
if "primary_provider_norm" not in df.columns and "primary_provider" in df.columns:
    s = df["primary_provider"].astype(str).str.lower()
    def map_provider(x):
        if "tradizionale" in x: return "Banca tradizionale"
        if "neobanca" in x or "fintech" in x: return "Neobanca/Fintech"
        if "online" in x: return "Banca online (gruppo trad.)"
        if "posta" in x: return "Poste"
        return "Altro/NA"
    df["primary_provider_norm"] = s.map(map_provider)
    print("CREATA: primary_provider_norm")

if "future_pref_norm" not in df.columns and "future_pref" in df.columns:
    s = df["future_pref"].astype(str).str.lower()
    def map_future(x):
        if "banca tradizionale" in x: return "Tradizionale"
        if "neobanca" in x or "fintech" in x: return "Fintech"
        if "mix" in x: return "Mix"
        if "non saprei" in x: return "NS/NA"
        return "Altro/NA"
    df["future_pref_norm"] = s.map(map_future)
    print("CREATA: future_pref_norm")

# 1) Tipi Fintech — dummies per riga (no explode in-place)
types_col = "fintech_services_multi"
TYPES = {
    "conti_pagamenti": r"(?:conti|pagamenti)",
    "invest_trading":  r"(?:invest|trading)",
    "risparmio_budgeting": r"(?:risparm|budget)",
    "credito_bnpl":    r"(?:credito|buy now|bnpl|klarna|scalapay|paga in 3)",
    "nessuno":         r"(?:nessuno)"
}
if types_col in df.columns:
    s = (df[types_col].astype(str)
                      .str.lower()
                      .str.replace(r"[|/]", ",", regex=True)
                      .str.replace(";", ",")
                      .str.replace("\t", ","))
    for t, pat in TYPES.items():
        df[f"use_{t}"] = s.str.contains(pat, regex=True, na=False).astype(int)
    print("CREATI dummies: ", [c for c in df.columns if c.startswith("use_")])

    # Tabella long separata (opzionale)
    types_long = (s.str.split(",").explode().str.strip().replace({"": np.nan}).dropna())
    def map_type(x):
        if re.search(TYPES["conti_pagamenti"], x): return "conti_pagamenti"
        if re.search(TYPES["invest_trading"], x):  return "invest_trading"
        if re.search(TYPES["risparmio_budgeting"], x): return "risparmio_budgeting"
        if re.search(TYPES["credito_bnpl"], x):    return "credito_bnpl"
        if re.search(TYPES["nessuno"], x):         return "nessuno"
        return x
    types_long = types_long.map(map_type)
    types_long_df = pd.DataFrame({"fintech_type": types_long})
    OUT_TYPES = BASE / "fintech_types_long.csv"
    types_long_df.to_csv(OUT_TYPES, index=False, encoding="utf-8-sig")
    print("Saved long table:", OUT_TYPES)
else:
    print("ATTENZIONE: colonna 'fintech_services_multi' non presente. Salto i dummies/types.")

# 2) Salva versione arricchita
ENR = BASE / "_dataset_clean_enriched.csv"
df.to_csv(ENR, index=False, encoding="utf-8-sig")
print("Saved enriched:", ENR)

# 3) Grafici (no seaborn, no colori fissati)
OUTIMG = BASE / "img"
OUTIMG.mkdir(exist_ok=True, parents=True)

def savefig(name):
    p = OUTIMG / f"{name}.png"
    plt.tight_layout()
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plot:", p)

# Primary provider — overall
if "primary_provider_norm" in df.columns:
    prov = df["primary_provider_norm"].value_counts().sort_values(ascending=False)
    plt.figure()
    prov.plot(kind="bar")
    plt.title("Primary provider — overall")
    plt.xlabel("")
    plt.ylabel("N")
    savefig("primary_provider_overall")

# Primary provider — 100% stacked per fascia
if "primary_provider_norm" in df.columns and "age_band_norm" in df.columns:
    tab = (df.pivot_table(index="age_band_norm", columns="primary_provider_norm",
                          aggfunc="size", fill_value=0)
             .reindex(["18-21","22-25","26-30"]))
    tab_pct = tab.div(tab.sum(axis=1), axis=0)*100

    plt.figure()
    bottom = np.zeros(len(tab_pct))
    x = np.arange(len(tab_pct.index))
    for col in tab_pct.columns:
        plt.bar(x, tab_pct[col].values, bottom=bottom, label=col)
        bottom += tab_pct[col].values
    plt.xticks(x, tab_pct.index)
    plt.ylabel("%")
    plt.title("Primary provider — per fascia (100% stacked)")
    plt.legend(frameon=False)
    savefig("primary_provider_by_age_100pct")

# Indici sintetici per fascia
if "age_band_norm" in df.columns:
    idx = (df.groupby("age_band_norm")[["IDX_TRAD","IDX_FIN","IDX_IMPORTANCE"]]
             .mean().reindex(["18-21","22-25","26-30"]))
    for col in ["IDX_TRAD","IDX_FIN","IDX_IMPORTANCE"]:
        plt.figure()
        idx[col].plot(kind="bar")
        plt.ylim(1,5)
        plt.title(f"{col} — media per fascia (1–5)")
        plt.xlabel("")
        plt.ylabel("Media (1–5)")
        savefig(f"{col}_by_age")

# Preferenza futura — overall e per fascia
if "future_pref_norm" in df.columns:
    fp = df["future_pref_norm"].value_counts().sort_values(ascending=False)
    plt.figure()
    fp.plot(kind="bar")
    plt.title("Preferenza futura — overall")
    plt.ylabel("N")
    plt.xlabel("")
    savefig("future_pref_overall")

if "future_pref_norm" in df.columns and "age_band_norm" in df.columns:
    fp_tab = (df.pivot_table(index="age_band_norm", columns="future_pref_norm",
                             aggfunc="size", fill_value=0)
                .reindex(["18-21","22-25","26-30"]))
    fp_pct = fp_tab.div(fp_tab.sum(axis=1), axis=0)*100

    plt.figure()
    bottom = np.zeros(len(fp_pct))
    x = np.arange(len(fp_pct.index))
    for col in fp_pct.columns:
        plt.bar(x, fp_pct[col].values, bottom=bottom, label=col)
        bottom += fp_pct[col].values
    plt.xticks(x, fp_pct.index)
    plt.ylabel("%")
    plt.title("Preferenza futura — per fascia (100% stacked)")
    plt.legend(frameon=False)
    savefig("future_pref_by_age_100pct")

# %%
# STEP 4B — Test statistici descrittivi (χ² + ANOVA/Kruskal)
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats

BASE = Path(r"C:\Users\Jacopo\Tesi_Fintech\dataset_pulito")
IN   = BASE / "_dataset_clean_enriched.csv"
df   = pd.read_csv(IN)
print("Loaded enriched:", IN, "shape:", df.shape)

# 1) χ²: age_band_norm × primary_provider_norm
if "age_band_norm" in df.columns and "primary_provider_norm" in df.columns:
    ct = pd.crosstab(df["age_band_norm"], df["primary_provider_norm"])
    chi2, p, dof, exp = stats.chi2_contingency(ct)
    print("\n[CHI2] age_band_norm × primary_provider_norm")
    print("chi2=", round(chi2,2), "dof=", dof, "p=", p)
    print("Crosstab:\n", ct)
else:
    print("\n[CHI2] SKIP: manca age_band_norm o primary_provider_norm")

# 2) ANOVA/Kruskal su indici per fascia
for col in ["IDX_TRAD","IDX_FIN","IDX_IMPORTANCE"]:
    if col in df.columns and "age_band_norm" in df.columns:
        groups = [g.dropna().values for _,g in df.groupby("age_band_norm")[col]]
        try:
            f,pv = stats.f_oneway(*groups)
            print(f"\n[ANOVA F] {col} per fascia -> stat={round(f,2)} p={pv}")
        except Exception:
            h,pv = stats.kruskal(*groups)
            print(f"\n[Kruskal H] {col} per fascia -> stat={round(h,2)} p={pv}")

# %%
## 4C — Effect size + post-hoc per §4.4
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
import statsmodels.stats.multicomp as mc

BASE = Path(r"C:\Users\Jacopo\Tesi_Fintech\dataset_pulito")
IN   = BASE / "_dataset_clean_enriched.csv"
df   = pd.read_csv(IN)

OUT_DIR = BASE / "kpi_effects"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# --- 1) Cramér's V per age x provider ---
ct = pd.crosstab(df["age_band_norm"], df["primary_provider_norm"])
chi2, p, dof, exp = stats.chi2_contingency(ct)
n = ct.to_numpy().sum()
k = min(ct.shape[0]-1, ct.shape[1]-1)
cramers_v = np.sqrt(chi2/(n*k))
print(f"Cramér's V = {cramers_v:.3f}  (chi2={chi2:.2f}, dof={dof}, p={p:.6f})")

# --- 2) Eta-squared (ANOVA) + Tukey HSD per ciascun indice ---
def anova_eta_tukey(col):
    d = df[["age_band_norm", col]].dropna()
    groups = [g[col].values for _, g in d.groupby("age_band_norm")]
    F, pval = stats.f_oneway(*groups)
    k_groups = d["age_band_norm"].nunique()
    n_tot = len(d)
    df_between = k_groups - 1
    df_within  = n_tot - k_groups
    eta2 = (F*df_between)/((F*df_between)+df_within)
    print(f"\n{col}: F={F:.2f}, p={pval:.3e}, eta^2={eta2:.3f}  (n={n_tot})")
    comp = mc.MultiComparison(d[col], d["age_band_norm"])
    tuk = comp.tukeyhsd(alpha=0.05)
    print(tuk.summary())
    res = pd.DataFrame(tuk._results_table.data[1:], columns=tuk._results_table.data[0])
    res.to_csv(OUT_DIR / f"tukey_{col}.csv", index=False, encoding="utf-8-sig")

for col in ["IDX_TRAD","IDX_FIN","IDX_IMPORTANCE"]:
    anova_eta_tukey(col)

print("\n[4C OK] Salvati i post-hoc in:", OUT_DIR)

# %%
# 4D — MNLogit robusto (X float, y int) + RRR + Marginal Effects
import numpy as np, pandas as pd
import statsmodels.api as sm
from pathlib import Path

BASE = Path(r"C:\Users\Jacopo\Tesi_Fintech\dataset_pulito")
IN   = BASE / "_dataset_clean_enriched.csv"
df   = pd.read_csv(IN)
print("Loaded enriched:", IN, "shape:", df.shape)

# 1) Dati e outcome numerico (0=Tradizionale, 1=Mix, 2=Fintech)
data = df[df["future_pref_norm"].isin(["Tradizionale","Mix","Fintech"])].copy()
need = ["IDX_TRAD","IDX_FIN","IDX_IMPORTANCE","age_band_norm","future_pref_norm"]
data = data.dropna(subset=need).copy()

cat = pd.Categorical(data["future_pref_norm"], categories=["Tradizionale","Mix","Fintech"])
y = pd.Series(cat.codes, index=data.index, name="y").astype(int)  # 0,1,2

# 2) Costruisci X: indici + dummies fascia (baseline=26-30), + costante
X = data[["IDX_TRAD","IDX_FIN","IDX_IMPORTANCE"]].copy()

# Ordina le fasce e imposta baseline 26-30
age = pd.Categorical(data["age_band_norm"], categories=["26-30","22-25","18-21"], ordered=True)
age_dummies = pd.get_dummies(age, drop_first=True)  # colonne: '22-25','18-21' (baseline=26-30)
X = pd.concat([X, age_dummies], axis=1)

# Cast forzato a numerico (float); droppa righe con NaN post-cast
for c in X.columns:
    X[c] = pd.to_numeric(X[c], errors="coerce")
mask = X.notna().all(axis=1) & y.notna()
X = X.loc[mask].astype(float)
y = y.loc[mask].astype(int)

# Aggiungi costante
X = sm.add_constant(X, has_constant="add")

print("\nDtypes X (verifica):")
print(X.dtypes)

# 3) Fit MNLogit
mn = sm.MNLogit(y, X).fit(method="newton", maxiter=200, disp=False)
print("\n[MNLogit] Preferenza futura ~ indici + fascia")
print(mn.summary())

# 4) McFadden pseudo-R^2
llf = mn.llf
llnull = mn.llnull
pseudo_r2 = 1 - (llf/llnull)
print(f"\nMcFadden pseudo-R^2: {pseudo_r2:.3f}")

# 5) Relative Risk Ratios (exp(coef))
rrr = np.exp(mn.params)
print("\nRelative Risk Ratios (exp(coef)):")
print(rrr.round(3))

# 6) Marginal effects (AME)
mfx = mn.get_margeff(at="overall")
print("\nMarginal effects (overall):")
print(mfx.summary())

# 7) Salva output “camera-ready”
OUT_COEF = BASE / "mnlogit_coef.csv"
OUT_RRR  = BASE / "mnlogit_rrr.csv"
OUT_MFX  = BASE / "mnlogit_mfx_overall.csv"
mn.params.round(4).to_csv(OUT_COEF, encoding="utf-8-sig")
rrr.round(4).to_csv(OUT_RRR, encoding="utf-8-sig")
pd.DataFrame(mfx.summary().tables[1].data[1:], columns=mfx.summary().tables[1].data[0]).to_csv(OUT_MFX, index=False, encoding="utf-8-sig")
print("\nSaved:", OUT_COEF, "\nSaved:", OUT_RRR, "\nSaved:", OUT_MFX)

# %%
# %%
# STEP 4E_fixVIF_safe — VIF coerente con 4D, con cast numerico duro (float64) e drop NA
import numpy as np, pandas as pd
from pathlib import Path
from statsmodels.stats.outliers_influence import variance_inflation_factor

BASE = Path(r"C:\Users\Jacopo\Tesi_Fintech\dataset_pulito")
IN   = BASE / "_dataset_clean_enriched.csv"
OUT  = BASE / "mnlogit_vif_fix.csv"

df = pd.read_csv(IN)

# Stesso design di 4D: indici + dummies fascia (baseline 26-30 => colonne '18-21' e '22-25')
data = df[df["future_pref_norm"].isin(["Tradizionale","Mix","Fintech"])].copy()
need = ["IDX_TRAD","IDX_FIN","IDX_IMPORTANCE","age_band_norm","future_pref_norm"]
data = data.dropna(subset=need).copy()

age_dum = pd.get_dummies(data["age_band_norm"], drop_first=True)  # '18-21','22-25'
X = pd.concat([data[["IDX_TRAD","IDX_FIN","IDX_IMPORTANCE"]], age_dum], axis=1)

# --- (opzionale) standardizza SOLO gli indici, NON i dummies
Xz = X.copy()
for c in ["IDX_TRAD","IDX_FIN","IDX_IMPORTANCE"]:
    mu, sd = Xz[c].mean(), Xz[c].std(ddof=0)
    if sd == 0 or np.isnan(sd):  # safety
        Xz[c] = 0.0
    else:
        Xz[c] = (Xz[c] - mu) / sd

# 1) Cast duro a float64 colonna-per-colonna
for c in Xz.columns:
    Xz[c] = pd.to_numeric(Xz[c], errors="coerce")

# 2) Drop righe con NaN (statsmodels non gestisce NA nel VIF)
Xz = Xz.dropna(axis=0)

# 3) Rimuovi eventuali colonne con varianza ~0 (singolarità nelle regressioni interne del VIF)
keep_cols = []
for c in Xz.columns:
    if float(Xz[c].std(ddof=0)) > 0:
        keep_cols.append(c)
Xz = Xz[keep_cols]

# 4) Ottenere una matrice numpy *pura* float64 (niente dtypes "nullable")
A = Xz.to_numpy(dtype=np.float64)

# 5) Calcola VIF (una colonna per volta)
vifs = []
for i in range(A.shape[1]):
    vifs.append(variance_inflation_factor(A, i))

vif_df = pd.DataFrame({"feature": Xz.columns, "VIF": vifs}).sort_values("VIF", ascending=False)
print("VIF (design 4D, safe-cast float64):\n", vif_df)

vif_df.to_csv(OUT, index=False, encoding="utf-8-sig")
print("Saved:", OUT)


# %%
# %%
# STEP 4F — Modelli predittivi (RandomForest & Gradient Boosting) su future_pref_norm
import numpy as np, pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

BASE = Path(r"C:\Users\Jacopo\Tesi_Fintech\dataset_pulito")
IN   = BASE / "_dataset_clean_enriched.csv"
OUTD = BASE / "ml_outputs"; OUTD.mkdir(exist_ok=True, parents=True)

df = pd.read_csv(IN)
print("Loaded enriched:", IN, "shape:", df.shape)

# === 1) Target e feature set ===
# Target: future_pref_norm (3 classi). Teniamo Trad/Mix/Fintech.
df = df[df["future_pref_norm"].isin(["Tradizionale","Mix","Fintech"])].copy()

# Feature candidate numeriche
likert_cols = [c for c in [
    "trad_trust","trad_fees","trad_innov","trad_ux","trad_service",
    "fin_trust","fin_fees","fin_ux","fin_service",
    "imp_no_fees","imp_brand","imp_onboarding","imp_branch","imp_app","imp_innovation","imp_cashback"
] if c in df.columns]

index_cols = [c for c in ["IDX_TRAD","IDX_FIN","IDX_IMPORTANCE"] if c in df.columns]

use_cols = [c for c in df.columns if c.startswith("use_")]  # dummies tipi fintech (creati allo step 4A)

# Dummies fascia età (baseline gestita dal modello; qui includiamo tutte)
age_dum = pd.get_dummies(df["age_band_norm"], prefix="age", drop_first=False)

# (Opzionale) Dummies provider principale attuale — se vuoi includerlo come segnale di preferenza
prov_dum = pd.get_dummies(df.get("primary_provider_norm", pd.Series(index=df.index)), prefix="prov", drop_first=False)

# Costruisci X finale (tutto numerico)
X_list = []
for cols in [likert_cols, index_cols, use_cols]:
    if cols: X_list.append(df[cols])
X_list += [age_dum, prov_dum]  # includi anche dummies
X = pd.concat(X_list, axis=1)

# Target codificato 0/1/2 nella solita ordine (Trad base interpretativa)
y_cat = pd.Categorical(df["future_pref_norm"], categories=["Tradizionale","Mix","Fintech"])
y = pd.Series(y_cat.codes, index=df.index).astype(int)

print(f"X shape: {X.shape} | y distribution: {np.bincount(y)} (0=Trad,1=Mix,2=Fintech)")

# === 2) Train/Test split stratificato ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# === 3) Pipeline con imputazione (median) ===
rf = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("clf", RandomForestClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=2,
        random_state=42, n_jobs=-1, class_weight=None
    ))
])

gb = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("clf", HistGradientBoostingClassifier(
        learning_rate=0.06, max_depth=None, max_iter=500,
        min_samples_leaf=20, random_state=42
    ))
])

# === 4) CV (5-fold stratificata) su training ===
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, model in [("RF", rf), ("HGB", gb)]:
    acc = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    f1m = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
    print(f"\n[{name}] CV Accuracy: {acc.mean():.3f} ± {acc.std():.03f} | CV F1-macro: {f1m.mean():.3f} ± {f1m.std():.03f}")

# === 5) Fit su tutto il training e valutazione su test ===
rf.fit(X_train, y_train)
gb.fit(X_train, y_train)

def eval_on_test(name, model):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    print(f"\n[{name}] TEST Accuracy={acc:.3f} | F1-macro={f1m:.3f}")
    print(f"[{name}] Classification report:\n", classification_report(
        y_test, y_pred, target_names=["Trad","Mix","Fintech"])
    )
    cm = confusion_matrix(y_test, y_pred)
    # salva confusion matrix
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"{name} — Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(3)
    plt.xticks(tick_marks, ["Trad","Mix","Fintech"], rotation=45)
    plt.yticks(tick_marks, ["Trad","Mix","Fintech"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.ylabel("True label"); plt.xlabel("Predicted label")
    OUTP = OUTD / f"cm_{name}.png"
    plt.tight_layout(); plt.savefig(OUTP, dpi=150); plt.close()
    print("Saved:", OUTP)
    return y_pred

y_pred_rf = eval_on_test("RF", rf)
y_pred_gb = eval_on_test("HGB", gb)

# === 6) Importanza delle feature ===
# Per RF e HGB (tree-based) possiamo leggere feature_importances_
def get_feature_names(Xdf): return list(Xdf.columns)

feat_names = get_feature_names(X)

def dump_importances(name, model):
    # estrai il classificatore dalla pipeline
    clf = model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        imps = pd.Series(clf.feature_importances_, index=feat_names).sort_values(ascending=False)
        top20 = imps.head(20)
        print(f"\n[{name}] Top-20 feature_importances_:\n", top20.round(4))
        imp_path = OUTD / f"importances_{name}.csv"
        imps.to_csv(imp_path, header=["importance"], encoding="utf-8-sig")
        print("Saved:", imp_path)
        # barplot top-20
        plt.figure(figsize=(8,6))
        top20[::-1].plot(kind="barh")
        plt.title(f"{name} — Top-20 importances")
        plt.tight_layout()
        OUTP = OUTD / f"importances_{name}.png"
        plt.savefig(OUTP, dpi=150); plt.close()
        print("Saved:", OUTP)

dump_importances("RF", rf)
dump_importances("HGB", gb)

# === 7) Permutation importance (più robusta) su test ===
def perm_imp(name, model, n_repeats=10):
    r = permutation_importance(
        model, X_test, y_test, n_repeats=n_repeats, random_state=42, n_jobs=-1
    )
    pi = pd.Series(r.importances_mean, index=feat_names).sort_values(ascending=False)
    top20 = pi.head(20)
    print(f"\n[{name}] Permutation importance (Top-20):\n", top20.round(4))
    pi_path = OUTD / f"perm_importance_{name}.csv"
    pi.to_csv(pi_path, header=["perm_imp_mean"], encoding="utf-8-sig")
    print("Saved:", pi_path)
    # plot
    plt.figure(figsize=(8,6))
    top20[::-1].plot(kind="barh")
    plt.title(f"{name} — Permutation importance (Top-20)")
    plt.tight_layout()
    OUTP = OUTD / f"perm_importance_{name}.png"
    plt.savefig(OUTP, dpi=150); plt.close()
    print("Saved:", OUTP)

perm_imp("RF", rf, n_repeats=20)
perm_imp("HGB", gb, n_repeats=20)

print("\n[4F OK] Output salvati in:", OUTD)


# %%
# %%
# STEP 4G — PCA sugli item (o sugli indici) + MNLogit su componenti ortogonali
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

BASE = Path(r"C:\Users\Jacopo\Tesi_Fintech\dataset_pulito")
IN   = BASE / "_dataset_clean_enriched.csv"
OUTD = BASE / "pca_outputs"; OUTD.mkdir(exist_ok=True, parents=True)

df = pd.read_csv(IN)

# 1) Scegli feature per PCA: qui uso TUTTI gli item Likert (più informativi)
likert_cols = [c for c in [
    "trad_trust","trad_fees","trad_innov","trad_ux","trad_service",
    "fin_trust","fin_fees","fin_ux","fin_service",
    "imp_no_fees","imp_brand","imp_onboarding","imp_branch","imp_app","imp_innovation","imp_cashback"
] if c in df.columns]

d = df.dropna(subset=likert_cols + ["age_band_norm","future_pref_norm"]).copy()

X_items = d[likert_cols].values
X_std   = StandardScaler().fit_transform(X_items)

# 2) PCA: tieni prime 3 componenti (spesso catturano >60% var.)
pca = PCA(n_components=3, random_state=42)
PC  = pca.fit_transform(X_std)

expl_var = pd.Series(pca.explained_variance_ratio_, index=[f"PC{i+1}" for i in range(PC.shape[1])])
print("Explained variance ratio:", expl_var.round(3))
expl_var.to_csv(OUTD / "pca_explained_variance.csv", header=["explained_var_ratio"], encoding="utf-8-sig")

# Salva loadings per interpretazione
loadings = pd.DataFrame(pca.components_.T, index=likert_cols, columns=[f"PC{i+1}" for i in range(PC.shape[1])])
loadings.to_csv(OUTD / "pca_loadings.csv", encoding="utf-8-sig")
print("Saved loadings & variance in:", OUTD)

# 3) MNLogit su PC + fascia
keep_idx = d.index
pc_df = pd.DataFrame(PC, index=keep_idx, columns=[f"PC{i+1}" for i in range(PC.shape[1])])

y_cat = pd.Categorical(d.loc[keep_idx, "future_pref_norm"], categories=["Tradizionale","Mix","Fintech"])
y = pd.Series(y_cat.codes, index=keep_idx).astype(int)

age_dum = pd.get_dummies(d.loc[keep_idx, "age_band_norm"], drop_first=True)  # baseline 26-30
Xp = pd.concat([pc_df, age_dum], axis=1).astype(float)
Xp = sm.add_constant(Xp)

mn_pc = sm.MNLogit(y, Xp).fit(method="newton", maxiter=200, disp=False)
print("\n[MNLogit on PCs] summary")
print(mn_pc.summary())

# Confronto pseudo-R2 con 4D (se vuoi)
pseudo_r2_pc = 1 - (mn_pc.llf / mn_pc.llnull)
print(f"McFadden pseudo-R^2 (PC): {pseudo_r2_pc:.3f}")

mn_pc.params.round(4).to_csv(OUTD / "mnlogit_pc_coef.csv", encoding="utf-8-sig")
np.exp(mn_pc.params).round(4).to_csv(OUTD / "mnlogit_pc_rrr.csv", encoding="utf-8-sig")
print("Saved MNLogit PC coef/RRR in:", OUTD)


# %%
# %%
# STEP 4H_fix — Multinomial Logistic REGOLARIZZATA (L2) con CV, feature names, contrasti vs baseline e RRR
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

BASE = Path(r"C:\Users\Jacopo\Tesi_Fintech\dataset_pulito")
IN   = BASE / "_dataset_clean_enriched.csv"
OUTD = BASE / "ml_outputs"; OUTD.mkdir(exist_ok=True, parents=True)

# --- 1) Dati e feature set
df = pd.read_csv(IN)
df = df[df["future_pref_norm"].isin(["Tradizionale","Mix","Fintech"])].copy()

# costruiamo X: indici + dummies fascia + dummies provider (se presente)
num_cols   = ["IDX_TRAD","IDX_FIN","IDX_IMPORTANCE"]
age_dum    = pd.get_dummies(df["age_band_norm"], drop_first=True)   # baseline=26-30
prov_dum   = pd.get_dummies(df.get("primary_provider_norm", pd.Series(index=df.index)), drop_first=True)

X = pd.concat([df[num_cols], age_dum, prov_dum], axis=1)
# y con ordine fissato
class_order = ["Tradizionale","Mix","Fintech"]
y = pd.Categorical(df["future_pref_norm"], categories=class_order, ordered=True).codes

# --- 2) Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.25, random_state=42
)

# --- 3) Pipeline: standardizza SOLO gli indici (non i dummies)
ct = ColumnTransformer(
    transformers=[("scale_idx", StandardScaler(with_mean=True, with_std=True), num_cols)],
    remainder="passthrough"  # lascia invariati i dummies
)

logitcv = LogisticRegressionCV(
    Cs=20,
    cv=5,
    penalty="l2",
    solver="lbfgs",          # multinomial di default da sklearn 1.5+
    scoring="f1_macro",
    max_iter=1000,
    n_jobs=-1,
    refit=True
)

pipe = Pipeline([("ct", ct), ("clf", logitcv)])

# --- 4) Fit + metriche test
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1m = f1_score(y_test, y_pred, average="macro")
print(f"[LR-CV] TEST Acc={acc:.3f} | F1-macro={f1m:.3f}")
print(classification_report(y_test, y_pred, target_names=["Trad","Mix","Fintech"]))

cm = confusion_matrix(y_test, y_pred)
pd.DataFrame(cm, index=class_order, columns=class_order).to_csv(OUTD / "logreg_cv_confusion_matrix.csv", encoding="utf-8-sig")

# --- 5) Coefficienti
clf = pipe.named_steps["clf"]

# nomi feature post-CT (scaler + passthrough)
feat_names = pipe.named_steps["ct"].get_feature_names_out(X.columns)  # sklearn >=1.0
feat_names = [f.replace("scale_idx__", "") for f in feat_names]       # pulizia prefisso

# coef_ ha una riga per classe nell'ordine clf.classes_
coef_full = pd.DataFrame(clf.coef_, columns=feat_names, index=[f"{c}" for c in clf.classes_])
coef_full.to_csv(OUTD / "logreg_cv_coef_full.csv", encoding="utf-8-sig")
print("Saved:", OUTD / "logreg_cv_coef_full.csv")

# Contrasti vs baseline 'Tradizionale': sottrai la riga della baseline alle altre (log-odds difference)
baseline = "Tradizionale"
if baseline in coef_full.index:
    contr = coef_full.loc[[c for c in coef_full.index if c != baseline]].copy()
    contr = contr.subtract(coef_full.loc[baseline].values, axis=1)
    # rinomina righe in "Classe_vs_Tradizionale"
    contr.index = [f"{c}_vs_{baseline}" for c in contr.index]
    contr.to_csv(OUTD / "logreg_cv_coef_contrasts_vs_trad.csv", encoding="utf-8-sig")
    # Relative Risk Ratios (exp)
    rrr = np.exp(contr)
    rrr.to_csv(OUTD / "logreg_cv_rrr_vs_trad.csv", encoding="utf-8-sig")
    print("Saved:", OUTD / "logreg_cv_coef_contrasts_vs_trad.csv")
    print("Saved:", OUTD / "logreg_cv_rrr_vs_trad.csv")
else:
    print("ATTENZIONE: baseline 'Tradizionale' non trovata in clf.classes_. Coefficienti completi salvati, contrasti saltati.")

# --- 6) Salva anche le C selezionate
C_sel = getattr(clf, "C_", None)
pd.DataFrame({"C_selected": np.atleast_1d(C_sel)}).to_csv(OUTD / "logreg_cv_C_selected.csv", index=False, encoding="utf-8-sig")
print("Saved:", OUTD / "logreg_cv_C_selected.csv")


# %%
# %%
# STEP 4I — Calibration + Brier score (probabilità ben tarate per use-case manageriali)
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

BASE = Path(r"C:\Users\Jacopo\Tesi_Fintech\dataset_pulito")
IN   = BASE / "_dataset_clean_enriched.csv"
OUTD = BASE / "ml_outputs"; OUTD.mkdir(exist_ok=True, parents=True)

df = pd.read_csv(IN)
df = df[df["future_pref_norm"].isin(["Tradizionale","Mix","Fintech"])].copy()

likert_cols = [c for c in [
    "trad_trust","trad_fees","trad_innov","trad_ux","trad_service",
    "fin_trust","fin_fees","fin_ux","fin_service",
    "imp_no_fees","imp_brand","imp_onboarding","imp_branch","imp_app","imp_innovation","imp_cashback"
] if c in df.columns]
use_cols = [c for c in df.columns if c.startswith("use_")]
age_dum  = pd.get_dummies(df["age_band_norm"], drop_first=False)
prov_dum = pd.get_dummies(df.get("primary_provider_norm", pd.Series(index=df.index)), drop_first=False)

X = pd.concat([df[likert_cols], age_dum, prov_dum, df[["IDX_TRAD","IDX_FIN","IDX_IMPORTANCE"]], df[use_cols]], axis=1)
y = pd.Categorical(df["future_pref_norm"], categories=["Tradizionale","Mix","Fintech"]).codes

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

base = RandomForestClassifier(n_estimators=600, min_samples_leaf=2, random_state=42, n_jobs=-1)
cal  = CalibratedClassifierCV(base, method="isotonic", cv=5)
cal.fit(X_train, y_train)

probs = cal.predict_proba(X_test)
# Brier score macro (media sulle classi in OVR)
briers = []
for k in range(3):
    y_bin = (y_test == k).astype(int)
    briers.append(brier_score_loss(y_bin, probs[:,k]))
print("Brier (per classe 0=Trad,1=Mix,2=Fintech):", [round(b,4) for b in briers], "| mean:", round(float(np.mean(briers)),4))

pd.DataFrame(probs, columns=["p_Trad","p_Mix","p_Fintech"]).to_csv(OUTD / "calibrated_probs_test.csv", index=False, encoding="utf-8-sig")
print("Saved calibrated probs:", OUTD / "calibrated_probs_test.csv")


# %%
# %%
# STEP 4J — Partial Dependence (PDP) per HGB sulle 8 feature più importanti (interpretabilità)
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import PartialDependenceDisplay

BASE = Path(r"C:\Users\Jacopo\Tesi_Fintech\dataset_pulito")
IN   = BASE / "_dataset_clean_enriched.csv"
OUTD = BASE / "ml_outputs"; OUTD.mkdir(exist_ok=True, parents=True)

df = pd.read_csv(IN)
df = df[df["future_pref_norm"].isin(["Tradizionale","Mix","Fintech"])].copy()

likert_cols = [c for c in [
    "trad_trust","trad_fees","trad_innov","trad_ux","trad_service",
    "fin_trust","fin_fees","fin_ux","fin_service",
    "imp_no_fees","imp_brand","imp_onboarding","imp_branch","imp_app","imp_innovation","imp_cashback"
] if c in df.columns]
use_cols = [c for c in df.columns if c.startswith("use_")]
age_dum  = pd.get_dummies(df["age_band_norm"], drop_first=False)
prov_dum = pd.get_dummies(df.get("primary_provider_norm", pd.Series(index=df.index)), drop_first=False)

X = pd.concat([df[likert_cols], age_dum, prov_dum, df[["IDX_TRAD","IDX_FIN","IDX_IMPORTANCE"]], df[use_cols]], axis=1)
y = pd.Categorical(df["future_pref_norm"], categories=["Tradizionale","Mix","Fintech"]).codes

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

hgb = HistGradientBoostingClassifier(
    learning_rate=0.06, max_iter=500, min_samples_leaf=20, random_state=42
).fit(X_train, y_train)

# Scorri top-8 feature via importanza Gini (per scegliere PDP)
imps = pd.Series(hgb.feature_importances_, index=X.columns).sort_values(ascending=False)
top8 = list(imps.head(8).index)
print("Top-8 per PDP:", top8)

# PDP su classe "Fintech" (2)
fig, ax = plt.subplots(figsize=(10, 10))
PartialDependenceDisplay.from_estimator(hgb, X_train, top8, target=2, ax=ax)
plt.tight_layout()
OUTP = OUTD / "pdp_hgb_top8.png"
plt.savefig(OUTP, dpi=150); plt.close()
print("Saved PDP:", OUTP)
