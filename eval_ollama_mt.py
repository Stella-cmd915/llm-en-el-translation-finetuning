import os, requests, time
from datasets import load_dataset
from tqdm import tqdm
import evaluate

# ----------- CONFIG -----------
MODEL_NAME   = "mistral-el-en"     # your Ollama model name
DIRECTION    = "EL2EN"             # or "EN2EL"
N_SAMPLES    = 500                 # adjust (200–1000 is common)
MAX_NEW_TOKS = 128
NUM_CTX      = 2048                # keep moderate on M2
TEMPERATURE  = 0
TIMEOUT_S    = 120                 # per request
# ------------------------------

cfg = "el-en" if DIRECTION.upper()=="EL2EN" else "en-el"
ds  = load_dataset("Helsinki-NLP/opus-100", cfg)
val = ds["test"] if "test" in ds else ds["validation"]
val = val.select(range(min(N_SAMPLES, len(val))))

def build_prompt(src):
    instr = ("Μετάφρασε στα αγγλικά το παρακάτω κείμενο."
             if DIRECTION.upper()=="EL2EN"
             else "Μετάφρασε στα ελληνικά το παρακάτω κείμενο.")
    return f"{instr}\n\nΚείμενο:\n{src}\n\nΜετάφραση:"

def ollama_generate(prompt):
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_ctx": NUM_CTX,
                "temperature": TEMPERATURE,
                "num_keep": 0
            }
        },
        timeout=TIMEOUT_S
    )
    r.raise_for_status()
    txt = r.json().get("response","").strip()
    # Heuristic: return only the translated segment after the cue
    if "Μετάφραση:" in txt:
        txt = txt.split("Μετάφραση:", 1)[-1].strip()
    return txt

preds, refs = [], []
for ex in tqdm(val, desc="Generating"):
    el = ex["translation"]["el"].strip()
    en = ex["translation"]["en"].strip()
    src, tgt = (el, en) if DIRECTION.upper()=="EL2EN" else (en, el)
    preds.append(ollama_generate(build_prompt(src)))
    refs.append(tgt)

# Metrics
bleu  = evaluate.load("sacrebleu")
chrf  = evaluate.load("chrf")
rouge = evaluate.load("rouge")

res_bleu  = bleu.compute(predictions=preds, references=[[r] for r in refs])
res_chrf  = chrf.compute(predictions=preds, references=[[r] for r in refs])
res_rouge = rouge.compute(predictions=preds, references=refs)

print("\n=== Results ===")
print(f"sacreBLEU: {res_bleu['score']:.2f}")
print(f"chrF2    : {res_chrf['score']:.2f}")
print(f"ROUGE-1  : {res_rouge['rouge1']*100:.2f}")
print(f"ROUGE-2  : {res_rouge['rouge2']*100:.2f}")
print(f"ROUGE-L  : {res_rouge['rougeL']*100:.2f}")
print(f"ROUGE-Lsum: {res_rouge['rougeLsum']*100:.2f}")

# Optional METEOR (more meaningful for EN targets)
try:
    meteor = evaluate.load("meteor")
    res_meteor = meteor.compute(predictions=preds, references=refs)
    print(f"METEOR   : {res_meteor['meteor']*100:.2f}")
except Exception as e:
    print("METEOR skipped:", e)
