# Ollama Export Guide
## Οδηγίες Μετατροπής σε Ollama Format

---

## 📋 Τι είναι αυτό το guide;

Αυτό το guide εξηγεί πώς να μετατρέψετε τα trained models (Llama-2 και DeepSeek) σε **Ollama format** για local χρήση.

---

##  Τι θα επιτύχετε;

Μετά από αυτό το guide, θα μπορείτε:
- ✅ Να τρέχετε τα models τοπικά με Ollama
- ✅ Να κάνετε μεταφράσεις EL→EN offline
- ✅ Να τα χρησιμοποιείτε μέσω CLI ή Python API

---

## Χρόνος: ~1 ώρα ανά model

---

##  Τι χρειάζεστε;

### Hardware:
- Υπολογιστής με 16GB+ RAM
- 30GB ελεύθερος χώρος

### Software:
-  Python 3.8+
-  llama.cpp
- Ollama

---

## Γρήγορη Επισκόπηση

Η διαδικασία έχει **4 φάσεις**:

```
 Φάση 1: Merge (Colab) → 10-15 λεπτά
 Φάση 2: Download → 10-20 λεπτά  
 Φάση 3: Convert (Local) → 15-30 λεπτά
 Φάση 4: Import (Ollama) → 5 λεπτά
```

---

##  Αναλυτικές Οδηγίες

### Φάση 1: Merge LoRA με Base Model (Colab)

**Πότε:** Αφού τελειώσει το training

**Πού:** Στο Google Colab notebook

**Τι κάνετε:**
1. Ανοίξτε το `ollama_export_guide.py`
2. Copy το **Part 1** (γραμμές 1-80)
3. Paste σε νέο κελί στο Colab
4. Run το κελί

**Αποτέλεσμα:**
- `llama2_merged/` folder (merged Llama-2)
- `deepseek_merged/` folder (merged DeepSeek)

**Χρόνος:** 10-15 λεπτά

---

### Φάση 2: Download Models (Colab)

**Τι κάνετε:**
1. Copy το **Part 2** από `ollama_export_guide.py`
2. Run σε νέο κελί
3. Download τα zip files

**Αποτέλεσμα:**
- `llama2_merged.zip` (~13 GB)
- `deepseek_merged.zip` (~13 GB)

**Χρόνος:** 10-20 λεπτά (ανάλογα με internet)

---

### Φάση 3: Convert σε GGUF (Local Machine)

**Πού:** Στον υπολογιστή σας

#### Βήμα 3.1: Install llama.cpp

**Mac/Linux:**
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make
```

**Windows:**
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build
cmake --build build --config Release
```

#### Βήμα 3.2: Install Python Dependencies

```bash
pip install torch transformers sentencepiece protobuf
```

#### Βήμα 3.3: Unzip Models

```bash
unzip llama2_merged.zip
unzip deepseek_merged.zip
```

#### Βήμα 3.4: Convert Llama-2

```bash
cd llama.cpp

python convert.py /path/to/llama2_merged/ \
  --outfile llama2-el-en.gguf \
  --outtype f16

# Quantize (προαιρετικό - μικρότερο file)
./quantize llama2-el-en.gguf llama2-el-en-q4_0.gguf q4_0
```

#### Βήμα 3.5: Convert DeepSeek

```bash
python convert.py /path/to/deepseek_merged/ \
  --outfile deepseek-el-en.gguf \
  --outtype f16

# Quantize
./quantize deepseek-el-en.gguf deepseek-el-en-q4_0.gguf q4_0
```

**Αποτέλεσμα:**
- `llama2-el-en.gguf` (13 GB)
- `llama2-el-en-q4_0.gguf` (4 GB)  Συνιστάται
- `deepseek-el-en.gguf` (13 GB)
- `deepseek-el-en-q4_0.gguf` (4 GB)  Συνιστάται

**Χρόνος:** 15-30 λεπτά ανά model

---

### Φάση 4: Import στο Ollama (Local)

#### Βήμα 4.1: Install Ollama

Κατεβάστε από: https://ollama.ai/download

#### Βήμα 4.2: Create Modelfile για Llama-2

Δημιουργήστε αρχείο `Modelfile-llama2`:

```
FROM ./llama2-el-en-q4_0.gguf

TEMPLATE """{{ if .System }}<|system|>
{{ .System }}<|end|>
{{ end }}{{ if .Prompt }}<|user|>
{{ .Prompt }}<|end|>
<|assistant|>
{{ end }}"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9

SYSTEM """You are a translation model for Greek to English translation."""
```

#### Βήμα 4.3: Create Modelfile για DeepSeek

Δημιουργήστε αρχείο `Modelfile-deepseek`:

```
FROM ./deepseek-el-en-q4_0.gguf

TEMPLATE """{{ if .System }}<|system|>
{{ .System }}<|end|>
{{ end }}{{ if .Prompt }}<|user|>
{{ .Prompt }}<|end|>
<|assistant|>
{{ end }}"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9

SYSTEM """You are a translation model for Greek to English translation."""
```

#### Βήμα 4.4: Import Models

```bash
# Import Llama-2
ollama create llama2-el-en -f Modelfile-llama2

# Import DeepSeek
ollama create deepseek-el-en -f Modelfile-deepseek

# Verify
ollama list
```

**Αποτέλεσμα:**
```
NAME              SIZE      
llama2-el-en      4.0 GB    
deepseek-el-en    4.0 GB    
```

**Χρόνος:** 5 λεπτά

---

## Testing

### Δοκιμή από CLI

**Llama-2:**
```bash
ollama run llama2-el-en "Η Αθήνα είναι η πρωτεύουσα της Ελλάδας."
```

**Αναμενόμενο αποτέλεσμα:**
```
Athens is the capital of Greece.
```

**DeepSeek:**
```bash
ollama run deepseek-el-en "Καλημέρα, πώς είσαι σήμερα;"
```

**Αναμενόμενο αποτέλεσμα:**
```
Good morning, how are you today?
```

---

### Δοκιμή από Python

```python
import requests
import json

def translate(text, model="llama2-el-en"):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "prompt": text,
        "stream": False
    }
    response = requests.post(url, json=data)
    return response.json()['response']

# Test
text = "Το φαγητό ήταν πολύ νόστιμο."
translation = translate(text, "llama2-el-en")
print(f"EL: {text}")
print(f"EN: {translation}")
```

---

## Μεγέθη Αρχείων

| Τύπος | Llama-2 | DeepSeek |
|-------|---------|----------|
| Merged (FP16) | 13 GB | 13 GB |
| GGUF (FP16) | 13 GB | 13 GB |
| GGUF (Q4_0) | 4 GB  | 4 GB |

**Σύσταση:** Χρησιμοποιήστε Q4_0 για καλύτερη ισορροπία μεγέθους/ποιότητας

---

##  Troubleshooting

### Πρόβλημα: "convert.py failed"
**Λύση:**
```bash
pip install torch transformers sentencepiece protobuf
```

### Πρόβλημα: "Out of memory"
**Λύση:**
- Κλείστε άλλες εφαρμογές
- Χρειάζεστε τουλάχιστον 16GB RAM

### Πρόβλημα: "Ollama model not found"
**Λύση:**
```bash
ollama list  # Ελέγξτε τα διαθέσιμα models
```

### Πρόβλημα: "Conversion πολύ αργή"
**Λύση:**
- Είναι φυσιολογικό (~20-30 λεπτά)
- ΜΗΝ διακόψετε τη διαδικασία

---

## Προηγμένες Ρυθμίσεις

### Αλλαγή Temperature

Για πιο ακριβείς μεταφράσεις:
```
PARAMETER temperature 0.3
```

Για πιο δημιουργικές μεταφράσεις:
```
PARAMETER temperature 0.9
```

### Custom Modelfile

```
FROM ./llama2-el-en-q4_0.gguf

PARAMETER temperature 0.5
PARAMETER top_p 0.85

SYSTEM """Provide accurate Greek to English translations.
Maintain formal tone."""
```

---

##  Πρόσθετοι Πόροι

- **llama.cpp:** https://github.com/ggerganov/llama.cpp
- **Ollama:** https://github.com/ollama/ollama
- **GGUF Format:** https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

---

##  Συμβουλές

1.  Χρησιμοποιήστε Q4_0 quantization
2. Set temperature 0.3-0.5 για μεταφράσεις
3. Test με διάφορα παραδείγματα πριν production
4. Keep models updated με νέα training data

---

##  Υποστήριξη

Για προβλήματα:
1. Ελέγξτε το Troubleshooting section
2. Δείτε το `ollama_export_guide.py` για λεπτομέρειες
3. Συμβουλευτείτε την τεκμηρίωση llama.cpp/Ollama

---

