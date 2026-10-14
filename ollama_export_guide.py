# ============================================
# OLLAMA EXPORT GUIDE - PYTHON CODE
# Για Llama-2 & DeepSeek Models
# ============================================

# ============================================
# PART 1: MERGE LORA WITH BASE MODEL
# ============================================

# ------------------------------------------
# 1A. LLAMA-2 MERGE
# ------------------------------------------

"""
Αυτός ο κώδικας τρέχει στο Colab ΜΕΤΑ το training.
Merge τα LoRA adapters με το base Llama-2 model.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Configuration
BASE_MODEL = "NousResearch/Llama-2-7b-hf"
LORA_PATH = "./llama2_opus100_qlora/final"  # Ή checkpoint-1000
OUTPUT_PATH = "./llama2_merged"

print("=" * 60)
print("🔄 LLAMA-2: MERGING LORA WITH BASE MODEL")
print("=" * 60)

# Load base model (FP16 για merge)
print("\n1️⃣ Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Load LoRA adapters
print("2️⃣ Loading LoRA adapters...")
model = PeftModel.from_pretrained(base_model, LORA_PATH)

# Merge LoRA weights into base model
print("3️⃣ Merging LoRA weights...")
model = model.merge_and_unload()

# Save merged model
print("4️⃣ Saving merged model...")
model.save_pretrained(OUTPUT_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)

print(f"\n✅ Llama-2 merged model saved to: {OUTPUT_PATH}")
print("=" * 60)


# ------------------------------------------
# 1B. DEEPSEEK MERGE
# ------------------------------------------

"""
Ίδια διαδικασία για DeepSeek
"""

# Configuration
BASE_MODEL = "deepseek-ai/deepseek-llm-7b-base"
LORA_PATH = "./deepseek_opus100_qlora/final"
OUTPUT_PATH = "./deepseek_merged"

print("=" * 60)
print("🔄 DEEPSEEK: MERGING LORA WITH BASE MODEL")
print("=" * 60)

# Load base model
print("\n1️⃣ Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True
)

# Load LoRA adapters
print("2️⃣ Loading LoRA adapters...")
model = PeftModel.from_pretrained(base_model, LORA_PATH)

# Merge
print("3️⃣ Merging LoRA weights...")
model = model.merge_and_unload()

# Save
print("4️⃣ Saving merged model...")
model.save_pretrained(OUTPUT_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)

print(f"\n✅ DeepSeek merged model saved to: {OUTPUT_PATH}")
print("=" * 60)


# ============================================
# PART 2: DOWNLOAD MERGED MODELS
# ============================================

"""
Download τα merged models για local conversion
"""

from google.colab import files
import os

# Zip Llama-2 merged model
print("\n📦 Zipping Llama-2 merged model...")
!zip -r llama2_merged.zip ./llama2_merged/

# Zip DeepSeek merged model
print("📦 Zipping DeepSeek merged model...")
!zip -r deepseek_merged.zip ./deepseek_merged/

# Check sizes
llama2_size = os.path.getsize('llama2_merged.zip') / (1024**3)
deepseek_size = os.path.getsize('deepseek_merged.zip') / (1024**3)

print(f"\n✅ Llama-2 zip: {llama2_size:.2f} GB")
print(f"✅ DeepSeek zip: {deepseek_size:.2f} GB")

# Download
print("\n💾 Downloading Llama-2...")
files.download('llama2_merged.zip')

print("💾 Downloading DeepSeek...")
files.download('deepseek_merged.zip')

print("\n✅ Downloads started!")


# ============================================
# PART 3: CONVERT TO GGUF (LOCAL MACHINE)
# ============================================

"""
Αυτά τα commands τρέχουν στο LOCAL machine σου (όχι Colab).
Χρειάζεσαι llama.cpp installed.

INSTALLATION (Local):
---------------------
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

PYTHON REQUIREMENTS:
--------------------
pip install torch transformers sentencepiece protobuf
"""

# ------------------------------------------
# 3A. LLAMA-2 CONVERSION
# ------------------------------------------

"""
Unzip το llama2_merged.zip και τρέξε:
"""

# Terminal commands (στο local machine):
"""
# Navigate to llama.cpp directory
cd llama.cpp

# Convert Llama-2 to GGUF
python convert.py /path/to/llama2_merged/ \
  --outfile llama2-el-en.gguf \
  --outtype f16

# Quantize (optional - για μικρότερο size)
./quantize llama2-el-en.gguf llama2-el-en-q4_0.gguf q4_0

# Result:
# llama2-el-en.gguf (FP16, ~13GB)
# llama2-el-en-q4_0.gguf (4-bit, ~4GB)
"""

# ------------------------------------------
# 3B. DEEPSEEK CONVERSION
# ------------------------------------------

"""
Unzip το deepseek_merged.zip και τρέξε:
"""

# Terminal commands:
"""
# Convert DeepSeek to GGUF
python convert.py /path/to/deepseek_merged/ \
  --outfile deepseek-el-en.gguf \
  --outtype f16

# Quantize
./quantize deepseek-el-en.gguf deepseek-el-en-q4_0.gguf q4_0

# Result:
# deepseek-el-en.gguf (FP16, ~13GB)
# deepseek-el-en-q4_0.gguf (4-bit, ~4GB)
"""


# ============================================
# PART 4: IMPORT TO OLLAMA
# ============================================

"""
Μετά τη conversion σε GGUF, import στο Ollama
"""

# ------------------------------------------
# 4A. CREATE MODELFILE - LLAMA-2
# ------------------------------------------

"""
Δημιούργησε αρχείο: Modelfile-llama2

Περιεχόμενο:
"""

MODELFILE_LLAMA2 = """
FROM ./llama2-el-en-q4_0.gguf

TEMPLATE \"\"\"{{ if .System }}<|system|>
{{ .System }}<|end|>
{{ end }}{{ if .Prompt }}<|user|>
{{ .Prompt }}<|end|>
<|assistant|>
{{ end }}\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40

SYSTEM \"\"\"You are a translation model trained to translate from Greek to English. 
When given Greek text, provide an accurate English translation.\"\"\"
"""

# Save to file
with open('Modelfile-llama2', 'w') as f:
    f.write(MODELFILE_LLAMA2)

print("✅ Modelfile-llama2 created!")

# ------------------------------------------
# 4B. CREATE MODELFILE - DEEPSEEK
# ------------------------------------------

MODELFILE_DEEPSEEK = """
FROM ./deepseek-el-en-q4_0.gguf

TEMPLATE \"\"\"{{ if .System }}<|system|>
{{ .System }}<|end|>
{{ end }}{{ if .Prompt }}<|user|>
{{ .Prompt }}<|end|>
<|assistant|>
{{ end }}\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40

SYSTEM \"\"\"You are a translation model trained to translate from Greek to English. 
When given Greek text, provide an accurate English translation.\"\"\"
"""

# Save to file
with open('Modelfile-deepseek', 'w') as f:
    f.write(MODELFILE_DEEPSEEK)

print("✅ Modelfile-deepseek created!")

# ------------------------------------------
# 4C. IMPORT TO OLLAMA
# ------------------------------------------

"""
Terminal commands (local machine με Ollama installed):
"""

# Import Llama-2
"""
ollama create llama2-el-en -f Modelfile-llama2
"""

# Import DeepSeek
"""
ollama create deepseek-el-en -f Modelfile-deepseek
"""

# Verify
"""
ollama list
"""

# Expected output:
"""
NAME              ID              SIZE      MODIFIED
llama2-el-en      abc123def...    4.0 GB    2 minutes ago
deepseek-el-en    xyz789ghi...    4.0 GB    1 minute ago
"""


# ============================================
# PART 5: TEST OLLAMA MODELS
# ============================================

"""
Test τα models στο Ollama
"""

# ------------------------------------------
# 5A. TEST LLAMA-2
# ------------------------------------------

# Terminal:
"""
ollama run llama2-el-en "Η Αθήνα είναι η πρωτεύουσα της Ελλάδας."
"""

# Expected output:
"""
Athens is the capital of Greece.
"""

# ------------------------------------------
# 5B. TEST DEEPSEEK
# ------------------------------------------

# Terminal:
"""
ollama run deepseek-el-en "Καλημέρα, πώς είσαι σήμερα;"
"""

# Expected output:
"""
Good morning, how are you today?
"""


# ============================================
# PART 6: PYTHON API USAGE
# ============================================

"""
Χρήση των Ollama models μέσω Python API
"""

import requests
import json

def translate_with_ollama(text, model="llama2-el-en"):
    """
    Translate Greek text to English using Ollama model
    """
    url = "http://localhost:11434/api/generate"
    
    data = {
        "model": model,
        "prompt": text,
        "stream": False
    }
    
    response = requests.post(url, json=data)
    result = response.json()
    
    return result['response']

# Example usage
greek_texts = [
    "Η Αθήνα είναι η πρωτεύουσα της Ελλάδας.",
    "Καλημέρα, πώς είσαι σήμερα;",
    "Το φαγητό ήταν πολύ νόστιμο."
]

print("=" * 60)
print("🧪 TESTING LLAMA-2 VIA OLLAMA API")
print("=" * 60)

for text in greek_texts:
    translation = translate_with_ollama(text, model="llama2-el-en")
    print(f"\nEL: {text}")
    print(f"EN: {translation}")

print("\n" + "=" * 60)
print("🧪 TESTING DEEPSEEK VIA OLLAMA API")
print("=" * 60)

for text in greek_texts:
    translation = translate_with_ollama(text, model="deepseek-el-en")
    print(f"\nEL: {text}")
    print(f"EN: {translation}")


# ============================================
# SUMMARY
# ============================================

print("=" * 60)
print("✅ OLLAMA EXPORT CODE COMPLETE!")
print("=" * 60)
print("\nΑυτός ο κώδικας περιλαμβάνει:")
print("1. ✅ LoRA merge για Llama-2")
print("2. ✅ LoRA merge για DeepSeek")
print("3. ✅ GGUF conversion commands")
print("4. ✅ Ollama import process")
print("5. ✅ Testing & API usage")
print("=" * 60)