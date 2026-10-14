# ============================================
# LLAMA-2 FINE-TUNING FOR EL→EN TRANSLATION
# Dataset: OPUS-100 (Helsinki-NLP)
# Method: QLoRA (4-bit quantization)
# ============================================

# ============================================
# ΚΕΛΙ 1: INSTALLATIONS
# ============================================
!pip install -q transformers datasets accelerate bitsandbytes peft evaluate sacrebleu rouge_score nltk

import nltk
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('omw-1.4', quiet=True)

print("✅ Εγκατάσταση ολοκληρώθηκε!")


# ============================================
# ΚΕΛΙ 2: IMPORTS & GPU CHECK
# ============================================
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    default_data_collator
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import evaluate
from tqdm import tqdm
import json

# Έλεγχος GPU
print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
print(f"💾 CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"✅ PyTorch version: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")


# ============================================
# ΚΕΛΙ 3: CONFIGURATION
# ============================================

# Model Configuration
MODEL_NAME = "NousResearch/Llama-2-7b-hf"  # Llama-2 7B (no gating)
DIRECTION = "EL2EN"  # Ελληνικά → Αγγλικά
OUTPUT_DIR = "./llama2_opus100_qlora"

# Dataset Configuration
MAX_TRAIN_SAMPLES = 20000  # Αριθμός training samples
MAX_VAL_SAMPLES = 1000     # Αριθμός validation samples
MAX_SEQ_LENGTH = 256       # Maximum sequence length

# Training Configuration
NUM_EPOCHS = 1
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch = 16
LEARNING_RATE = 2e-4

# LoRA Configuration
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05

# Direction setup
if DIRECTION == "EN2EL":
    source_lang = "en"
    target_lang = "el"
else:  # EL2EN
    source_lang = "el"
    target_lang = "en"

print("✅ Configuration έτοιμο!")
print(f"📦 Model: {MODEL_NAME}")
print(f"🔄 Direction: {source_lang.upper()} → {target_lang.upper()}")
print(f"📊 Training samples: {MAX_TRAIN_SAMPLES}")
print(f"⚡ Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")


# ============================================
# ΚΕΛΙ 4: QUANTIZATION CONFIG
# ============================================

# 4-bit quantization για εξοικονόμηση μνήμης
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print("✅ Quantization config έτοιμο!")


# ============================================
# ΚΕΛΙ 5: MODEL & TOKENIZER LOADING
# ============================================

# NOTE: NousResearch/Llama-2-7b-hf ΔΕΝ χρειάζεται authentication!
# Αν θέλετε να χρησιμοποιήσετε το official meta-llama/Llama-2-7b-hf,
# θα χρειαστείτε HF token και access approval από το Meta.

print("🔄 Φόρτωση Llama-2 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, 
    padding_side='right'
    # Δεν χρειάζεται token για NousResearch
)

# Set pad token (Llama-2 doesn't have one by default)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print("🔄 Φόρτωση Llama-2 model με 4-bit quantization...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
    # Δεν χρειάζεται token για NousResearch
)

# Prepare model για k-bit training
model = prepare_model_for_kbit_training(model)

print("✅ Llama-2 φορτώθηκε επιτυχώς!")
print(f"📊 Model parameters: {model.num_parameters() / 1e9:.2f}B")


# ============================================
# ΚΕΛΙ 6: DATASET LOADING
# ============================================

print("🔄 Φόρτωση OPUS-100 dataset...")

# Load dataset (el-en config)
dataset = load_dataset("Helsinki-NLP/opus-100", "el-en")

print(f"✅ Dataset loaded: {len(dataset['train'])} train samples")
print(f"📊 Direction: {source_lang.upper()} → {target_lang.upper()}")

# Sample
print("\n📝 Sample:")
sample = dataset['train'][0]['translation']
print(f"  EN: {sample['en']}")
print(f"  EL: {sample['el']}")


# ============================================
# ΚΕΛΙ 7: DATA PREPROCESSING
# ============================================

def preprocess_with_padding(examples):
    """
    Preprocessing function που κάνει tokenization με padding.
    Για Causal LM, labels = input_ids με -100 στα pad tokens.
    """
    prompts = []
    
    for translation in examples['translation']:
        source_text = translation[source_lang]
        target_text = translation[target_lang]
        
        # Instruction format: source → target
        full_text = f"Translate {source_lang.upper()}→{target_lang.upper()}: {source_text} = {target_text}"
        prompts.append(full_text)
    
    # Tokenize με padding
    model_inputs = tokenizer(
        prompts,
        max_length=MAX_SEQ_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    # Labels = inputs για causal LM
    model_inputs["labels"] = model_inputs["input_ids"].clone()
    
    # Αντικατάσταση padding tokens με -100 (ignored by loss)
    model_inputs["labels"][model_inputs["labels"] == tokenizer.pad_token_id] = -100
    
    return model_inputs

print("🔄 Preprocessing dataset...")

# Επιλογή υποσύνολου
train_dataset = dataset['train'].select(range(min(MAX_TRAIN_SAMPLES, len(dataset['train']))))
val_dataset = dataset['test'].select(range(min(MAX_VAL_SAMPLES, len(dataset['test']))))

# Preprocessing
train_dataset = train_dataset.map(
    preprocess_with_padding,
    batched=True,
    batch_size=100,
    remove_columns=dataset['train'].column_names
)

val_dataset = val_dataset.map(
    preprocess_with_padding,
    batched=True,
    batch_size=100,
    remove_columns=dataset['test'].column_names
)

# Set format για PyTorch
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

print(f"✅ Preprocessed {len(train_dataset)} training samples")
print(f"✅ Preprocessed {len(val_dataset)} validation samples")


# ============================================
# ΚΕΛΙ 8: LORA CONFIGURATION
# ============================================

# Target modules για Llama-2 architecture
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
)

# Εφαρμογή LoRA στο model
model = get_peft_model(model, lora_config)

# Trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())

print("✅ LoRA Configuration applied!")
print(f"📊 Trainable params: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)")
print(f"📊 All params: {all_params:,}")


# ============================================
# ΚΕΛΙ 9: TRAINING ARGUMENTS
# ============================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    warmup_steps=100,
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    fp16=True,
    optim="paged_adamw_8bit",
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

print("✅ Training arguments configured!")
print(f"⚡ Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"📊 Total training steps: ~{len(train_dataset) // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS) * NUM_EPOCHS}")


# ============================================
# ΚΕΛΙ 10: TRAINER SETUP
# ============================================

# Trainer με default data collator
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=default_data_collator
)

print("✅ Trainer initialized!")
print("🚀 Ready to start training!")


# ============================================
# ΚΕΛΙ 11: START TRAINING
# ============================================

print("=" * 60)
print("🔥 ΕΚΚΙΝΗΣΗ LLAMA-2 TRAINING!")
print("=" * 60)
print(f"📦 Model: {MODEL_NAME}")
print(f"📊 Training samples: {len(train_dataset)}")
print(f"📊 Validation samples: {len(val_dataset)}")
print(f"⚡ Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"🔥 Training steps: ~{len(train_dataset) // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)}")
print(f"⏱️ Estimated time: ~3-4 hours")
print("=" * 60)

# START TRAINING!
trainer.train()

print("\n" + "=" * 60)
print("✅ TRAINING ΟΛΟΚΛΗΡΩΘΗΚΕ!")
print("=" * 60)


# ============================================
# ΚΕΛΙ 12: SAVE MODEL
# ============================================

# Αποθήκευση LoRA adapters
output_path = f"{OUTPUT_DIR}/final"
trainer.model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

print(f"✅ Model saved to: {output_path}")


# ============================================
# ΚΕΛΙ 13: TEST GENERATION (Optional)
# ============================================

# Test translation
test_inputs = [
    "Η Αθήνα είναι η πρωτεύουσα της Ελλάδας.",
    "Καλημέρα, πώς είσαι σήμερα;",
    "Το φαγητό ήταν πολύ νόστιμο."
]

print("\n🧪 Test Translations:")
print("=" * 60)

model.eval()
for i, test_input in enumerate(test_inputs, 1):
    prompt = f"Translate EL→EN: {test_input} ="
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the translation part
    translation = translation.split("=")[-1].strip()
    
    print(f"\n{i}. EL: {test_input}")
    print(f"   EN: {translation}")

print("=" * 60)
print("✅ Testing complete!")