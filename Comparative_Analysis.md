# Σύγκριση Μοντέλων: Mistral 7B, Llama-2 7B & DeepSeek 7B
## Αξιολόγηση για Μετάφραση Ελληνικά-Αγγλικά

**Ημερομηνία:** 30 Νοεμβρίου 2025  
**Dataset:** OPUS-100 (EL-EN)  
**Μέθοδος:** QLoRA (4-bit quantization)

---

## Εισαγωγή

Στο πλαίσιο αυτού του project, εκπαιδεύτηκαν και αξιολογήθηκαν τρία διαφορετικά language models για τη μετάφραση κειμένων από Ελληνικά σε Αγγλικά. Το κάθε μοντέλο έχει τα δικά του χαρακτηριστικά και ιδιαιτερότητες, τα οποία αναλύονται στη συνέχεια.

---

## Τεχνικά Χαρακτηριστικά

Και τα τρία μοντέλα εκπαιδεύτηκαν με τις ίδιες ρυθμίσεις για δίκαιη σύγκριση:

**Κοινές παράμετροι:**
- Training samples: 20,000 ζεύγη μεταφράσεων
- Validation samples: 1,000
- Batch size: 4 (effective 16 με gradient accumulation)
- Learning rate: 2e-4
- LoRA rank: 32
- Epochs: 1
- GPU: Tesla T4 (16GB)

**Διαφορές στα μοντέλα:**

*Mistral 7B:*
- Χρησιμοποιεί Grouped Query Attention (πιο γρήγορο)
- Context window: 8K tokens (διπλάσιο από τα άλλα)
- Πλήρως ανοιχτό (δεν χρειάζεται authentication)

*Llama-2 7B:*
- Το πιο ευρέως χρησιμοποιημένο από την κοινότητα
- Απαιτεί Meta license approval
- Πολλή τεκμηρίωση και community support

*DeepSeek 7B:*
- Βελτιστοποιημένο για reasoning tasks
- Μεγαλύτερο vocabulary (~100K tokens)
- Ανοιχτό, αλλά λιγότερο γνωστό

---

## Αποτελέσματα Training

Το training ήταν επιτυχές και για τα τρία μοντέλα. Παρατηρήθηκε σταθερή μείωση του loss χωρίς σημάδια overfitting:

**Training Loss:**
- Αρχική τιμή: ~1.15 (και στα τρία)
- Τελική τιμή: ~1.08 (και στα τρία)
- Μείωση: ~6%

**Validation Loss:**
- Αρχική τιμή: ~1.19
- Τελική τιμή: ~1.13
- Σταθεροποίηση: Ναι

Στην πράξη, αυτό σημαίνει ότι τα μοντέλα έμαθαν καλά χωρίς να "απομνημονεύσουν" τα training data.

---

## Ποιοτική Αξιολόγηση

Δοκιμάστηκαν όλα τα μοντέλα με τα ίδια παραδείγματα. Τα αποτελέσματα ήταν εντυπωσιακά παρόμοια:

**Παράδειγμα 1:**
- Είσοδος: "Η Αθήνα είναι η πρωτεύουσα της Ελλάδας."
- Mistral: "Athens is the capital of Greece."
- Llama-2: "Athens is the capital of Greece."
- DeepSeek: "Athens is the capital of Greece."

Όλα τέλεια. Απλή πρόταση, προφανώς.

**Παράδειγμα 2:**
- Είσοδος: "Καλημέρα, πώς είσαι σήμερα;"
- Mistral: "Good morning, how are you today?"
- Llama-2: "Good morning, how are you today?"
- DeepSeek: "Good morning, how are you today?"

Πάλι ταυτόσημα αποτελέσματα.

**Παράδειγμα 3:**
- Είσοδος: "Το φαγητό ήταν πολύ νόστιμο."
- Mistral: "The food was very delicious."
- Llama-2: "The food was very tasty."
- DeepSeek: "The food was very delicious."

Εδώ υπάρχει μια μικρή διαφορά - το Llama-2 διάλεξε "tasty" ενώ τα άλλα "delicious". Και οι δύο επιλογές είναι σωστές.

---

## Ταχύτητα & Απόδοση

Από τη χρήση τους, παρατηρήθηκαν μικρές διαφορές στην ταχύτητα:

**Mistral 7B:**
Φαίνεται λίγο πιο γρήγορο, πιθανώς λόγω του Grouped Query Attention. Στη χρήση δεν είναι τεράστια η διαφορά, αλλά σε μεγάλους όγκους κειμένου θα φανεί.

**Llama-2 7B:**
Σταθερή ταχύτητα, χωρίς εκπλήξεις. Δουλεύει όπως περιμένεις.

**DeepSeek 7B:**
Παρόμοια απόδοση με το Llama-2. Το μεγαλύτερο vocabulary μπορεί να βοηθήσει σε πιο σύνθετα κείμενα.

---

## Πρακτικές Παρατηρήσεις

**Ευκολία χρήσης:**

*Mistral:*
Το πιο εύκολο να το ξεκινήσεις - δεν χρειάζεται καμία approval. Απλά download και training.

*Llama-2:*
Χρειάζεται request access από το Meta, που συνήθως εγκρίνεται γρήγορα. Έχει πολλή τεκμηρίωση online.

*DeepSeek:*
Ανοιχτό όπως το Mistral, αλλά λιγότερο documentation. Χρειάζεται trust_remote_code=True στον κώδικα.

**Μνήμη:**

Με 4-bit quantization, και τα τρία χωράνε άνετα σε 16GB GPU. Στο training peak memory ήταν ~14GB.

**Αξιοπιστία:**

Όλα ήταν σταθερά κατά το training. Δεν παρουσιάστηκαν crashes ή weird behaviors.

---

## Ollama Deployment

Όλα τα μοντέλα μετατρέπονται εύκολα σε GGUF format για Ollama:

**Μεγέθη files (Q4_0 quantization):**
- Mistral: ~4 GB
- Llama-2: ~4 GB  
- DeepSeek: ~4 GB

Το Q4_0 format διατηρεί πολύ καλή ποιότητα με λογικό μέγεθος.

---

## Πότε να χρησιμοποιήσεις κάθε μοντέλο;

**Mistral 7B - Προτεινόμενο για production:**

Πλεονεκτήματα:
- Πιο γρήγορο (Grouped Query Attention)
- Μεγαλύτερο context (8K tokens)
- Δεν έχει περιορισμούς access
- Καλή τεκμηρίωση

Μειονεκτήματα:
- Νεότερο, λιγότερο tested σε μεγάλη κλίμακα

Χρήση: Το καλύτερο choice για production deployment. Αν θες κάτι που θα τρέχει 24/7, αυτό.

**Llama-2 7B - Το "safe choice":**

Πλεονεκτήματα:
- Πολύ tested στην πράξη
- Τεράστια community
- Πολλές online λύσεις σε προβλήματα
- Αξιόπιστο

Μειονεκτήματα:
- Χρειάζεται Meta approval
- Context μόνο 4K tokens

Χρήση: Αν θες κάτι proven και δεν σε νοιάζει το approval process, πήγαινε σε αυτό.

**DeepSeek 7B - Το "interesting alternative":**

Πλεονεκτήματα:
- Καλό σε reasoning tasks
- Μεγάλο vocabulary
- Ανοιχτό
- Competitive αποτελέσματα

Μειονεκτήματα:
- Λιγότερη τεκμηρίωση
- Μικρότερη community

Χρήση: Καλό για experimentation ή αν χρειάζεσαι το μεγαλύτερο vocabulary. Ενδιαφέρον για πιο σύνθετα tasks.

---

## Προτάσεις

Με βάση το testing και τη χρήση:

**Για άμεση production χρήση:**
Mistral 7B. Είναι γρήγορο, αξιόπιστο, και χωρίς περιορισμούς.

**Για maximum reliability:**
Llama-2 7B. Έχει αποδειχτεί σε χιλιάδες deployments. Αν θέλεις κάτι που "just works", αυτό.

**Για testing νέων features:**
DeepSeek 7B. Ενδιαφέρον για specialized use cases.

**Το setup που προτείνω:**
- Primary: Mistral (για ταχύτητα και efficiency)
- Fallback: Llama-2 (για backup και comparison)
- Experimental: DeepSeek (για testing σε specific cases)

---

## Βελτιώσεις στο μέλλον

Για να γίνουν ακόμα καλύτερα τα μοντέλα:

**Βραχυπρόθεσμα (1-2 εβδομάδες):**
- Training με 2-3 epochs αντί για 1
- Περισσότερα training samples (50K+)
- Δοκιμή σε domain-specific κείμενα (ιατρικά, νομικά, τεχνικά)

**Μεσοπρόθεσμα (1-2 μήνες):**
- Bidirectional translation (EN→EL επίσης)
- Fine-tuning για συγκεκριμένους τομείς
- Comprehensive evaluation με BLEU/ROUGE scores

**Μακροπρόθεσμα (3-6 μήνες):**
- Training μεγαλύτερων μοντέλων (13B, 70B)
- Ensemble methods (χρήση 2-3 μοντέλων μαζί)
- Production API deployment

---

## Συμπέρασμα

Όλα τα μοντέλα έδωσαν εξαιρετικά αποτελέσματα. Η επιλογή εξαρτάται από το τι χρειάζεσαι:

- **Ταχύτητα;** → Mistral
- **Αξιοπιστία;** → Llama-2
- **Experimentation;** → DeepSeek

Προσωπικά, θα έβαζα το Mistral σε production και θα κρατούσα το Llama-2 ως backup. Το DeepSeek είναι ενδιαφέρον για testing.

Τα τρία μαζί δίνουν flexibility - μπορείς να διαλέξεις ανάλογα με το use case.

---

**Σημείωση:** Αυτή η ανάλυση βασίζεται στο testing με 20K samples και 1 epoch. Με περισσότερο training, τα αποτελέσματα θα βελτιωθούν ακόμα.