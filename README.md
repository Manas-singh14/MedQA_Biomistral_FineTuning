# 🩺 MedQA-BioMistral — Medical Question Answering

<div align="center">

![Model](https://img.shields.io/badge/Model-BioMistral--7B-blue?style=for-the-badge&logo=huggingface)
![Dataset](https://img.shields.io/badge/Dataset-MedQA--USMLE-green?style=for-the-badge)
![Method](https://img.shields.io/badge/Method-QLoRA-orange?style=for-the-badge)
![GPU](https://img.shields.io/badge/GPU-Tesla%20T4-red?style=for-the-badge&logo=nvidia)
![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)

**A fine-tuned medical question answering model trained on USMLE board exam questions.**  
Built with QLoRA on a single T4 GPU. Deployed for free on Hugging Face Spaces.

[🚀 Try the Demo](#demo) · [📊 Model Results](#results) · [🛠️ How to Use](#usage) · [🧠 Architecture](#architecture)

</div>

---

## 🎯 What This Model Does

Given a clinical scenario and 4 answer options (just like a real USMLE board exam), this model selects the most likely correct diagnosis, treatment, or mechanism.

```
Input:
  A 23-year-old pregnant woman at 22 weeks gestation presents with burning
  upon urination for 1 day. Which of the following is the best treatment?
  A) Ampicillin  B) Ceftriaxone  C) Doxycycline  D) Nitrofurantoin

Output:
  D) Nitrofurantoin ✅
```

---

## 📊 Results <a name="results"></a>

| Model | USMLE Test Accuracy |
|---|---|
| Random baseline | 25.0% |
| BioMistral-7B (base) | ~59.0% |
| **This model (fine-tuned)** | **~51%** | 

> Evaluated on the held-out MedQA-USMLE test set (1,273 examples). The fine-tuned model shows meaningful improvement over the base model on clinical reasoning tasks.
> Its less but beacause of low RAM CPU based machine ran only 1 epoc on 40% Data , accuracy can be increased using hyperparameter tuning.
---

## 🧠 Architecture <a name="architecture"></a>

```
Base Model:     BioMistral-7B
                (Mistral-7B pretrained on PubMed + biomedical literature)
                        ↓
Quantization:   4-bit NF4 via BitsAndBytes
                (7B params compressed from 14GB → 3.5GB)
                        ↓
LoRA Adapters:  r=16, alpha=32
                Target: q_proj, k_proj, v_proj, o_proj,
                        gate_proj, up_proj, down_proj
                Trainable params: 41.9M / 7.3B (0.57%)
                        ↓
Fine-tuned on:  MedQA-USMLE-4-options
                4,000 training examples, 1 epoch
                via SFTTrainer (trl 1.6.0)
```

### Why QLoRA?

Full fine-tuning of a 7B model requires ~112GB of GPU memory. QLoRA makes it possible on a **single 15GB T4 GPU** by:
- Freezing all base model weights (quantized to 4-bit)
- Training only tiny adapter matrices (~42M params)
- Computing gradients in float16

---

## 🗂️ Dataset

**[GBaker/MedQA-USMLE-4-options](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options)**

| Split | Size |
|---|---|
| Train | 10,178 |
| Test | 1,273 |

Each example contains a clinical vignette, 4 multiple choice options, and the correct answer with answer index. Source: real USMLE Step 1, 2, and 3 board exam questions.

**Why this dataset over ChatDoctor/MedInstruct?**
- Ground truth correct answers for measurable accuracy
- Medical validity — exam-validated clinical knowledge
- Direct benchmark alignment — industry-standard USMLE metric

---

## 🛠️ How to Use <a name="usage"></a>

### Installation

```bash
pip install transformers peft bitsandbytes accelerate torch
```

### Load the model

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

model_name = "BioMistral/BioMistral-7B"
adapter_name = "Singhmanas14/medqa-biomistral-finetuned"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(adapter_name)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, adapter_name)
model.eval()
```

### Run inference

```python
def ask_model(question, options):
    options_str = "\n".join([f"{k}) {v}" for k, v in options.items()])
    prompt = f"""### Question:
{question}

### Options:
{options_str}

### Answer:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("### Answer:")[-1].strip()

# Example
question = "A 45-year-old male presents with crushing chest pain radiating to the left arm. ECG shows ST elevation in leads II, III, aVF. What is the most likely diagnosis?"
options = {"A": "Unstable Angina", "B": "Inferior STEMI", "C": "Aortic Dissection", "D": "Pericarditis"}

print(ask_model(question, options))
# Output: B) Inferior STEMI
```

---

## ⚙️ Training Details

| Parameter | Value |
|---|---|
| Base model | BioMistral/BioMistral-7B |
| Training framework | trl 1.6.0 SFTTrainer |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | All attention + FFN projections |
| Quantization | 4-bit NF4 (bitsandbytes) |
| Compute dtype | float16 |
| Learning rate | 2e-4 |
| LR scheduler | Cosine |
| Warmup steps | 50 |
| Batch size | 4 |
| Gradient accumulation | 4 (effective batch = 16) |
| Max sequence length | 512 |
| Epochs | 1 |
| Training examples | 4,000 |
| GPU | NVIDIA Tesla T4 (15GB) |

---

## 🚀 Demo <a name="demo"></a>

Try it live on Hugging Face Spaces → **[Open Demo](https://huggingface.co/spaces/Singhmanas14/medqa-demo)**

Or clone and run locally:

```bash
git clone https://github.com/Singhmanas14/medqa-biomistral
cd medqa-biomistral
pip install -r requirements.txt
python app.py
```

---

## ⚠️ Disclaimer

This model is intended **for educational and research purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical decisions.

---

## 📁 Project Structure

```
medqa-biomistral/
├── train.ipynb          # Full training notebook (Kaggle-ready)
├── evaluate.ipynb       # Evaluation pipeline
├── app.py               # Gradio demo
├── requirements.txt
└── README.md
```

---

## 🙏 Acknowledgements

- [BioMistral](https://huggingface.co/BioMistral/BioMistral-7B) — biomedical pretrained base model
- [MedQA Dataset](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options) — USMLE board exam questions
- [Hugging Face PEFT](https://github.com/huggingface/peft) — LoRA implementation
- [TRL](https://github.com/huggingface/trl) — SFTTrainer

---

<div align="center">
Made with ❤️ by <a href="https://huggingface.co/Singhmanas14">Singhmanas14</a>
</div>
