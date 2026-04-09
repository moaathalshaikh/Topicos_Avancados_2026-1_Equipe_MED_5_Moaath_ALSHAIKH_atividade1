# Medical LLM Evaluation – Moaath Almohammad Alshaikh

**Universidade Federal de Sergipe (UFS)**  
**Disciplina:** Tópicos Avançados em Computação  
**Professor:** Glauco Carneiro  
**Semestre:** 2026.1  
**Equipe:** 5 – Domínio Médico  
**Aluno:** Moaath Almohammad Alshaikh — Matrícula: 2024108522

---

## Overview

This repository contains my individual contribution to Activity 1. The task involved curating a subset of medical questions from the K-QA dataset and running inference with three large language models, comparing their performance on both open-ended and multiple-choice question (MCQ) formats.

My assigned question ranges:
- **Open-ended:** Questions 101–117 (K-QA dataset)
- **MCQ:** Questions 163–189 (USMLE dataset)

---

## Dataset

| Type | Source | Questions assigned |
|------|--------|--------------------|
| Open-ended (M1) | [K-QA – Itaymanes](https://github.com/Itaymanes/K-QA/blob/main/dataset/questions_w_answers.jsonl) | 101–117 (17 questions) |
| MCQ (M2) | USMLE | 163–189 (27 questions) |

The K-QA dataset includes the fields `Question`, `Free_form_answer`, `Must_have`, `Nice_to_have`, `Sources`, and `ICD_10_diag`, as described in the reference article:  
> [K-QA: A Real-World Medical Q&A Benchmark — BioNLP 2024](https://aclanthology.org/2024.bionlp-1.22/)

---

## Models Evaluated

Three models were selected and run via Hugging Face `transformers` with 4-bit quantization (BitsAndBytes NF4) on a T4 GPU (Google Colab):

| Model | HuggingFace ID | Parameters |
|-------|---------------|------------|
| **BioMistral-7B** | `BioMistral/BioMistral-7B` | 7B |
| **Mistral-7B-Instruct-v0.1** | `mistralai/Mistral-7B-Instruct-v0.1` | 7B |
| **Qwen1.5-1.8B-Chat** | `Qwen/Qwen1.5-1.8B-Chat` | 1.8B |

All models were loaded with the same quantization configuration for a fair comparison:
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)
```

---

## Evaluation Methodology

### Open-ended Questions
Responses were scored using a **Must-Have keyword coverage metric**. Each answer from the model was compared against the `Must_have` list provided in the K-QA dataset. A keyword phrase is considered matched if at least half of its tokens appear in the model's response (case-insensitive).

```
score = matched_keywords / total_must_have_keywords
```

### MCQ Questions
Each model was prompted to output a single letter (A–F). The predicted letter was extracted using pattern matching and compared to the correct answer. The metric used is straightforward **accuracy** (correct / total).

---

## Results

### MCQ Performance

| Model | Correct | Total | Accuracy |
|-------|---------|-------|----------|
| BioMistral-7B | 9 | 27 | **33.3%** |
| Mistral-7B-Instruct | 9 | 27 | **33.3%** |
| Qwen1.5-1.8B | 10 | 27 | **37.0%** |

### Open-ended Performance (Must-Have Score)

| Model | Mean | Median | Std | Min | Max | ≥0.5 (Good) | ≥0.8 (Excellent) |
|-------|------|--------|-----|-----|-----|-------------|------------------|
| BioMistral-7B | 0.365 | 0.400 | 0.402 | 0.00 | 1.00 | 6 | 4 |
| Mistral-7B-Instruct | 0.612 | 0.600 | 0.372 | 0.00 | 1.00 | 11 | 7 |
| Qwen1.5-1.8B | 0.447 | 0.500 | 0.359 | 0.00 | 1.00 | 9 | 3 |

---

## Analysis & Discussion

**Mistral-7B-Instruct** was the clear leader on open-ended questions, achieving the highest mean score (0.612) and the most "excellent" responses (7 out of 17). Despite this, its MCQ accuracy was tied for last at 33.3%, suggesting it reasons well in free-form contexts but struggles with structured multiple-choice.

**Qwen1.5-1.8B** performed surprisingly competitively for a 1.8B model, narrowly leading on MCQ accuracy (37%) and achieving a decent open-ended mean (0.447). Its smaller size makes it a computationally attractive option.

**BioMistral-7B**, despite being domain-specific (pre-trained on medical literature), did not outperform the general-purpose Mistral on open-ended tasks in this evaluation. This may reflect limitations in instruction-following for the specific prompt format used.

Overall, MCQ accuracy across all three models was below 40%, which reflects the difficulty of the USMLE questions and the inherent limitations of models that were not fine-tuned for clinical reasoning tasks. The open-ended format, evaluated through keyword coverage, revealed more differentiation between models.

---

## Repository Structure

```
.
├── README_Moaath.md                          ← this file
├── moaath_BioMistral-7B.ipynb               ← BioMistral inference notebook
├── moaath_Mistral_7B_Instruct_v0_1.ipynb   ← Mistral inference notebook
├── moaath_Qwen1_5-1_8B-Chat.ipynb          ← Qwen inference notebook
├── biomistral_summary.csv                   ← BioMistral results summary
├── mistral_summary.csv                      ← Mistral results summary
└── qwen_summary.csv                         ← Qwen results summary
```

---

## How to Reproduce

1. Open any of the notebooks in Google Colab with a T4 GPU runtime.
2. Mount Google Drive and adjust the path variables at the top of each notebook to match your Drive structure.
3. The datasets (`questions_w_answers.jsonl` and the USMLE CSVs) must be available at the configured paths.
4. Run all cells in order. Each notebook will produce a summary CSV automatically.

Dependencies installed automatically inside the notebooks:
```
transformers, accelerate, bitsandbytes, sentencepiece
```
