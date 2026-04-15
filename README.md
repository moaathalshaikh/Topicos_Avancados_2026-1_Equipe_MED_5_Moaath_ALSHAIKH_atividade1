# Medical QA Evaluation Using Open-Source Small Language Models – Moaath Almohammad Alshaikh

**Federal University of Sergipe (UFS) | Universidade Federal de Sergipe (UFS)**  
**Disciplina:** Advanced Topics in Software Engineering and Information Systems I  
**Professor:** Glauco Carneiro  
**Semester:** 2026.1  
**Team:** 5 – Medical Domain  
**Ph.D Student:** Moaath Almohammad Alshaikh — Matrícula: 202611011441

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

**Mistral-7B-Instruct** was best at open-ended — mean score 0.61, with 7 excellent answers. But it scored only 33% on multiple choice.

**Qwen1.5-1.8B** was slightly better at multiple choice — 37%. Good result for such a small model.

**BioMistral-7B** was trained on medical texts but still scored the lowest on open-ended questions. Domain training alone is not enough.

Overall, all three models scored below 40% on multiple choice. USMLE questions are hard. Open-ended results showed more difference between the models.
---

## Repository Structure

```
.
├── README.md                          ← main repository overview + video link
│
├── Moaath/                            ← curation files and individual README
│   └── SUBMISSION.md
│
├── notebooks-ipynb/                   ← inference notebooks (one per model)
│   ├── moaath_BioMistral-7B.ipynb
│   ├── moaath_Mistral_7B_Instruct_v0_1.ipynb
│   └── moaath_Qwen1_5-1_8B-Chat.ipynb
│
├── results/                           ← CSV output files per model
│   ├── biomistral_mcq_moaath.csv
│   ├── biomistral_open_moaath.csv
│   ├── biomistral_summary.csv
│   ├── mistral_mcq_moaath.csv
│   ├── mistral_open_moaath.csv
│   ├── mistral_summary.csv
│   ├── qwen_mcq_moaath.csv
│   ├── qwen_open_moaath.csv
│   └── qwen_summary.csv
│
├── report/                            ← activity report (PDF/DOCX)
│   └── Moaath_Paper_Atividade_1.docx
│
└── video/                             ← demonstration video link
    └── VIDEO.md
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
