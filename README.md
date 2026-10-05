# YanlongGPT-CNT-v2

**YanlongGPT-CNT-v2** is a compact, domain-specific language model and retrieval-augmented generation (RAG) research pipeline for carbon nanotube (CNT) science.

The project was built from scratch to study how a small decoder-only language model can acquire CNT-domain language, scientific question-answering behavior, and evidence-conditioned generation through sequential pretraining, supervised fine-tuning (SFT), and RAG-SFT.

> **Research prototype.** YanlongGPT-CNT is intended for research and demonstration. Generated scientific claims should be checked against the retrieved evidence and original literature.

---

## Overview

YanlongGPT-CNT-v2 follows a three-stage model-development pipeline:

```text
Historical CNT literature (≤2024)
            |
            v
Data cleaning and corpus construction
            |
            v
Byte-level BPE tokenizer
            |
            v
Domain pretraining
            |
            v
Base Model
            |
            v
Scientific supervised fine-tuning
            |
            v
SFT Model
            |
            v
Evidence-grounded RAG-SFT
            |
            v
RAG-SFT Model
```

A separate external retrieval corpus is constructed from **2025–2026 literature**:

```text
2025–2026 CNT literature
          |
          v
Document chunking and cleaning
          |
          v
BGE embeddings
          |
          v
FAISS vector index
          |
          v
Top-k evidence retrieval
          |
          v
Evidence + Question
          |
          v
RAG-SFT Model
          |
          v
Generated answer
```

This temporal design keeps 2025–2026 papers outside tokenizer construction, pretraining, ordinary SFT, and RAG-SFT, allowing them to serve as an **out-of-training retrieval corpus**.

---

## Key Features

- Self-built decoder-only Transformer language model
- CNT-domain ByteLevel BPE tokenizer
- Domain pretraining on historical CNT literature
- Scientific instruction tuning with CNT question-answer pairs
- Evidence-grounded RAG-SFT with traceable source chunks
- BGE-based semantic embedding
- FAISS vector retrieval
- Paper-level train/validation splitting for RAG-SFT
- Temporal holdout between training literature and external retrieval literature
- Controlled Base / SFT / RAG-SFT model comparison
- Anti-repetition decoding for compact-model inference

---

## Model Configuration

| Component | Setting |
|---|---:|
| Architecture | Decoder-only Transformer |
| Parameters | 22,269,696 (~22.3M) |
| Transformer layers | 8 |
| Attention heads | 8 |
| Embedding dimension | 384 |
| Context length | 1024 tokens |
| Vocabulary size | 10,000 |
| Tokenizer | ByteLevel BPE |
| Dropout | 0.1 |

The three exposed model stages use the **same architecture** and differ only in training stage:

```text
Base checkpoint
      |
      v
Scientific SFT
      |
      v
SFT checkpoint
      |
      v
Evidence-grounded RAG-SFT
      |
      v
RAG-SFT checkpoint
```

---

## Dataset Summary

### Historical training corpus

| Item | Value |
|---|---:|
| Historical papers | 2,544 |
| Publication cutoff | ≤2024 |
| Approximate text size | 83.59M characters |
| Total tokenized corpus | 24,417,025 tokens |
| Pretraining train papers | 2,417 |
| Pretraining validation papers | 127 |
| Pretraining train tokens | 23,213,172 |
| Pretraining validation tokens | 1,203,853 |

The pretraining split is performed at the **paper level**.

### Scientific SFT

| Item | Value |
|---|---:|
| Clean scientific QA pairs | 12,366 |
| Training objective | Answer-only supervised loss |

Representative QA categories include:

- growth mechanism
- catalyst effect
- process parameter
- carbon source
- structure control
- characterization
- structure-property relationship
- optimization
- failure and limitation
- quantitative condition
- comparison
- synthesis strategy

### RAG-SFT

| Item | Value |
|---|---:|
| Raw grounded QA pairs | 7,695 |
| Clean grounded QA pairs | 7,282 |
| Training samples | 6,933 |
| Validation samples | 349 |
| Training source papers | 2,309 |
| Validation source papers | 122 |
| Paper overlap | 0 |

Each RAG-SFT sample preserves a provenance chain linking the generated question and answer to one or two exact evidence chunks from the source paper.

### External RAG corpus

| Item | Value |
|---|---:|
| Effective papers | 252 |
| Publication years | 2025–2026 |
| Clean chunks | 20,467 |
| Embedding model | BAAI/bge-base-en-v1.5 |
| Embedding dimension | 768 |
| Vector index | FAISS IndexFlatIP |

---

## Evidence-Grounded Data Construction

The RAG-SFT dataset is generated directly from historical papers using the following structure:

```text
Source paper
    |
    v
Deterministic text chunks
    |
    v
Selected evidence chunk(s)
    |
    v
Evidence -> Question -> Answer generation
    |
    v
Source / chunk / text / numeric validation
    |
    v
Clean grounded QA dataset
```

The cleaner validates, among other items:

- source file identity
- publication year
- evidence chunk IDs
- reconstructed chunk text
- source SHA1 metadata
- evidence span consistency
- duplicate samples
- token length
- unsupported numeric claims

The final cleaner retained **7,282 / 7,695 samples (94.63%)**. Rejected samples were associated with unsupported numeric claims.

---

## Repository Structure

```text
YanlongGPT-CNT-v2/
|
|-- scripts/
|   |
|   |-- 01_clean_md.py
|   |-- 02_build_corpus.py
|   |-- 03_analyze_corpus.py
|   |-- 04_train_tokenizer.py
|   |-- 05_prepare_dataset.py
|   |-- 06_yanlongllm.py
|   |-- 07_inference_yanlongllm.py
|   |-- 08_generate_cnt_sft.py
|   |-- 09_clean_cnt_sft.py
|   |-- 10_sft_dataset.py
|   |-- 11_test_sft.py
|   |-- 12_chat_sft.py
|   |
|   |-- rag/
|   |   |-- 13_chunk_papers.py
|   |   |-- 14_clean_chunks.py
|   |   |-- 15_build_embedding.py
|   |   |-- 16_build_faiss.py
|   |   |-- 17_retriever.py
|   |   `-- 18_chat_rag.py
|   |
|   |-- ragsft/
|   |   |-- 19_build_rag_sft.py
|   |   |-- 20_clean_rag_sft.py
|   |   |-- 21_rag_sft_dataset.py
|   |   `-- 22_test_rag_sft.py
|   |
|   |-- train/
|   |   |-- train_yanlongllm.py
|   |   |-- train_yanlongllm_sft.py
|   |   `-- train_yanlongllm_ragsft.py
|   |
|   `-- data/
|       `-- auxiliary preprocessing utilities
|
|-- tokenizer/
|   `-- tokenizer files
|
|-- environment.yml
|-- .gitignore
|-- LICENSE
`-- README.md
```

Large datasets, model checkpoints, downloaded embedding models, API keys, and the local web application are intentionally excluded from the public repository.

---

## Installation

The reference environment uses **Python 3.12**.

```bash
conda env create -f environment.yml
conda activate yanlonggpt-cnt
```

The original experiments were conducted with:

```text
Python 3.12.2
PyTorch 2.5.1+cu121
NVIDIA RTX 4060 Laptop GPU
CUDA 12.1
```

GPU support depends on the locally installed PyTorch/CUDA configuration.

---

## Pipeline Usage

The scripts are numbered in pipeline order.

### 1. Corpus preprocessing

```bash
python scripts/01_clean_md.py
python scripts/02_build_corpus.py
python scripts/03_analyze_corpus.py
```

### 2. Tokenizer and pretraining dataset

```bash
python scripts/04_train_tokenizer.py
python scripts/05_prepare_dataset.py
```

### 3. Domain pretraining

```bash
python scripts/train/train_yanlongllm.py
```

### 4. Ordinary scientific SFT

Generate and clean the scientific QA dataset:

```bash
python scripts/08_generate_cnt_sft.py
python scripts/09_clean_cnt_sft.py
```

Train the SFT model:

```bash
python scripts/train/train_yanlongllm_sft.py
```

### 5. Build the external RAG index

```bash
python scripts/rag/13_chunk_papers.py
python scripts/rag/14_clean_chunks.py
python scripts/rag/15_build_embedding.py
python scripts/rag/16_build_faiss.py
```

Test retrieval:

```bash
python scripts/rag/17_retriever.py
```

### 6. Build and train the RAG-SFT stage

```bash
python scripts/ragsft/19_build_rag_sft.py
python scripts/ragsft/20_clean_rag_sft.py
python scripts/ragsft/21_rag_sft_dataset.py
python scripts/train/train_yanlongllm_ragsft.py
```

### 7. Controlled evaluation

```bash
python scripts/ragsft/22_test_rag_sft.py
```

---

## API-Key-Dependent Dataset Generation

Some dataset-generation scripts use an OpenAI-compatible API client.

API credentials are **not included** in this repository.

If you regenerate these datasets, provide your own local credential file or adapt the relevant scripts to your preferred API provider. The local key file is excluded through `.gitignore`.

Never commit API credentials to a public repository.

---

## Inference and Decoding

The final comparison pipeline uses deterministic anti-repetition decoding:

```text
Maximum new tokens: 160
Repetition penalty: 1.18
No-repeat n-gram size: 4
Repetition window: 96
Decoding strategy: penalized greedy
```

These settings primarily improve generation stability. They do not guarantee factual correctness or strict evidence grounding.

---

## Data and Model Availability

The following assets are **not included** in the public repository:

- original scientific papers
- cleaned literature corpus
- tokenized pretraining datasets
- generated SFT datasets
- generated RAG-SFT datasets
- 2025–2026 external literature corpus
- FAISS index and embedding matrices
- downloaded BGE model files
- Base / SFT / RAG-SFT checkpoints
- API credentials
- local FastAPI / frontend demo

Original papers are excluded because the repository does not redistribute third-party copyrighted literature.

Users may prepare their own legally accessible scientific corpus and reproduce the pipeline using the provided code.

---

## Limitations

YanlongGPT-CNT-v2 is a compact **22.3M-parameter research model**, not a production scientific assistant.

Although RAG-SFT improves the model's ability to condition generation on explicit evidence, evidence grounding remains imperfect at this model scale. Retrieved passages and generated claims should therefore be inspected against the original literature before scientific use.

The 2025–2026 corpus is used as an external retrieval source rather than as training data, but retrieval itself can still introduce irrelevant or noisy passages.

---

## Reproducibility Notes

For a faithful reconstruction of the project:

1. Use literature published no later than 2024 for tokenizer construction, pretraining, ordinary SFT, and RAG-SFT.
2. Keep 2025–2026 literature separate if reproducing the temporal holdout experiment.
3. Preserve paper-level train/validation separation.
4. Do not mix external retrieval documents into model-training stages.
5. Record source provenance for grounded QA generation.

---

## License

The source code in this repository is released under the **MIT License**. See [`LICENSE`](LICENSE).

The MIT License applies to the repository's source code and does not grant rights to third-party scientific papers, external datasets, pretrained third-party models, or other content not distributed as part of this repository.

---

## Author

**Yanlong Zhao**

YanlongGPT-CNT-v2 is an independent research prototype developed for AI-for-science exploration in carbon nanotube synthesis and processing.
