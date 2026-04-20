# CodeLens 🔍
> Cross-Modal Bug Report Retrieval Using Fused Text and Visual Crash Evidence
 
## Problem Statement
Given a bug report (natural language description + crash screenshot), retrieve the most semantically relevant code functions from a codebase — bridging the vocabulary gap between natural language and source code.
 
## Architecture
```
[Bug Text + Screenshot]
        │
   ┌────┴────┐
CodeBERT   CLIP
   │          │
text_vec  img_vec
   └────┬────┘
    Late Fusion
        │
   ChromaDB ANN
        │
   Top-20 candidates
        │
  Cross-Encoder Re-rank
        │
   Top-5 Results → React UI
```
 
## Modules
| Module         | Description                              |
|----------------|------------------------------------------|
| `data_loader`  | Fetch & preprocess CodeSearchNet dataset |
| `embedder`     | CodeBERT + CLIP encoders                 |
| `indexer`      | ChromaDB vector indexing                 |
| `retriever`    | Bi-encoder ANN search                    |
| `reranker`     | Cross-encoder re-ranking                 |
| `fusion`       | Late fusion of text + image modalities   |
| `evaluator`    | MRR, NDCG@10, Precision@K metrics        |
| `api`          | FastAPI backend                          |
| `frontend`     | React UI                                 |
 
## Datasets
- **CodeSearchNet** (Python, ~180K functions) — code corpus
- **Defects4J** (854 real bugs) — evaluation ground truth
- **Hand-curated GitHub Issues** (50–100) — multimodal eval set
 
## Evaluation Metrics
- MRR (Mean Reciprocal Rank)
- NDCG@10 (Normalized Discounted Cumulative Gain)
- Precision@K
 
## Setup
```bash
pip install -r requirements.txt
python -m data_loader.pipeline --output data/processed/
python -m pytest tests/
```
 
