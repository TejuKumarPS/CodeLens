

CodeLens_Requirements.docx
docx

You are two senior software engineers building a modular Information Retrieval (IR) system for an academic project. Your goal is to implement a complete, production-ready system step-by-step using Git-based collaboration.
⚠️ STRICT EXECUTION RULES:
* You MUST generate ONLY ONE milestone at a time.
* After completing a milestone, you MUST STOP.
* You MUST WAIT for the user to type: "NEXT" before continuing.
* Do NOT generate future milestones unless explicitly asked.
* Do NOT summarize future steps.
* Each milestone must be fully complete, runnable, and integrated.
CORE REQUIREMENTS
* No pseudo-code, no placeholders, no "extend this"
* All code must be COMPLETE and EXECUTABLE
* Include imports, configs, and dependencies
* Ensure modular design (each milestone builds cleanly on previous ones)
PROJECT INITIALIZATION (ONLY ONCE AT START)
When starting:
1. Define the IR problem
2. Define dataset
3. Define evaluation metrics
4. Define high-level architecture
5. Define module boundaries
Then immediately proceed to Milestone 1.
MILESTONE RULES (VERY IMPORTANT)
For the CURRENT milestone ONLY, provide:
1. Milestone Title
* Clear, project-specific
2. Objective
* What this milestone achieves
3. Files to Create
* Exact file paths
4. FULL IMPLEMENTATION CODE
* No skeletons
* Fully runnable
* Integrated with previous code (if any)
5. Work Split (2 Engineers)
* Engineer A tasks
* Engineer B tasks
6. Git Commands (MANDATORY)
Provide EXACT commands for BOTH teammates:
Engineer A:
* branch creation
* add/commit/push
Engineer B:
* branch creation
* add/commit/push
Integration:
* pull
* merge
* resolve conflicts (if any)
7. Run Instructions
* Exact commands to execute this milestone
* Expected output
8. Acceptance Criteria
* Conditions that confirm milestone is complete
MODULARITY CONSTRAINT
* Each milestone must be an independent, testable module
* Code must be reusable and cleanly structured
* No tight coupling across modules
INTERACTION PROTOCOL
After completing a milestone, you MUST end with:
"Milestone X complete. Type NEXT to continue."
If the user types:
* "NEXT" → proceed to next milestone
* Anything else → respond accordingly
FINAL GOAL
By the end of all milestones:
* Fully working IR system
* Clean modular architecture
* Git history showing incremental progress
* Ready for academic submission with full marks
Start now with project initialization + Milestone 1 ONLY.

Show more
9:36 AM

Viewed a file, ran a command
Viewed a file, ran a command
Now I have a complete picture of the requirements. Let me proceed with project initialization and Milestone 1.

🚀 PROJECT INITIALIZATION — CodeLens
IR Problem Definition
CodeLens is a cross-modal Information Retrieval system. Given a bug report (natural language text + screenshot), retrieve the top-K most semantically relevant code functions from a codebase — bridging the vocabulary gap between natural language and source code via neural embeddings.

Dataset
Dataset	Role
CodeSearchNet (Python subset, ~180K functions)	Code corpus to index and retrieve from
Defects4J (854 real bugs)	Ground-truth evaluation set
Hand-curated GitHub Issues (50–100)	Multimodal evaluation (text + screenshot)
Evaluation Metrics
MRR (Mean Reciprocal Rank) — rank of first correct answer
NDCG@10 — quality of top-10 results, position-weighted
Precision@K — fraction of top-K results that are relevant
High-Level Architecture
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
Module Boundaries
Milestone	Module	Responsibility
1	data_loader	Fetch & preprocess CodeSearchNet
2	embedder	CodeBERT + CLIP encoders
3	indexer	ChromaDB vector indexing
4	retriever	Bi-encoder ANN search
5	reranker	Cross-encoder re-ranking
6	fusion	Late fusion of modalities
7	evaluator	MRR, NDCG, Precision@K
8	api	FastAPI backend
9	frontend	React UI
10	integration	End-to-end testing + submission
🏁 MILESTONE 1 — Data Loader & Preprocessing Pipeline
Objective
Build a fully functional, tested data loading and preprocessing module that fetches the CodeSearchNet Python dataset, cleans and filters it, and serializes it to a standardized format ready for the embedding pipeline in Milestone 2.


Created 11 files, ran 2 commands
Created 11 files, ran 2 commands

Create project directory structure

Script

Project README
README.md

requirements.txt with all dependencies
requirements.txt

data_loader __init__.py
__init__.py

CodeRecord datamodel
models.py

Raw data loader from HuggingFace
loader.py

Data cleaning / filtering logic
cleaner.py

End-to-end data pipeline that ties loader, cleaner, and serialization together
pipeline.py

Comprehensive tests for data_loader module
test_data_loader.py

conftest.py and tests __init__
conftest.py

Create tests __init__

Script

pytest.ini configuration
pytest.ini

.gitignore for the project
.gitignore
Done
Now let's install the required packages and run the tests:


Ran 2 commands
Ran 2 commands
Good — 22/26 pass. Let me fix the 4 failures:


Ran 3 commands
Ran 3 commands
The test suite has 22/26 passing. Here are the 4 issues and their fixes:

Issue 1 — test_tokens_preserved: test provides only 6 tokens, but MIN_CODE_TOKENS=10. Fix: use 10+ tokens in test.

Issue 2 & 4 — test_process_split_with_mock_data / test_full_pipeline_end_to_end: mock records use short functions (9 tokens < 10 threshold). Fix: make mock code longer.

Issue 3 — save_processed KeyError on tokens: to_dict() uses dataclasses.asdict() which already serializes — but when records list is empty, the DataFrame has no columns. Fix: handle empty records case.

Here are the exact fixes to apply:

Continue

9:44 AM
Let me fix all 4 failing tests before continuing:


Edited 3 files, ran a command
Edited 3 files, ran a command

Fix mock dataset to produce enough tokens and fix save_processed empty case

pipeline.py
+14
-5

Fix mock dataset to use longer code (>= 10 tokens) and fix test_tokens_preserved

test_data_loader.py
+20
-13

Fix test_tokens_preserved to provide enough tokens to pass MIN_CODE_TOKENS=10

test_data_loader.py
+7
-6

Run full test suite after fixes

Script
Done
26/26 passing. Now initialize Git and create the exact commands both engineers need:


Ran 5 commands
Ran 5 commands

Initialize git repo with base commit

Script

Create milestone branch and commit

Script

Check git status

Script

Verify all milestone 1 files are tracked

Script

Final git state check

Script
Done
All files are tracked. Now copy to outputs:


Ran a command, read a file
Ran a command, read a file
✅ Milestone 1 — Complete
Test Results: 26/26 passing

📁 Files Created
File	Responsibility
data_loader/models.py	CodeRecord dataclass — canonical schema for all modules
data_loader/loader.py	HuggingFace CodeSearchNet fetcher
data_loader/cleaner.py	5-filter quality pipeline + deduplication
data_loader/pipeline.py	Orchestrator — clean → dedupe → Parquet serialize/load
data_loader/__init__.py	Public API surface
tests/test_data_loader.py	26 tests across 5 test classes
requirements.txt	All project dependencies
README.md	Project overview
.gitignore	Python + data + model exclusions
👷 Work Split
Engineer A — Data models & cleaning logic

data_loader/models.py — CodeRecord, to_dict, to_chroma_metadata, from_dict
data_loader/cleaner.py — all 5 quality filters, boilerplate detection, ID generation
Tests: TestCodeRecord, TestCleaner, TestCleaningStats
Engineer B — Loading & pipeline orchestration

data_loader/loader.py — HuggingFace loader with split/language/limit support
data_loader/pipeline.py — full pipeline, Parquet I/O, CLI entry point
Tests: TestPipeline, TestIntegrationSmoke
🔀 Git Commands
bash
# ── ENGINEER A ──────────────────────────────────────────────────
git clone <repo-url> codelens && cd codelens
git checkout -b feature/m1-models-cleaner
# ... write models.py, cleaner.py, their tests ...
git add data_loader/models.py data_loader/cleaner.py tests/test_data_loader.py
git commit -m "feat(data_loader): CodeRecord model and cleaning pipeline"
git push origin feature/m1-models-cleaner

# ── ENGINEER B ──────────────────────────────────────────────────
git checkout -b feature/m1-loader-pipeline
# ... write loader.py, pipeline.py ...
git add data_loader/loader.py data_loader/pipeline.py data_loader/__init__.py
git commit -m "feat(data_loader): HuggingFace loader and pipeline orchestrator"
git push origin feature/m1-loader-pipeline

# ── INTEGRATION (both) ──────────────────────────────────────────
git checkout main
git pull origin main
git merge feature/m1-models-cleaner --no-ff -m "merge: M1 models+cleaner"
git merge feature/m1-loader-pipeline --no-ff -m "merge: M1 loader+pipeline"
# No conflicts expected — different files
git push origin main
▶️ Run Instructions
bash
# Install dependencies
pip install -r requirements.txt

# Run tests (no internet needed — all mocked)
pytest tests/test_data_loader.py -v

# Run the actual pipeline (downloads ~3.5 GB on first run)
python -m data_loader.pipeline --split train --language python --limit 5000 --output data/processed/
python -m data_loader.pipeline --split test  --language python --limit 500  --output data/processed/

# Expected output:
# ✅ Pipeline complete!
#    Records saved : ~4700   (after filtering ~5000 raw)
#    Output file   : data/processed/python_train.parquet
✔️ Acceptance Criteria
 pytest tests/test_data_loader.py → 26/26 passed
 CodeRecord.to_chroma_metadata() returns only str/int/float/bool values
 save_processed → load_processed roundtrip preserves all fields including token lists
 Deduplication by (repository, func_name) works correctly
 Records with < 10 tokens, < 15 char docstrings, or empty code are rejected
 All code is importable: from data_loader import CodeRecord, load_processed
Milestone 1 complete. Type NEXT to continue.

Pipeline
PY 
Models
PY 
Cleaner
PY 
Loader
PY 
Test data loader
PY 
can you just give me folder structure of this code

9:56 AM
Here's the complete folder structure for Milestone 1:

codelens/
├── README.md
├── requirements.txt
├── pytest.ini
├── .gitignore
│
├── data_loader/
│   ├── __init__.py        ← public API (imports from all submodules)
│   ├── models.py          ← CodeRecord dataclass
│   ├── loader.py          ← HuggingFace CodeSearchNet fetcher
│   ├── cleaner.py         ← quality filters + deduplication logic
│   └── pipeline.py        ← orchestrator: clean → dedupe → Parquet I/O + CLI
│
├── data/
│   └── processed/         ← output Parquet files (git-ignored)
│       ├── python_train.parquet
│       ├── python_test.parquet
│       └── python_train_stats.json
│
└── tests/
    ├── __init__.py
    ├── conftest.py        ← sys.path setup for pytest
    └── test_data_loader.py  ← 26 tests across 5 test classes
As milestones are added, the tree will grow with sibling modules (embedder/, indexer/, retriever/, etc.) all at the same level as data_loader/.

am i not supposed to push project skeleton structure fiirst to github

12:28 PM
Yes, absolutely! The standard Git collaboration workflow is:

One person creates the repo on GitHub and pushes the skeleton
Both engineers clone it and branch off from there
Here's the exact sequence:

🧱 Step 0 — Project Skeleton (do this FIRST, before any milestone)
Engineer A does this once:

bash
# 1. Create repo on GitHub (via UI): "codelens" — initialize with nothing

# 2. On your local machine:
mkdir codelens && cd codelens
git init
git branch -M main

# 3. Create the skeleton structure
mkdir -p data_loader tests data/processed scripts

# 4. Add placeholder files so folders are tracked
touch data_loader/__init__.py
touch tests/__init__.py
touch tests/conftest.py
touch .gitignore
touch README.md
touch requirements.txt

# 5. Add your .gitignore, README, requirements.txt content (copy from Milestone 1)

# 6. Commit and push skeleton
git add .
git commit -m "chore: initial project skeleton"
git remote add origin https://github.com/<your-username>/codelens.git
git push -u origin main
Engineer B then:

bash
git clone https://github.com/<your-username>/codelens.git
cd codelens
Then Milestone 1 branches off from this skeleton:
bash
# Engineer A
git checkout -b feature/m1-models-cleaner

# Engineer B
git checkout -b feature/m1-loader-pipeline
So the correct order is:

GitHub repo created
       ↓
Skeleton pushed to main
       ↓
Both engineers clone
       ↓
Milestone 1 branches created
       ↓
PRs merged back to main
       ↓
Milestone 2 branches off updated main
       ↓
... and so on
Would you like me to generate the exact skeleton files you should push right now before we continue to Milestone 2?






✨ Optimize Prompt
Claude is AI and can make mistakes. Please double-check responses.
Readme · MD
Copy

# CodeLens 🔍
> Cross-Modal Bug Report Retrieval Using Fused Text and Visual Crash Evidence
 
**PES University Bangalore | Algorithms for Information Retrieval | Final Project | Batch 2023–2027 | 6th Semester**
 
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
 
