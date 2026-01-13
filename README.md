````md
# Stratos – Intelligent Resume–Job Matching System

Stratos is an explainable, deterministic resume–job matching system designed to assist recruiters in ranking candidates using **semantic alignment**, **skill sufficiency**, and **experience fit** rather than keyword matching.

The system combines **LLM-generated reference resumes**, **sentence embeddings**, and a **two-phase ranking validation framework**, delivered through a desktop GUI.

---

## 🚀 Features

### 🔹 Hypothetical Ideal Resume (HYRE)
- Converts a Job Description (JD) into a deterministic **ideal candidate resume**
- Generated using a locally hosted LLM (via Ollama)
- Acts as a semantic reference for candidate comparison

### 🔹 Semantic Resume Matching
- SentenceTransformer embeddings (`all-MiniLM-L6-v2`)
- Cosine similarity between resumes and HYRE
- Cached embeddings for reproducibility and performance

### 🔹 Skill Taxonomy Matching
- Extracts technical skills from resumes and JD
- Computes normalized skill overlap score
- Used for scoring and decision logic

### 🔹 Hybrid Scoring Model
```text
Hybrid Score = α · Semantic Similarity + (1 − α) · Skill Overlap
````

(Default: α = 0.8)

Includes **experience-based penalty adjustments** for:

* Missing experience data
* Junior candidates for senior roles
* JD-specific minimum experience violations

---

### 🔹 Two-Phase Ranking Validation

1. Model-based hybrid ranking
2. Rule-based validation using skill sufficiency and experience fit

Ensures ranking stability and prevents underqualified candidates from ranking highly due to semantic similarity alone.

---

### 🔹 Explainable Rankings

Each candidate receives:

* Matched and missing JD skills
* Skill overlap and semantic scores
* Experience assessment
* Deterministic, audit-ready explanation
  (no hiring recommendation is made)

---

### 🔹 Decision Support Labels

Candidates are categorized as:

* **Consider**
* **Maybe**
* **Reject**

Based on skill coverage and experience alignment.

---

### 🔹 Desktop GUI (PySide6)

Multi-step workflow:

1. CV Upload & Embedding
2. Job Description Input
3. Ranked Results
4. Candidate Comparison
5. Model vs Final Ranking Comparison
6. Evaluation Metrics Dashboard

---

## 📊 Evaluation Metrics

The system includes an evaluation module using proxy relevance logic:

* **Recall@K**
* **NDCG@K**
* **Rank-Biased Overlap (RBO)** (Human vs Model ranking)
* Recall vs Threshold visualization

> Metrics are analytical tools and **do not represent hiring decisions**.

---

## 🛠️ Tech Stack

* **Language**: Python 3.10+
* **UI**: PySide6 (Qt)
* **NLP / ML**:

  * sentence-transformers
  * scikit-learn
  * numpy, pandas
* **LLM**:

  * Ollama (local inference)
  * Model: `gpt-oss:20b`
* **Document Parsing**:

  * pypdf
  * python-docx
* **Visualization**:

  * matplotlib

---

## ⚙️ Environment Variables

```bash
OLLAMA_HOST=http://localhost:11434
LLM_MODEL=gpt-oss:20b
HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

Ensure Ollama is running and the LLM is available:

```bash
ollama pull gpt-oss:20b
```

---

## ▶️ Running the Application

```bash
python app.py
```

---

## 📁 Input Requirements

### CV Files

Supported formats:

* `.txt`, `.md`, `.log`, `.pdf`, `.docx`

Uploaded via folder selection.

### Job Description

* Plain text
* Pasted or uploaded from file

---

## 🧱 Project Structure

```
stratos/
│
├── stratos_i.py        # Core NLP & ranking pipeline
├── helpers.py          # Skill extraction & ranking utilities
├── app.py              # PySide6 GUI application
├── logo.png
├── requirements.txt
└── README.md
```

---

## 🔁 Determinism & Reproducibility

* LLM temperature set to `0.0`
* Cached HYRE generation and embeddings
* Fixed random seeds
* Explicit penalty-based scoring rules

Ensures consistent outputs across runs.

---

## ⚠️ Limitations

* Skill extraction depends on predefined taxonomy
* Proxy relevance is rule-based
* Intended as a **decision support tool**, not an automated hiring system

---

## 📜 License

**Internal / Proprietary**
Not intended for public redistribution.

---

## 👩‍💻 Development Team

* **Rishita Battula** – Backend, Matching Model, Data Pipeline
* **Kalidindi Ritika** – UI/UX, Evaluation Metrics, Reporting
* **Guide**: Selvaraj Vadivelu – Dean, Delivery Management

```
Just tell me 👍
```
