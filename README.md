<div align=\"center\">

# ⚡ TRACE-ATS
**An Advanced, Multi-Signal Resume Scoring Engine**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![BAAI Embeddings](https://img.shields.io/badge/Model-BAAI%2Fbge--large--en--v1.5-yellow)](https://huggingface.co/BAAI/bge-large-en-v1.5)
[![PyTorch MPS](https://img.shields.io/badge/PyTorch-MPS%20Acceleration-ee4c2c?logo=pytorch)](https://pytorch.org/)

TRACE is a high-performance Applicant Tracking System (ATS) that moves beyond simplistic keyword-matching. It uses a heavily optimized 5-signal deterministic and semantic algorithm to score resumes against Job Descriptions (JDs) with human-like accuracy.

[Features](#-key-features) • [The Architecture](#-the-trace-algorithm) • [Tech Stack](#-tech-stack) • [Installation](#-installation--running)

</div>

---

## 🎯 Key Features

- **Beyond Keyword Matching:** Most legacy ATS systems fail by blindly matching text. TRACE utilizes a hybrid approach, blending deterministic checks with state-of-the-art **Semantic Embeddings (BAAI/bge-large)** to measure conceptual relevance.
- **Context-Aware PDF Extraction:** Utilizes `PyMuPDF` to ensure high-fidelity text extraction without destroying layout structures, table metadata, or spacing.
- **Dynamic Lexicon Skill Extraction:** Analyzes Job Descriptions on the fly to build structured tech requirements, separating hard technical skills from soft skills dynamically.
- **Keyword Stuffing Penalties:** Anomaly detection algorithms actively penalize resumes designed solely to trick legacy ATS parsers by stuffing hidden keywords.
- **Gorgeous UI:** A complete, decoupled frontend built in Streamlit featuring custom-injected CSS, providing a glassmorphism dark-mode aesthetic. 

---

## 🧠 The TRACE Algorithm

The core of the engine computes the final matching percentage by combining five distinct signals using deeply calibrated weights:

### `L` • Lexical Overlap (25%)
A standard cosine similarity check using **TF-IDF**. 
* **Optimization:** Runs on `(1, 2)` n-grams to capture multi-word technologies (e.g., "Machine Learning") instead of breaking them into unrelated terms. Utilizes *sublinear term-frequency scaling* to prevent users from artificially boosting scores by repeating a word endlessly.

### `S` • Semantic Relevance (25%)
Measures the contextual alignment between the resume and JD using the **BAAI/bge-large-en-v1.5** transformer model.
* **Optimization:** Standard transformer models silently truncate documents after 512 tokens. TRACE intelligently chunks long resumes into overlapping 200-word segments, embedding each segment individually and taking the **Max-Pool Similarity**, guaranteeing that deep historical experience isn't ignored. Fully accelerated via Apple MPS / CUDA.

### `C` • Checklist Factor (30%)
Calculates the raw coverage of explicit job requirements.
* **Optimization:** Instead of naively searching for the most frequent words (which often results in extracting generic noise like "team" or "experience"), TRACE uses a curated `TECH_LEXICON`. It extracts *only* verified frameworks, languages, and methodologies, weighting hard tech methodologies 3x heavier than soft skills.

### `E` • Effort & Quality (15%)
Rewards the completeness of the resume structure.
* **Optimization:** Evaluates document length on a strict **bell-curve**. Resumes under 150 words are heavily penalized, while resumes over 1,000 words take a slight brevity penalty. The 300-700 "Goldilocks" zone receives optimal scores, boosted further by Regex classifiers dynamically verifying the presence of core sections (Education, Projects, Skills).

### `A` • Anomaly Detection (-5% Penalty)
Prevents malicious resume engineering.
* **Optimization:** Scans for statistically abnormal token repetition rates specifically targeting non-stopwords. If a resume stuffs a framework name $>5$ times into white text, TRACE detects the anomaly severity and diversity to apply a targeted algorithmic penalty.

---

## 💻 Tech Stack

**Backend (API & Engine):**
- **FastAPI:** High-performance async routing.
- **PyTorch & Sentence-Transformers:** Handling the heavy semantic mapping via `bge-large` with hardware acceleration.
- **Scikit-Learn:** Mathematical implementations of vectorizers, pairwise cosine metrics, and NLP feature extraction.
- **PyMuPDF (fitz):** The gold standard in Python for PDF byte-stream extraction.
- **uv:** Ultra-fast, localized Rust-based package resolution.

**Frontend:**
- **Streamlit:** Allows rapid prototyping wrapped in heavily customized CSS to create a bespoke, modern interface.

---

## 🚀 Installation & Running

This project uses `uv` for lightning-fast dependency management, ensuring the massive data science dependencies (PyTorch, SciPy, Scikit-learn) are installed securely without polluting your global Python environment.

### 1. Setup the Environment
```bash
# Clone the repository
git clone https://github.com/Praroop1435/TRACE-ATS.git
cd TRACE-ATS

# Ensure you have uv installed (curl -LsSf https://astral.sh/uv/install.sh | sh)
uv sync
```

### 2. Start the Backend API (Scoring Engine)
```bash
# Runs on localhost:8000
uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Start the Frontend Application
In a **new terminal tab**, run:
```bash
# Runs on localhost:8501
uv run streamlit run frontend/app.py --server.port 8501
```

> **Note on Initial Run:** The first time the backend starts, it will download the ~1.3GB BAAI transformer weights. Authentication automatically passes through via `.env` injection.

---
<div align=\"center\">
<i>Designed & engineered rigorously to solve real-world talent acquisition bottlenecks.</i>
</div>
