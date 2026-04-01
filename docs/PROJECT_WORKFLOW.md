# Total Project Workflow — TRACE-ATS

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRACE-ATS SYSTEM                             │
│                                                                     │
│  ┌──────────────┐    HTTP/REST     ┌──────────────────────────────┐ │
│  │   FRONTEND   │◄───────────────►│         BACKEND              │ │
│  │  (Streamlit)  │                 │        (FastAPI)             │ │
│  │  Port: 8501   │                 │        Port: 8000            │ │
│  └──────────────┘                  └──────────────────────────────┘ │
│                                                                     │
│                                    ┌──────────────────────────────┐ │
│                                    │    JD Library (107 files)    │ │
│                                    │       /JD/*.txt              │ │
│                                    └──────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Input Ingestion

### 1.1 Resume Upload (PDF)

```
User uploads PDF ──► PyMuPDF (fitz) ──► Raw Text Extraction
                      │
                      ├── Iterates over all pages
                      ├── Extracts text preserving layout
                      └── Joins pages with newlines
```

**Module:** `backend/pdf_reader.py`
- Uses `pymupdf` (PyMuPDF v1.24+) for high-fidelity text extraction
- Accepts raw bytes from in-memory file uploads (no disk I/O)
- Handles multi-page PDFs with per-page extraction
- Returns clean joined text string

### 1.2 Job Description Input

Two input modes supported:

```
Mode 1: JD Library Selection                Mode 2: Custom Paste
┌─────────────────────┐                     ┌─────────────────────┐
│ JD/ directory        │                     │ User pastes raw text│
│ 107 .txt files       │                     │ into text area      │
│ Categorized:         │                     │                     │
│  • Fresher (7)       │                     │ Minimum 20 chars    │
│  • Experienced (7)   │                     │ required            │
│  • General (93)      │                     │                     │
└──────────┬──────────┘                     └──────────┬──────────┘
           │                                           │
           └───────── jd_service.py ───────────────────┘
                      │
                      ├── _classify(): Fresher / Experienced / General
                      ├── _role_name(): Filename → Human-readable title
                      └── list_jds(): Grouped catalogue with LRU caching
```

**Module:** `backend/jd_service.py`

---

## Phase 2: Text Preprocessing

```
Raw Text ──► clean_text() ──► Cleaned Text
              │
              ├── Convert to lowercase
              ├── Collapse newlines → spaces
              ├── Remove non-alphanumeric (preserve +, #, ., /)
              └── Normalize whitespace
```

**Module:** `backend/trace_algorithm.py :: clean_text()`

The cleaning preserves characters critical for technology names:
- `+` → C++, C#
- `#` → C#
- `/` → CI/CD
- `.` → Node.js, .NET

---

## Phase 3: Signal Computation (The TRACE Engine)

The core algorithm computes 6 independent signals, each measuring a different dimension of resume-JD alignment:

### Signal Pipeline

```
Cleaned Resume ─┐
                 │
Cleaned JD ─────┤
                 │
                 ▼
    ┌────────────────────────────────────────────────────────┐
    │              SIGNAL COMPUTATION ENGINE                 │
    │                                                        │
    │   ┌─────────┐  ┌─────────┐  ┌─────────┐              │
    │   │ L-Score  │  │ S-Score  │  │ C-Score  │              │
    │   │ Lexical  │  │ Semantic │  │ Checklist│              │
    │   │ TF-IDF   │  │ Embeddings│ │ Coverage │              │
    │   │ w=0.15   │  │ w=0.25   │  │ w=0.30   │              │
    │   └─────────┘  └─────────┘  └─────────┘              │
    │                                                        │
    │   ┌─────────┐  ┌─────────┐  ┌─────────┐              │
    │   │ E-Score  │  │ A-Score  │  │ X-Score  │              │
    │   │ Effort   │  │ Anomaly  │  │ Experience│             │
    │   │ Quality  │  │ Penalty  │  │ Mismatch │              │
    │   │ w=0.15   │  │ w=-0.05  │  │ w=-0.10  │              │
    │   └─────────┘  └─────────┘  └─────────┘              │
    │                                                        │
    │   ─────────────────────────────────────────            │
    │   TRACE = 0.15L + 0.25S + 0.30C + 0.15E               │
    │          − 0.05A − 0.10X                               │
    │   Clamped to [0.0, 1.0]                                │
    └────────────────────────────────────────────────────────┘
```

---

### 3.1 L-Score: Lexical Overlap (Weight: +15%)

```
Resume + JD ──► TfidfVectorizer ──► Cosine Similarity ──► Normalize
                │
                ├── ngram_range = (1, 2)     │ Capture "machine learning"
                ├── sublinear_tf = True       │ Dampen frequency dominance
                ├── max_features = 5000       │ Vocabulary cap
                └── min_df = 1                │ Include rare terms

Normalization: score / 0.18 (capped at 1.0)
  └── 0.18 threshold calibrated for short documents (< 400 words)
```

### 3.2 S-Score: Semantic Relevance (Weight: +25%)

```
Resume ──► Chunking (200 words, 50% overlap) ──► BAAI/bge-large-en-v1.5 Encoding
                                                    │
JD ────► Retrieval Query Prefix ──────────────────► BAAI/bge-large-en-v1.5 Encoding
                                                    │
                                              Max-Pool Cosine Similarity
                                                    │
                                              Normalize: score / 0.65

Key Details:
  • Model: BAAI/bge-large-en-v1.5 (1024-dim, 512-token max)
  • Hardware: Apple MPS / CUDA / CPU fallback
  • Chunking prevents silent truncation of long resumes
  • Query prefix: "Represent this sentence for searching relevant passages: "
```

### 3.3 C-Score: Checklist Coverage (Weight: +30%)

```
JD Text ──► extract_skills_from_jd()
             │
             ├── Step 1: PHRASE_LEXICON scan    (~35 multi-word phrases)
             │   e.g., "machine learning", "deep learning", "data science"
             │
             ├── Step 2: TECH_LEXICON scan      (~100 single-word terms)
             │   e.g., "python", "tensorflow", "docker", "kubernetes"
             │   Filtered against JD_NOISE (~120 noise words)
             │
             └── Step 3: SOFT_SKILLS scan       (~13 soft skills)
                 e.g., "communication", "leadership", "teamwork"
             │
             ▼
        Extracted Skills List (up to 20)
             │
             ▼
Resume ────► Substring Match Against Each Skill
             │
             ├── tech_coverage = tech_matched / tech_total
             └── soft_coverage = soft_matched / soft_total
             │
             ▼
        Dynamic Weighting:
          • Both present: 0.80 × tech + 0.20 × soft
          • Only tech:    1.00 × tech
          • Only soft:    1.00 × soft
```

### 3.4 E-Score: Effort & Quality (Weight: +15%)

```
Resume ──► Word Count ──► Bell-Curve Scoring
            │
            ├── < 150 words:   0.0 → 0.5   (poor)
            ├── 150–300 words: 0.5 → 0.8   (okay)
            ├── 300–700 words: 0.8 → 1.0   (optimal: "Goldilocks zone")
            ├── 700–1000 words: 1.0 → 0.8   (slightly long)
            └── > 1000 words:  0.8 → 0.6   (too long)
            │
            ▼
        + Section Coverage Bonus
            │
            ├── Detects: skills, experience, education, projects, summary, certifications
            └── Bonus: +0.05 per key section (up to +0.20)
            │
            ▼
        Capped at 1.0
```

### 3.5 A-Score: Anomaly Detection (Weight: −5%)

```
Resume Tokens ──► Filter Stopwords (sklearn + custom safe list)
                   │
                   ├── Remove tokens ≤ 2 characters
                   └── Remove tokens in _ANOMALY_SAFE set
                   │
                   ▼
             Frequency Counter on Meaningful Tokens
                   │
                   ├── Stuffed = tokens appearing > 5 times
                   ├── raw_ratio = stuffed_count / total_meaningful
                   └── diversity_penalty = unique_stuffed / total_unique
                   │
                   ▼
             Score = 0.7 × raw_ratio + 0.3 × diversity_penalty
             Capped at 1.0
```

### 3.6 X-Score: Experience Mismatch (Weight: −10%)

```
JD ──► Regex Extraction ──► Required Years
        │
        ├── "5+ years of experience"
        ├── "minimum 3 years"
        ├── "3-5 years"
        └── Takes MAX of all matches
        │
Resume ──► Regex Extraction + Date Range Inference ──► Claimed Years
            │
            ├── Pattern: "5 years of experience"
            ├── Pattern: "over 3 years"
            └── Date Ranges: "2019 – 2024" → 5 years
            │
            ▼
       Gap = JD_years − Resume_years
            │
            ├── gap ≤ 0:  0.0  (meets requirement)
            ├── gap = 1:  0.2  (minor shortfall)
            ├── gap = 2:  0.5  (moderate shortfall)
            ├── gap = 3:  0.75 (significant shortfall)
            ├── gap ≥ 4:  1.0  (severely underqualified)
            │
            Special Cases:
            ├── JD has no requirement:     0.0 (no penalty)
            └── Resume doesn't mention:   0.4 (benefit of doubt)
```

---

## Phase 4: Composite Scoring

```
                    TRACE Formula
                    ═════════════
    TRACE = W_L·L + W_S·S + W_C·C + W_E·E + W_A·A + W_X·X
          = 0.15L + 0.25S + 0.30C + 0.15E − 0.05A − 0.10X

    Positive Signals (sum = 0.85):
      L (15%)  →  Lexical TF-IDF overlap
      S (25%)  →  Semantic embedding alignment
      C (30%)  →  Skill checklist coverage
      E (15%)  →  Resume quality & structure

    Penalty Signals (sum = −0.15):
      A (−5%)  →  Keyword stuffing detection
      X (−10%) →  Experience gap penalty

    Final Score: clamped to [0.0, 1.0]
```

**Also computed (for comparison):**
- Lexical Only: `L` raw
- Semantic Only: `S` raw
- Naive Hybrid: `0.5L + 0.5S`

---

## Phase 5: API Response & Frontend Rendering

### 5.1 API Endpoint

```
POST /api/compare
  ├── Input:  resume PDF + JD (file selection or text)
  └── Output: JSON response
       │
       ├── scores:          { lexical_only, semantic_only, naive_hybrid, trace }
       ├── signals:         { L, S, C, E, A, X } with name, value, weight
       ├── experience:      { jd_years_required, resume_years_found, gap, status }
       ├── extracted_skills: [...]
       ├── matched_skills:   [...]
       ├── missing_skills:   [...]
       └── resume_word_count: int
```

### 5.2 Frontend Rendering

```
┌────────────────────────────────────────────────────────────────┐
│                     TRACE-ATS FRONTEND                         │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 1. SCORE HERO                                             │  │
│  │    Large percentage display + Verdict badge               │  │
│  │    (Excellent ≥75% / Good ≥55% / Average ≥35% / Poor)    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 2. SIGNAL BREAKDOWN                                       │  │
│  │    6 cards: L | S | C | E | A | X                         │  │
│  │    Each shows: % value, weight, progress bar              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 3. MODEL COMPARISON                                       │  │
│  │    4 metrics: Lexical | Semantic | Hybrid | TRACE         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 4. JD SKILL ANALYSIS                                      │  │
│  │    ✅ Matched Skills (green chips)                         │  │
│  │    ❌ Missing Skills (red chips)                           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 5. EXPERIENCE ANALYSIS                                    │  │
│  │    JD Requires | Resume Shows | Gap | Status Badge        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 6. META STATS                                             │  │
│  │    Resume Words | JD Skills Found | Skills Matched | Score│  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

---

## Complete Data Flow (End-to-End)

```
┌──────────┐     ┌──────────────┐     ┌─────────────────┐
│  User    │     │  Streamlit   │     │    FastAPI       │
│  Browser │     │  Frontend    │     │    Backend       │
└────┬─────┘     └──────┬───────┘     └────────┬────────┘
     │                  │                       │
     │ 1. Upload PDF    │                       │
     │ + Select JD      │                       │
     │─────────────────►│                       │
     │                  │ 2. POST /api/compare  │
     │                  │  (multipart/form-data)│
     │                  │──────────────────────►│
     │                  │                       │ 3. pdf_reader.read_pdf_bytes()
     │                  │                       │ 4. jd_service.get_jd_text()
     │                  │                       │ 5. trace_algorithm.clean_text()
     │                  │                       │ 6. Compute L, S, C, E, A, X
     │                  │                       │ 7. _compute_trace()
     │                  │                       │ 8. extract_skills_from_jd()
     │                  │  9. JSON Response     │
     │                  │◄──────────────────────│
     │ 10. Render UI    │                       │
     │◄─────────────────│                       │
     │  (Score Hero,    │                       │
     │   Signals,       │                       │
     │   Skills,        │                       │
     │   Experience)    │                       │
     │                  │                       │
```

---

## File Structure

```
Trace-ATS/
├── backend/
│   ├── main.py              # FastAPI application, routes, CORS
│   ├── trace_algorithm.py   # Core TRACE engine (6 signals + composer)
│   ├── pdf_reader.py        # PyMuPDF text extraction
│   └── jd_service.py        # JD file management & categorization
│
├── frontend/
│   └── app.py               # Streamlit UI (custom CSS, glassmorphism)
│
├── JD/                      # 107 job description text files
│   ├── *_Fresher.txt        # Entry-level JDs
│   ├── *_NonFresher.txt     # Experienced JDs
│   └── *-job-description.txt # General JDs
│
├── pyproject.toml           # Dependencies (uv managed)
├── uv.lock                  # Locked dependency graph
├── .env                     # HuggingFace token
└── README.md                # Project documentation
```

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Runtime** | Python 3.11+ | Core language |
| **Package Manager** | uv | Fast Rust-based dependency resolution |
| **Backend Framework** | FastAPI | Async HTTP API server |
| **ASGI Server** | Uvicorn | Production-grade ASGI |
| **PDF Extraction** | PyMuPDF (fitz) | High-fidelity PDF text parsing |
| **TF-IDF** | Scikit-Learn | TfidfVectorizer, cosine_similarity |
| **Embeddings** | Sentence-Transformers | BAAI/bge-large-en-v1.5 |
| **Deep Learning** | PyTorch | Tensor ops, MPS/CUDA acceleration |
| **Frontend** | Streamlit | Rapid prototyping UI framework |
| **Schema Validation** | Pydantic | Request/response validation |
| **Config** | python-dotenv | Environment variable management |

---

*Document prepared for: TRACE-ATS Project — Academic Submission*
*Last Updated: April 2026*
