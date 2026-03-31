"""
trace_algorithm.py — TRACE-ATS scoring engine v2.1

Six signals:
  L  =  Lexical   (TF-IDF cosine with bigrams + sublinear TF)
  S  =  Semantic  (sentence-transformer with chunked encoding)
  C  =  Checklist (skill-aware JD keyword coverage using a tech lexicon)
  E  =  Effort    (optimal-range word count + section coverage)
  A  =  Anomaly   (keyword-stuffing penalty, stopword-aware)
  X  =  Experience Mismatch (penalty when resume experience < JD requirement)

TRACE score = 0.15*L + 0.25*S + 0.30*C + 0.15*E − 0.05*A − 0.10*X

v2.0 improvements over v1:
  • L: Added bigram (1,2)-grams and sublinear TF so multi-word phrases like
    "machine learning" are properly matched, and raw term frequency
    doesn't dominate.
  • S: Long texts are chunked into 200-word segments, each embedded
    separately, and the MAX similarity across chunks is used. This
    prevents the sentence-transformer from silently truncating resumes.
  • C: Instead of naive top-k frequency tokens, we now use a curated
    TECH_LEXICON + PHRASE_LEXICON to extract real skills/technologies
    from the JD. Generic words like "experience" and "team" are excluded.
  • E: Replaced the linear word-count ramp with an optimal-range curve
    (300–700 words → 1.0) plus bonuses for section completeness.
  • A: Now filters out English stopwords before counting repetitions,
    so common words like "the/and/in" don't inflate the penalty.
  • Weights rebalanced: C and S share top importance (skill match +
    domain relevance), L is supportive, E rewards quality, A is a
    gentle penalty only for real stuffing.
"""

import re
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

# Load .env so HF_TOKEN is picked up by huggingface_hub
load_dotenv()

# ── Model (loaded once at import time) ────────────────────────────
import torch
_device = "mps" if torch.backends.mps.is_available() else "cpu"
_model = SentenceTransformer("BAAI/bge-large-en-v1.5", device=_device)

# ── Technology / Skill Lexicons ───────────────────────────────────
# These ensure C_score extracts REAL skills, not generic words.

TECH_LEXICON = {
    # Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "ruby", "php", "scala", "kotlin", "swift", "r", "sql", "bash", "perl",
    "dart", "lua", "matlab", "julia",
    # ML / AI
    "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn", "xgboost",
    "lightgbm", "catboost", "huggingface", "transformers", "langchain",
    "llamaindex", "openai", "faiss", "pinecone", "chromadb", "weaviate",
    "onnx", "tensorrt", "triton", "mlflow", "wandb", "dvc", "ray",
    "sagemaker", "vertex",
    # Data
    "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly",
    "tableau", "powerbi", "excel", "spark", "hadoop", "hive", "presto",
    "airflow", "dagster", "prefect", "dbt", "kafka", "flink",
    "snowflake", "bigquery", "redshift", "databricks",
    # Web / Backend
    "react", "vue", "angular", "nextjs", "svelte",
    "tailwind", "bootstrap", "node", "express", "fastapi", "flask",
    "django", "spring", "rails", "graphql", "rest", "grpc",
    # DevOps / Cloud
    "docker", "kubernetes", "terraform", "ansible", "jenkins", "github",
    "gitlab", "circleci", "aws", "gcp", "azure", "linux", "nginx",
    "prometheus", "grafana", "datadog", "elk",
    # Databases
    "mysql", "postgresql", "postgres", "mongodb", "redis", "cassandra",
    "dynamodb", "neo4j", "sqlite",
    # Mobile
    "android", "ios", "flutter", "react native",
    # Blockchain
    "solidity", "ethereum", "web3", "hardhat", "foundry",
    # Concepts (important for matching)
    "nlp", "cv", "cnn", "rnn", "lstm", "gan", "bert", "gpt",
    "llm", "rag", "etl", "ci/cd", "api", "microservices",
    "devops", "mlops", "git",
}

PHRASE_LEXICON = {
    "machine learning", "deep learning", "natural language processing",
    "computer vision", "data science", "data engineering", "data analysis",
    "data visualization", "feature engineering", "model deployment",
    "time series", "recommendation systems", "reinforcement learning",
    "transfer learning",
    "cloud computing", "distributed systems", "object detection",
    "image classification", "sentiment analysis", "text classification",
    "speech recognition", "neural network", "decision tree",
    "random forest", "gradient boosting", "linear regression",
    "logistic regression", "support vector", "principal component",
    "a b testing", "statistical modeling", "business intelligence",
    "project management", "product management", "quality assurance",
    "version control", "continuous integration", "continuous deployment",
    "unit testing", "system design",
}

SOFT_SKILLS = {
    "communication", "leadership", "teamwork", "collaboration", "analytical",
    "problem solving", "critical thinking", "management", "organization",
    "presentation", "mentoring", "adaptability", "initiative",
}

# Words that appear in JDs but are NOT real skills
JD_NOISE = {
    "job", "description", "requirements", "required", "preferred",
    "responsibilities", "role", "position", "experience", "strong",
    "excellent", "knowledge", "ability", "team", "teams", "skills",
    "understanding", "familiarity", "proficiency", "working", "proven",
    "years", "year", "plus", "like", "similar", "including", "using",
    "work", "develop", "developing", "build", "building", "create",
    "creating", "design", "designing", "implement", "implementing",
    "manage", "managing", "support", "maintain", "maintaining", "lead",
    "leading", "ensure", "help", "assist", "field", "degree",
    "bachelor", "master", "bsc", "msc", "btech", "mtech", "phd",
    "relevant", "related", "minimum", "equivalent", "candidate",
    "apply", "company", "organization", "industry",
    "environment", "dynamic", "drive", "driven", "self", "starter",
    "passionate", "motivated", "detail", "oriented", "cross", "functional",
    "daily", "basis", "best", "practices", "highly", "written", "verbal",
    "computer", "science", "engineering", "math", "physics", "quantitative",
    "looking", "seeking", "join", "talented", "engineers", "developers",
    "scientists", "innovative", "cutting", "edge", "technologies",
    "technology", "tools", "stack", "agile", "scrum", "css", "html",
    "full", "front", "end", "back", "web", "development", "prompt",
    "creativity", "opportunity", "benefits", "salary", "bonus", "equity",
    "remote", "hybrid", "office", "location", "based", "demonstrated",
    "demonstrable", "exceptional", "outstanding", "solid", "track", "record",
    "collaborative", "inclusive", "mission", "vision", "values", "culture",
    "diversity", "expert", "expertise", "good", "great", "fluent", "hands",
    "on", "elasticsearch", "solving", "problem", "paced", "fast","rest"
}

# ── Text Cleaning ─────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s\+\#\./]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ── Signal: L (Lexical) ──────────────────────────────────────────

def L_score(resume: str, jd: str) -> float:
    """
    Lexical similarity via TF-IDF cosine.
    v2: Uses (1,2)-grams and sublinear TF for better phrase matching.
    v2.1: Lowered normalization ceiling — short resumes (< 400 words)
    naturally produce lower raw cosines, so 0.18 is a fairer threshold.
    """
    vec = TfidfVectorizer(
        ngram_range=(1, 2),       # capture "machine learning" etc.
        sublinear_tf=True,        # dampen raw frequency dominance
        min_df=1,
        max_features=5000,
    )
    tfidf = vec.fit_transform([resume, jd])
    score = float(cosine_similarity(tfidf[0], tfidf[1])[0][0])
    # Normalize: raw TF-IDF cosine between resume/JD is typically 0.05–0.30
    # For short texts the ceiling is lower; 0.18+ maps to ~1.0
    return min(1.0, score / 0.18)


# ── Signal: S (Semantic) ─────────────────────────────────────────

def _chunk_text(text: str, chunk_words: int = 200) -> list[str]:
    """Split text into overlapping chunks for better embedding coverage."""
    words = text.split()
    if len(words) <= chunk_words:
        return [text]
    chunks = []
    step = chunk_words // 2  # 50% overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_words])
        if len(chunk.split()) > 20:  # skip tiny tail chunks
            chunks.append(chunk)
    return chunks


def S_score(resume: str, jd: str) -> float:
    """
    Semantic similarity via sentence-transformer embeddings.
    v2: Chunks long resumes to avoid silent truncation, takes MAX similarity.
    Model upgrade: Uses BAAI/bge-large-en-v1.5 via Apple MPS.
    """
    resume_chunks = _chunk_text(resume, chunk_words=200)
    
    # BGE model requires an instruction for retrieval queries
    query = "Represent this sentence for searching relevant passages: " + jd
    jd_emb = _model.encode([query])[0]

    best = 0.0
    for chunk in resume_chunks:
        chunk_emb = _model.encode([chunk])[0]
        dot = np.dot(chunk_emb, jd_emb)
        norms = np.linalg.norm(chunk_emb) * np.linalg.norm(jd_emb)
        if norms > 0:
            sim = float(dot / norms)
            best = max(best, sim)

    # Semantic cosine for same-domain texts is typically 0.35–0.75
    # Normalize so 0.65+ maps to ~1.0
    return min(1.0, max(0.0, best / 0.65))


# ── Signal: C (Checklist) ────────────────────────────────────────

def extract_skills_from_jd(jd_text: str, top_k: int = 20) -> list[str]:
    """
    Extract real skills/technologies from JD text.
    v2: Uses curated TECH_LEXICON + PHRASE_LEXICON instead of raw frequency.
    Returns up to top_k matched skills.
    """
    jd_lower = jd_text.lower()
    found_skills: list[str] = []

    # 1. Multi-word phrases first (highest value)
    for phrase in PHRASE_LEXICON:
        if phrase in jd_lower:
            found_skills.append(phrase)

    # 2. Single-word tech terms
    jd_clean = clean_text(jd_text)
    tokens = set(jd_clean.split())
    for token in tokens:
        if token in TECH_LEXICON and token not in JD_NOISE:
            if token not in found_skills:
                found_skills.append(token)

    # 3. Soft skills (lower priority)
    for skill in SOFT_SKILLS:
        if skill in jd_lower and skill not in found_skills:
            found_skills.append(skill)

    return found_skills[:top_k]


def C_score(resume: str, jd: str) -> float:
    """
    Checklist coverage: fraction of JD-extracted skills found in resume.
    v2.1: Uses lexicon-based extraction with DYNAMIC weighting.
    If the JD has no soft skills, 100% weight goes to tech coverage
    so a perfect tech match yields 1.0 (not 0.80).
    """
    resume_lower = resume.lower()
    skills = extract_skills_from_jd(jd)

    if not skills:
        return 0.5  # neutral if JD has no extractable skills

    tech_matched = 0
    tech_total = 0
    soft_matched = 0
    soft_total = 0

    for skill in skills:
        is_soft = skill in SOFT_SKILLS
        if is_soft:
            soft_total += 1
            if skill in resume_lower:
                soft_matched += 1
        else:
            tech_total += 1
            if skill in resume_lower:
                tech_matched += 1

    tech_coverage = (tech_matched / max(1, tech_total))
    soft_coverage = (soft_matched / max(1, soft_total))

    # Dynamic weighting: if one category is absent, give full weight to the other
    if tech_total == 0 and soft_total == 0:
        return 0.5
    elif soft_total == 0:
        return tech_coverage          # 100% tech-driven
    elif tech_total == 0:
        return soft_coverage           # 100% soft-driven
    else:
        # Blend: 80% tech, 20% soft when both present
        return 0.80 * tech_coverage + 0.20 * soft_coverage


# ── Signal: E (Effort / Quality) ─────────────────────────────────

_SECTION_KEYWORDS = {
    "skills":        ["skills", "technical skills", "tech stack", "competencies", "technologies"],
    "experience":    ["experience", "work experience", "professional experience", "employment"],
    "education":     ["education", "academics", "qualifications", "degree"],
    "projects":      ["projects", "project work", "personal projects", "portfolio"],
    "summary":       ["summary", "objective", "about me", "profile"],
    "certifications":["certifications", "certificates", "courses", "licenses"],
}


def _detect_sections(resume: str) -> dict[str, bool]:
    """Detect which resume sections are present."""
    resume_lower = resume.lower()
    found = {}
    for section, keywords in _SECTION_KEYWORDS.items():
        found[section] = any(kw in resume_lower for kw in keywords)
    return found


def E_score(resume: str) -> float:
    """
    Effort / Quality heuristic.
    v2: Combines optimal word-count range + section coverage bonus.

    Word count scoring:
      < 150 words  → poor (linear ramp 0.0–0.5)
      150–300      → okay (0.5–0.8)
      300–700      → optimal (0.8–1.0)
      700–1000     → slightly long (1.0–0.8)
      > 1000       → too long (0.8–0.6)

    Section bonus: +0.1 for each key section found (skills, experience,
    education, projects), capped so total E ≤ 1.0.
    """
    words = len(resume.split())

    # Word-count curve
    if words < 150:
        wc_score = (words / 150) * 0.5
    elif words < 300:
        wc_score = 0.5 + ((words - 150) / 150) * 0.3
    elif words <= 700:
        wc_score = 0.8 + ((words - 300) / 400) * 0.2
    elif words <= 1000:
        wc_score = 1.0 - ((words - 700) / 300) * 0.2
    else:
        wc_score = max(0.6, 0.8 - ((words - 1000) / 1000) * 0.2)

    # Section coverage bonus
    sections = _detect_sections(resume)
    key_sections = ["skills", "experience", "education", "projects"]
    section_count = sum(sections.get(s, False) for s in key_sections)
    section_bonus = section_count * 0.05  # up to +0.20

    return min(1.0, wc_score + section_bonus)


# ── Signal: A (Anomaly) ──────────────────────────────────────────

# Extended stopwords for anomaly detection
_ANOMALY_SAFE = ENGLISH_STOP_WORDS | {
    "the", "and", "for", "with", "using", "used", "work", "worked",
    "based", "data", "system", "systems", "new", "also", "well",
    "including", "various", "project", "projects", "developed",
    "built", "experience", "team", "skills",
}


def A_score(resume: str) -> float:
    """
    Anomaly score: detects keyword stuffing.
    v2: Only counts NON-stopword tokens repeated excessively (>5 times).
    Uses a ratio relative to unique token count for fairness.
    """
    tokens = resume.lower().split()
    if not tokens:
        return 0.0

    # Only count meaningful tokens
    meaningful = [t for t in tokens if t not in _ANOMALY_SAFE and len(t) > 2]
    if not meaningful:
        return 0.0

    freq = Counter(meaningful)
    unique_count = len(freq)

    # Count tokens that appear suspiciously often (>5 times)
    stuffed_tokens = {t: c for t, c in freq.items() if c > 5}
    stuffed_count = sum(stuffed_tokens.values())

    # Ratio of stuffed occurrences to total meaningful tokens
    raw_ratio = stuffed_count / len(meaningful)

    # Also consider: how many DIFFERENT tokens are stuffed
    diversity_penalty = len(stuffed_tokens) / max(1, unique_count)

    # Blend: mostly raw ratio, with diversity as a secondary signal
    score = 0.7 * raw_ratio + 0.3 * diversity_penalty

    return min(1.0, score)


# ── Signal: X (Experience Mismatch) ──────────────────────────────

# Patterns to extract years-of-experience from JD
_JD_EXP_PATTERNS = [
    # "5+ years", "5-7 years", "minimum 3 years", "at least 2 years"
    r'(\d+)\s*\+?\s*(?:to|-|–)?\s*(?:\d+)?\s*years?\s+(?:of\s+)?(?:experience|exp\.?|professional)',
    r'(?:minimum|at\s+least|min\.?)\s+(\d+)\s*\+?\s*years?',
    r'(\d+)\s*\+?\s*years?\s+(?:of\s+)?(?:relevant|industry|hands[- ]?on|total|overall)',
    r'(\d+)\s*\+?\s*years?\s+(?:of\s+)?(?:experience|exp\.?)',
    r'(\d+)\s*-\s*\d+\s*years?',
]

# Patterns to extract years-of-experience from resume
_RESUME_EXP_PATTERNS = [
    # "5+ years of experience", "over 3 years"
    r'(\d+)\s*\+?\s*years?\s+(?:of\s+)?(?:experience|professional|industry)',
    r'(?:over|more\s+than)\s+(\d+)\s*years?',
    r'(\d+)\s*\+?\s*years?\s+(?:of\s+)?(?:relevant|hands[- ]?on|total)',
]


def _extract_jd_experience(jd: str) -> int | None:
    """Extract the required years of experience from a JD. Returns None if not found."""
    jd_lower = jd.lower()
    candidates: list[int] = []
    for pattern in _JD_EXP_PATTERNS:
        for m in re.finditer(pattern, jd_lower):
            val = int(m.group(1))
            if 0 < val <= 30:  # sanity check
                candidates.append(val)
    return max(candidates) if candidates else None


def _extract_resume_experience(resume: str) -> int | None:
    """Extract claimed years of experience from a resume. Returns None if not found."""
    resume_lower = resume.lower()
    candidates: list[int] = []
    for pattern in _RESUME_EXP_PATTERNS:
        for m in re.finditer(pattern, resume_lower):
            val = int(m.group(1))
            if 0 < val <= 40:
                candidates.append(val)
    # Also try to infer from date ranges (e.g., "2019 – 2024")
    date_ranges = re.findall(r'(20\d{2})\s*(?:–|-|to)\s*(20\d{2}|present|current)', resume_lower)
    import datetime
    current_year = datetime.datetime.now().year
    total_span = 0
    for start_str, end_str in date_ranges:
        start = int(start_str)
        end = current_year if end_str in ("present", "current") else int(end_str)
        span = max(0, end - start)
        total_span = max(total_span, span)  # take the longest implied span
    if total_span > 0:
        candidates.append(total_span)
    return max(candidates) if candidates else None


def X_score(resume: str, jd: str) -> tuple[float, dict]:
    """
    Experience mismatch penalty.
    Returns (penalty_value, details_dict).
      penalty_value: 0.0 = no penalty, 1.0 = maximum penalty
      details_dict: {jd_years, resume_years, gap, status}

    Penalty curve:
      gap <= 0 (meets or exceeds)  → 0.0
      gap == 1                      → 0.2  (minor shortfall)
      gap == 2                      → 0.5
      gap == 3                      → 0.75
      gap >= 4                      → 1.0  (severely underqualified)

    If the JD doesn't specify experience, no penalty is applied.
    If the resume doesn't state experience, a moderate penalty of 0.4 is applied
    (benefit of the doubt — ATS can't verify so light penalty).
    """
    jd_years = _extract_jd_experience(jd)
    resume_years = _extract_resume_experience(resume)

    details = {
        "jd_years_required": jd_years,
        "resume_years_found": resume_years,
        "gap": None,
        "status": "not_applicable",
    }

    # If JD doesn't specify experience, no penalty
    if jd_years is None:
        details["status"] = "jd_no_requirement"
        return 0.0, details

    # If resume doesn't mention experience at all
    if resume_years is None:
        details["status"] = "resume_no_mention"
        details["gap"] = jd_years
        return 0.4, details  # moderate penalty — can't verify

    gap = jd_years - resume_years
    details["gap"] = gap

    if gap <= 0:
        details["status"] = "meets_requirement"
        return 0.0, details
    elif gap == 1:
        details["status"] = "minor_shortfall"
        return 0.2, details
    elif gap == 2:
        details["status"] = "moderate_shortfall"
        return 0.5, details
    elif gap == 3:
        details["status"] = "significant_shortfall"
        return 0.75, details
    else:
        details["status"] = "severely_underqualified"
        return 1.0, details


# ── Composite Scoring ─────────────────────────────────────────────

# v2.1 WEIGHTS — rebalanced for more accurate ATS scoring:
#   C (30%): Skill coverage is the most actionable & reliable signal
#   S (25%): Semantic domain relevance catches what keywords miss
#   L (15%): Lexical TF-IDF is supportive but noisy for short docs
#   E (15%): Quality/completeness rewards well-structured resumes
#   A (−5%): Gentle stuffing penalty (only triggers on real abuse)
#   X (−10%): Experience mismatch penalty

W_L = 0.15
W_S = 0.25
W_C = 0.30
W_E = 0.15
W_A = -0.05
W_X = -0.10


def _compute_trace(l: float, s: float, c: float, e: float, a: float, x: float) -> float:
    """Compute the TRACE score from individual signals."""
    raw = W_L * l + W_S * s + W_C * c + W_E * e + W_A * a + W_X * x
    # Clamp to [0, 1]
    return max(0.0, min(1.0, raw))


def lexical_only(resume: str, jd: str) -> float:
    return L_score(resume, jd)


def semantic_only(resume: str, jd: str) -> float:
    return S_score(resume, jd)


def naive_hybrid(resume: str, jd: str) -> float:
    L = L_score(resume, jd)
    S = S_score(resume, jd)
    return 0.5 * L + 0.5 * S


def trace_score(resume: str, jd: str) -> float:
    """The TRACE-ATS composite score."""
    L = L_score(resume, jd)
    S = S_score(resume, jd)
    C = C_score(resume, jd)
    E = E_score(resume)
    A = A_score(resume)
    X, _ = X_score(resume, jd)
    return _compute_trace(L, S, C, E, A, X)


# ── Full Comparison (all 4 philosophies) ──────────────────────────

def compare_models(resume_raw: str, jd_raw: str) -> dict:
    """
    Run all 4 scoring approaches and return detailed breakdown.
    Inputs should be RAW text (cleaning is done internally).
    """
    resume = clean_text(resume_raw)
    jd = clean_text(jd_raw)

    # Individual signals for the TRACE breakdown
    l = L_score(resume, jd)
    s = S_score(resume, jd)
    c = C_score(resume, jd)
    e = E_score(resume)
    a = A_score(resume)
    x, x_details = X_score(resume, jd)

    trace = _compute_trace(l, s, c, e, a, x)

    # Also extract the raw skills for display
    skills = extract_skills_from_jd(jd_raw)

    # Determine which skills the resume matched
    resume_lower = resume.lower()
    matched_skills = [sk for sk in skills if sk in resume_lower]
    missing_skills = [sk for sk in skills if sk not in resume_lower]

    return {
        "scores": {
            "lexical_only": round(float(l), 4),
            "semantic_only": round(float(s), 4),
            "naive_hybrid": round(float(min(1.0, 0.5 * l + 0.5 * s)), 4),
            "trace": round(float(trace), 4),
        },
        "signals": {
            "L": {"name": "Lexical (TF-IDF Bigram)", "value": round(float(l), 4), "weight": W_L},
            "S": {"name": "Semantic (Chunked Embedding)", "value": round(float(s), 4), "weight": W_S},
            "C": {"name": "Checklist (Skill Coverage)", "value": round(float(c), 4), "weight": W_C},
            "E": {"name": "Effort (Quality + Sections)", "value": round(float(e), 4), "weight": W_E},
            "A": {"name": "Anomaly (Stuffing Penalty)", "value": round(float(a), 4), "weight": W_A},
            "X": {"name": "Experience (Mismatch Penalty)", "value": round(float(x), 4), "weight": W_X},
        },
        "experience": x_details,
        "extracted_skills": skills,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "resume_word_count": len(resume.split()),
    }
