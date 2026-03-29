"""
main.py — FastAPI application for TRACE-ATS.

Endpoints:
  GET  /api/health        → health check
  GET  /api/jd/list       → grouped JD catalogue
  GET  /api/jd/{filename} → single JD text
  POST /api/score         → TRACE score for a resume + JD
  POST /api/compare       → all 4 model scores + signal breakdown
"""

import sys
from pathlib import Path

# Ensure backend modules are importable when run from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

from pdf_reader import read_pdf_bytes
from trace_algorithm import compare_models
from jd_service import list_jds, get_jd_text

# ── App ───────────────────────────────────────────────────────────
app = FastAPI(
    title="TRACE-ATS",
    description="Resume scoring via Lexical, Semantic, Checklist, Effort & Anomaly signals",
    version="1.0.0",
)

# Allow Streamlit frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API Routes ────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "engine": "TRACE-ATS v1.0"}


@app.get("/api/jd/list")
async def jd_list():
    """Return all JDs grouped by category (fresher / experienced / general)."""
    return list_jds()


@app.get("/api/jd/{filename}")
async def jd_detail(filename: str):
    """Return the raw text content of a specific JD file."""
    text = get_jd_text(filename)
    if text is None:
        raise HTTPException(status_code=404, detail=f"JD not found: {filename}")
    return {"filename": filename, "text": text}


@app.post("/api/score")
async def score_resume(
    file: UploadFile = File(...),
    jd_file: Optional[str] = Form(None),
    jd_text: Optional[str] = Form(None),
):
    """
    Score a resume PDF against a JD.
    Provide either `jd_file` (filename from /api/jd/list) OR `jd_text` (raw paste-in).
    Returns only the TRACE score.
    """
    jd = _resolve_jd(jd_file, jd_text)
    resume_text = await _extract_resume(file)
    result = compare_models(resume_text, jd)
    return {
        "trace_score": result["scores"]["trace"],
        "trace_percent": round(result["scores"]["trace"] * 100, 1),
    }


@app.post("/api/compare")
async def compare_resume(
    file: UploadFile = File(...),
    jd_file: Optional[str] = Form(None),
    jd_text: Optional[str] = Form(None),
):
    """
    Full comparison: all 4 scoring models + TRACE signal breakdown.
    Provide either `jd_file` OR `jd_text`.
    """
    jd = _resolve_jd(jd_file, jd_text)
    resume_text = await _extract_resume(file)
    return compare_models(resume_text, jd)


# ── Helpers ───────────────────────────────────────────────────────

def _resolve_jd(jd_file: Optional[str], jd_text: Optional[str]) -> str:
    """Resolve JD from file selection or raw text input."""
    if jd_text and jd_text.strip():
        return jd_text.strip()
    if jd_file:
        text = get_jd_text(jd_file)
        if text is None:
            raise HTTPException(status_code=404, detail=f"JD not found: {jd_file}")
        return text
    raise HTTPException(status_code=400, detail="Provide either jd_file or jd_text")


async def _extract_resume(file: UploadFile) -> str:
    """Extract text from uploaded PDF using PyMuPDF."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    data = await file.read()
    text = read_pdf_bytes(data)
    if not text.strip():
        raise HTTPException(status_code=422, detail="Could not extract text from PDF")
    return text

