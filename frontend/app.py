"""
TRACE-ATS — Streamlit Frontend
Calls the FastAPI backend at http://localhost:8000
"""

import streamlit as st
import requests

API_BASE = "http://localhost:8000"

# ── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="TRACE — ATS",
    page_icon="⚡",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark theme tweaks */
    .stApp { background-color: #0a0a0f; }

    /* Header gradient */
    .trace-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem;
    }
    .trace-header h1 {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #06b6d4, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .trace-header p {
        color: #6b7280;
        font-size: 0.95rem;
    }

    /* Badge */
    .engine-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 14px;
        background: rgba(6, 182, 212, 0.08);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 99px;
        font-size: 0.75rem;
        color: #06b6d4;
        font-weight: 500;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }
    .engine-badge .dot {
        width: 6px; height: 6px;
        border-radius: 50%;
        background: #10b981;
        display: inline-block;
    }

    /* Score hero */
    .score-hero {
        text-align: center;
        padding: 2rem;
        background: rgba(18, 18, 30, 0.65);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        margin-bottom: 1.5rem;
    }
    .score-big {
        font-size: 4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #06b6d4, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1;
    }
    .score-label {
        color: #6b7280;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 4px;
    }

    /* Verdict badges */
    .verdict-excellent {
        display: inline-block; padding: 4px 16px; border-radius: 99px;
        background: rgba(16,185,129,0.12); color: #10b981;
        border: 1px solid rgba(16,185,129,0.2); font-weight: 500;
    }
    .verdict-good {
        display: inline-block; padding: 4px 16px; border-radius: 99px;
        background: rgba(6,182,212,0.12); color: #06b6d4;
        border: 1px solid rgba(6,182,212,0.2); font-weight: 500;
    }
    .verdict-average {
        display: inline-block; padding: 4px 16px; border-radius: 99px;
        background: rgba(245,158,11,0.12); color: #f59e0b;
        border: 1px solid rgba(245,158,11,0.2); font-weight: 500;
    }
    .verdict-poor {
        display: inline-block; padding: 4px 16px; border-radius: 99px;
        background: rgba(244,63,94,0.12); color: #f43f5e;
        border: 1px solid rgba(244,63,94,0.2); font-weight: 500;
    }

    /* Signal card */
    .signal-card {
        background: rgba(18, 18, 30, 0.65);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
    }
    .signal-card h4 { color: #9ca3af; font-size: 0.82rem; font-weight: 500; margin-bottom: 0.5rem; }

    /* Skill chips */
    .skill-chip {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 99px;
        font-size: 0.78rem;
        font-weight: 500;
        border: 1px solid rgba(255,255,255,0.06);
        background: #12121a;
        color: #9ca3af;
        margin: 3px 3px;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    div[data-testid="stDecoration"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ── Header ───────────────────────────────────────────────────────
st.markdown("""
<div class="trace-header">
    <span class="engine-badge"><span class="dot"></span> Engine Ready</span>
    <h1>TRACE — ATS</h1>
    <p>Multi-signal resume scoring: TF-IDF lexical matching, semantic embeddings,<br>
    skill coverage, effort analysis & anomaly detection.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Load JD List ─────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_jd_list():
    try:
        resp = requests.get(f"{API_BASE}/api/jd/list", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


jd_data = load_jd_list()

# ── Input Section ────────────────────────────────────────────────
col_upload, col_jd = st.columns(2)

with col_upload:
    st.subheader("📄 Upload Resume")
    uploaded_file = st.file_uploader(
        "Drop your resume PDF",
        type=["pdf"],
        help="Only PDF files supported (via PyMuPDF)",
    )
    if uploaded_file:
        st.success(f"✓ {uploaded_file.name}")

with col_jd:
    st.subheader("📋 Job Description")
    jd_mode = st.radio(
        "Source",
        ["Select from library", "Paste custom JD"],
        horizontal=True,
        label_visibility="collapsed",
    )

    jd_file_selected = None
    jd_text_custom = None

    if jd_mode == "Select from library":
        if jd_data:
            # Build options with category labels
            options = []
            option_map = {}

            for cat, label in [("fresher", "🟢 Fresher"), ("experienced", "🔵 Experienced"), ("general", "⚪ General")]:
                items = jd_data.get(cat, [])
                for item in items:
                    display = f"{label}  ›  {item['role']}"
                    options.append(display)
                    option_map[display] = item["filename"]

            selected = st.selectbox("Choose a job description", options, index=None, placeholder="Search or select a JD…")
            if selected:
                jd_file_selected = option_map[selected]
        else:
            st.warning("⚠️ Could not load JD list. Is the backend running?")
    else:
        jd_text_custom = st.text_area(
            "Paste the full job description",
            height=180,
            placeholder="Paste the job description text here…",
        )

# ── Analyze Button ───────────────────────────────────────────────
st.markdown("")

can_analyze = uploaded_file is not None and (
    jd_file_selected is not None or (jd_text_custom and len(jd_text_custom.strip()) > 20)
)

if st.button("⚡ Analyze Resume", type="primary", use_container_width=True, disabled=not can_analyze):
    with st.spinner("Running TRACE analysis…"):
        try:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            form_data = {}

            if jd_file_selected:
                form_data["jd_file"] = jd_file_selected
            else:
                form_data["jd_text"] = jd_text_custom.strip()

            resp = requests.post(f"{API_BASE}/api/compare", files=files, data=form_data, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            st.session_state["results"] = data
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to backend. Start it with: `uv run uvicorn backend.main:app --reload`")
        except Exception as e:
            st.error(f"❌ {e}")

# ── Results ──────────────────────────────────────────────────────
if "results" in st.session_state:
    data = st.session_state["results"]
    trace_score = data["scores"]["trace"]
    pct = max(0, min(trace_score * 100, 100))

    st.markdown("---")

    # ── TRACE Score Hero ──
    if pct >= 75:
        verdict_html = '<span class="verdict-excellent">🏆 Excellent Match</span>'
    elif pct >= 55:
        verdict_html = '<span class="verdict-good">✅ Good Match</span>'
    elif pct >= 35:
        verdict_html = '<span class="verdict-average">⚠️ Average Match</span>'
    else:
        verdict_html = '<span class="verdict-poor">❌ Poor Match</span>'

    st.markdown(f"""
    <div class="score-hero">
        <div class="score-big">{pct:.1f}%</div>
        <div class="score-label">TRACE Score</div>
        <div style="margin-top: 1rem;">{verdict_html}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Signal Breakdown ──
    st.subheader("📡 Signal Breakdown")

    signals = data["signals"]
    signal_keys = list(signals.keys())
    cols = st.columns(len(signal_keys))

    signal_colors = {
        "L": "#06b6d4", "S": "#8b5cf6", "C": "#10b981",
        "E": "#f59e0b", "A": "#f43f5e", "X": "#ec4899",
    }

    for i, key in enumerate(signal_keys):
        sig = signals[key]
        val_pct = abs(sig["value"]) * 100
        weight_label = f"+{sig['weight']*100:.0f}%" if sig["weight"] > 0 else f"{sig['weight']*100:.0f}%"
        color = signal_colors.get(key, "#6b7280")

        with cols[i]:
            st.markdown(f"""
            <div class="signal-card">
                <h4>{sig['name']}</h4>
                <div style="font-size: 1.8rem; font-weight: 700; color: {color};">{val_pct:.1f}%</div>
                <div style="color: #6b7280; font-size: 0.75rem; margin-top: 4px;">Weight: {weight_label}</div>
            </div>
            """, unsafe_allow_html=True)

            st.progress(min(val_pct / 100, 1.0))

    # ── Model Comparison ──
    st.markdown("")
    st.subheader("📊 Model Comparison")

    models = [
        ("📝 Lexical Only (TF-IDF)", "lexical_only"),
        ("🧠 Semantic Only (SBERT)", "semantic_only"),
        ("⚖️ Naive Hybrid (50/50)", "naive_hybrid"),
        ("⚡ TRACE-ATS", "trace"),
    ]

    comp_cols = st.columns(len(models))
    for i, (label, key) in enumerate(models):
        score = data["scores"][key]
        score_pct = score * 100
        with comp_cols[i]:
            st.metric(label=label, value=f"{score_pct:.1f}%", delta=f"{score:.4f}")

    # ── Extracted Skills ──
    st.markdown("")
    st.subheader("🔑 JD Skill Analysis")

    matched = data.get("matched_skills", [])
    missing = data.get("missing_skills", [])
    all_skills = data.get("extracted_skills", [])

    if matched:
        st.markdown("**✅ Matched Skills:**")
        matched_html = " ".join(
            f'<span class="skill-chip" style="border-color: rgba(16,185,129,0.4); color: #10b981;">✓ {s}</span>'
            for s in matched
        )
        st.markdown(matched_html, unsafe_allow_html=True)

    if missing:
        st.markdown("**❌ Missing Skills:**")
        missing_html = " ".join(
            f'<span class="skill-chip" style="border-color: rgba(244,63,94,0.3); color: #f43f5e;">✗ {s}</span>'
            for s in missing
        )
        st.markdown(missing_html, unsafe_allow_html=True)

    # ── Experience Analysis ──
    exp_data = data.get("experience", {})
    if exp_data and exp_data.get("status") != "not_applicable":
        st.markdown("")
        st.subheader("📅 Experience Analysis")

        status = exp_data.get("status", "")
        jd_yrs = exp_data.get("jd_years_required")
        res_yrs = exp_data.get("resume_years_found")
        gap = exp_data.get("gap")

        # Status badge
        status_map = {
            "jd_no_requirement": ("ℹ️ JD does not specify experience", "#6b7280"),
            "resume_no_mention": ("⚠️ Resume does not mention experience", "#f59e0b"),
            "meets_requirement": ("✅ Meets or exceeds requirement", "#10b981"),
            "minor_shortfall": ("⚠️ Minor shortfall (1 year gap)", "#f59e0b"),
            "moderate_shortfall": ("⚠️ Moderate shortfall (2 year gap)", "#f97316"),
            "significant_shortfall": ("❌ Significant shortfall (3 year gap)", "#f43f5e"),
            "severely_underqualified": ("❌ Severely underqualified (4+ year gap)", "#dc2626"),
        }
        label, color = status_map.get(status, (status, "#6b7280"))

        exp_cols = st.columns(4)
        exp_cols[0].metric("JD Requires", f"{jd_yrs} yrs" if jd_yrs is not None else "—")
        exp_cols[1].metric("Resume Shows", f"{res_yrs} yrs" if res_yrs is not None else "—")
        exp_cols[2].metric("Gap", f"{gap} yrs" if gap is not None else "—")
        exp_cols[3].markdown(
            f'<div style="margin-top: 0.8rem; padding: 8px 16px; border-radius: 8px; '
            f'background: rgba(18,18,30,0.65); border: 1px solid {color}40; color: {color}; '
            f'font-weight: 500; text-align: center;">{label}</div>',
            unsafe_allow_html=True,
        )

    # ── Meta Stats ──
    st.markdown("")
    meta_cols = st.columns(4)
    meta_cols[0].metric("Resume Words", data.get("resume_word_count", "—"))
    meta_cols[1].metric("JD Skills Found", len(all_skills))
    meta_cols[2].metric("Skills Matched", f"{len(matched)}/{len(all_skills)}" if all_skills else "—")
    meta_cols[3].metric("TRACE Score", f"{pct:.1f}%")

