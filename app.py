"""
VaultDocs AI: Local-First Contract Risk Screening Assistant
Professional 3-Column Legal Contract Analysis Dashboard
"""

import html
import os
import re
import tempfile
import time
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple

import ollama
import streamlit as st
from pydantic import ValidationError

# Suppress benign Hugging Face Hub unauthenticated request warning
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")

from models.schemas import AuditReport, RiskFinding, parse_audit_report
from reports.report_generator import generate_markdown_report
from services.citation_validator import CitationResult, validate_report_citations
from services.clause_splitter import ClauseSplitter
from services.pdf_parser import PDFParser
from services.retriever import FastClauseRetriever, get_dense_embeddings

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="VaultDocs AI — Legal Contract Analysis Dashboard",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Custom Styling: Modern Legal-Tech Aesthetic
# ---------------------------------------------------------
st.markdown(
    """
<style>
/* Base typography and dashboard spacing */
.main-header-title {
    font-size: 26px;
    font-weight: 800;
    letter-spacing: -0.6px;
    margin-bottom: 2px;
}
.main-header-subtitle {
    font-size: 13.5px;
    color: #64748b;
    margin-bottom: 16px;
}

/* Metric Cards in Risk Overview */
.metric-card {
    background: rgba(128, 128, 128, 0.05);
    border: 1px solid rgba(128, 128, 128, 0.16);
    border-radius: 10px;
    padding: 14px 18px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    height: 100%;
}
.metric-label {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    opacity: 0.75;
    margin-bottom: 4px;
}
.metric-value {
    font-size: 23px;
    font-weight: 800;
    letter-spacing: -0.5px;
}
.metric-sub {
    font-size: 11.5px;
    margin-top: 3px;
    font-weight: 500;
    opacity: 0.85;
}

/* Category & Priority Badges */
.badge-pill {
    display: inline-block;
    padding: 3px 9px;
    border-radius: 12px;
    font-size: 10.5px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    margin-right: 4px;
}
.badge-critical { background: rgba(239, 68, 68, 0.16); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.35); }
.badge-high { background: rgba(249, 115, 22, 0.16); color: #f97316; border: 1px solid rgba(249, 115, 22, 0.35); }
.badge-medium { background: rgba(234, 179, 8, 0.16); color: #eab308; border: 1px solid rgba(234, 179, 8, 0.35); }
.badge-low { background: rgba(6, 182, 212, 0.16); color: #06b6d4; border: 1px solid rgba(6, 182, 212, 0.35); }
.badge-cat { background: rgba(99, 102, 241, 0.12); color: #6366f1; border: 1px solid rgba(99, 102, 241, 0.28); }
.badge-verified { background: rgba(34, 197, 94, 0.16); color: #16a34a; border: 1px solid rgba(34, 197, 94, 0.35); }
.badge-unverified { background: rgba(239, 68, 68, 0.16); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.35); }

/* Left Column: Clause Card */
.clause-list-item {
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 8px;
    transition: all 0.15s ease-in-out;
}

/* Center Column: Document Viewer */
.doc-viewer-box {
    background: rgba(128, 128, 128, 0.04);
    border: 1px solid rgba(128, 128, 128, 0.18);
    border-radius: 8px;
    padding: 16px 20px;
    font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
    font-size: 12.5px;
    line-height: 1.65;
    white-space: pre-wrap;
    word-break: break-word;
}
.doc-page-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 12px;
    background: rgba(128, 128, 128, 0.08);
    border-radius: 6px;
    margin-bottom: 12px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.active-excerpt-callout {
    background: rgba(99, 102, 241, 0.1);
    border-left: 4px solid #6366f1;
    padding: 10px 14px;
    border-radius: 6px;
    margin-bottom: 12px;
    font-size: 12px;
}

/* Right Column: Clause Inspector */
.inspector-quote-box {
    background: rgba(128, 128, 128, 0.06);
    border-left: 3px solid #6366f1;
    border-radius: 6px;
    padding: 12px 14px;
    font-size: 12.5px;
    font-style: italic;
    line-height: 1.5;
    margin: 8px 0 14px 0;
}
.inspector-label {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #6366f1;
    margin-top: 12px;
    margin-bottom: 4px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Helper Functions & Scoring Mechanics
# ---------------------------------------------------------
@st.cache_data(ttl=60)
def get_ollama_models() -> Tuple[List[str], Set[str], bool]:
    """Retrieve locally installed Ollama models, prioritizing fast models while detecting connectivity."""
    default_models = ["llama3.2:1b", "gemma:2b", "qwen2.5:3b"]
    try:
        res = ollama.list()
        installed = []
        if hasattr(res, "models"):
            installed = [m.model for m in res.models]
        elif isinstance(res, dict) and "models" in res:
            installed = [m.get("name") or m.get("model") for m in res["models"]]

        all_models = list(installed)
        for m in default_models:
            if m not in all_models and f"{m}:latest" not in all_models:
                all_models.append(m)
        return all_models or default_models, set(installed), True
    except Exception:
        return default_models, set(), False


def get_priority_meta(priority: str) -> Dict[str, Any]:
    """Return risk score and styling metadata for a priority classification."""
    p = priority.strip().capitalize()
    if p in ["Critical", "Urgent"]:
        return {
            "score": 92,
            "badge_color": "#ef4444",
            "label": "CRITICAL",
            "icon": "🚨",
        }
    elif p in ["High", "Major"]:
        return {
            "score": 78,
            "badge_color": "#f97316",
            "label": "HIGH",
            "icon": "⚠️",
        }
    elif p in ["Medium", "Moderate"]:
        return {
            "score": 55,
            "badge_color": "#eab308",
            "label": "MEDIUM",
            "icon": "⚖️",
        }
    else:
        return {
            "score": 30,
            "badge_color": "#06b6d4",
            "label": "LOW",
            "icon": "ℹ️",
        }


def calculate_overall_risk(findings: List[RiskFinding]) -> Tuple[int, str, str]:
    """Compute overall contract risk score (0-100), risk status, and indicator color."""
    if not findings:
        return 12, "Low / Clean Triage", "#22c55e"

    total_weight = 0
    weights = {"critical": 35, "high": 25, "medium": 15, "low": 5}
    for f in findings:
        p = f.priority.lower()
        matched_weight = 10
        for k, v in weights.items():
            if k in p:
                matched_weight = v
                break
        total_weight += matched_weight

    score = min(98, max(15, total_weight))
    if score >= 70:
        return score, "High Risk Exposure", "#ef4444"
    elif score >= 40:
        return score, "Moderate Risk", "#f97316"
    else:
        return score, "Low Risk", "#22c55e"


# ---------------------------------------------------------
# Sidebar: Local Model & Document Ingestion
# ---------------------------------------------------------
st.sidebar.header("⚙️ Local AI Engine")
available_models, installed_models, ollama_online = get_ollama_models()

if not ollama_online:
    st.sidebar.error("🔴 **Ollama is offline or unreachable**\n\nRun in terminal:\n`ollama serve`")
else:
    st.sidebar.success("🟢 **Ollama Connected** (127.0.0.1:11434)")

# Smart default model selection: choose llama3.2:1b (fastest on CPU) or gemma:2b if available
default_model_idx = 0
if "llama3.2:1b" in installed_models:
    default_model_idx = available_models.index("llama3.2:1b")
elif "gemma:2b" in installed_models:
    default_model_idx = available_models.index("gemma:2b")
elif installed_models:
    first_inst = next(iter(installed_models))
    if first_inst in available_models:
        default_model_idx = available_models.index(first_inst)

model_choice = st.sidebar.selectbox(
    "Select Local Model",
    available_models,
    index=default_model_idx,
    help="Models run 100% locally on your machine via Ollama.",
)

if model_choice == "llama3.2:1b":
    st.sidebar.caption("⚡ **Fastest on CPU** (~45s inference, reliable JSON)")
elif model_choice == "gemma:2b":
    st.sidebar.caption("⚖️ **Balanced** (~45s inference)")
elif "qwen" in model_choice.lower():
    st.sidebar.caption("🧠 **In-Depth** (slower on CPU, ~85s inference)")

if installed_models and model_choice not in installed_models and f"{model_choice}:latest" not in installed_models:
    st.sidebar.warning(f"⚠️ `{model_choice}` not downloaded yet. Run:\n`ollama pull {model_choice}`")

temperature = st.sidebar.slider("Inference Temperature", 0.0, 0.4, 0.1, 0.05)

retrieval_mode = st.sidebar.radio(
    "Retrieval Engine",
    [
        "⚡ Fast Hybrid Ranker (Instant & Recommended)",
        "🧠 Dense Vector Search (FAISS)",
    ],
    index=0,
    help="Fast Hybrid Ranker executes instantly (<5ms) without PyTorch overhead. Dense Vector Search uses neural embeddings.",
)

st.sidebar.markdown("---")
st.sidebar.header("📁 Contract Document")
uploaded_file = st.sidebar.file_uploader(
    "Upload Agreement or NDA (PDF)",
    type=["pdf"],
    help="Contract is processed entirely in local memory with PyMuPDF.",
)

audit_focus = st.sidebar.selectbox(
    "Select Audit Focus Area",
    [
        "General Risk & Liability Summary",
        "Payment Terms & Late Penalties",
        "Intellectual Property (IP) Rights",
        "Termination & Notice Periods",
        "Indemnification & Warranties",
        "Non-Compete & Exclusivity Restrictions",
    ],
)

# ---------------------------------------------------------
# Session State Ingestion Handling
# ---------------------------------------------------------
if uploaded_file is not None:
    # Process PDF only if it's a new upload or not yet loaded in state
    if st.session_state.get("current_file_name") != uploaded_file.name:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        try:
            pages = PDFParser.extract_text_with_pages(tmp_path)
            chunks = ClauseSplitter.split_into_chunks(pages)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        st.session_state["pages"] = pages
        st.session_state["chunks"] = chunks
        st.session_state["current_file_name"] = uploaded_file.name
        st.session_state["audit_report"] = None
        st.session_state["citation_results"] = None
        st.session_state["selected_clause_idx"] = 0
        st.session_state["doc_page_idx"] = 0
        st.session_state["relevant_chunks"] = []
        st.session_state["faiss_index"] = None
        st.session_state["embedder"] = None
        st.session_state["audit_elapsed_sec"] = None

# Action trigger button in sidebar
run_audit_clicked = False
if uploaded_file and st.session_state.get("pages"):
    st.sidebar.success(f"Indexed {len(st.session_state['chunks'])} clauses ({len(st.session_state['pages'])} pages).")
    run_audit_clicked = st.sidebar.button(
        "🚀 Run Risk Screening Audit",
        type="primary",
        use_container_width=True,
    )

# ---------------------------------------------------------
# Execution: Retrieval & Local Ollama Inference
# ---------------------------------------------------------
if run_audit_clicked and st.session_state.get("chunks"):
    pages = st.session_state["pages"]
    chunks = st.session_state["chunks"]
    start_time = time.time()

    with st.status(f"Auditing contract with {model_choice}...", expanded=True) as status_box:
        # Step 1: Retrieval
        status_box.write(f"🔍 Finding clauses relevant to: **{audit_focus}**...")
        if retrieval_mode.startswith("⚡"):
            relevant_chunks = FastClauseRetriever.rank_chunks(chunks, audit_focus, top_k=5)
        else:
            if st.session_state.get("faiss_index") is None or st.session_state.get("embedder") is None:
                status_box.write("🧠 Computing neural embeddings (cached for this document)...")
                embedder, faiss_index = get_dense_embeddings(chunks)
                st.session_state["embedder"] = embedder
                st.session_state["faiss_index"] = faiss_index
            else:
                embedder = st.session_state["embedder"]
                faiss_index = st.session_state["faiss_index"]

            query_vector = embedder.encode([f"unfavorable contract risk liability: {audit_focus}"]).astype("float32")
            k = min(5, len(chunks))
            _, indices = faiss_index.search(query_vector, k)
            relevant_chunks = [chunks[i] for i in indices[0]]

        # Step 2: Context assembly
        context = "\n\n".join([f"[Page {c['page_number']}]: {c['content']}" for c in relevant_chunks])

        # Step 3: Local LLM Screening
        status_box.write(f"🤖 Generating structured risk analysis with `{model_choice}`...")
        prompt = f"""You are an expert contract-risk screening attorney for freelancers.
Analyze these contract excerpts focusing specifically on: {audit_focus}.

Identify 2 to 4 genuine unfavorable clauses, liabilities, or ambiguities.
Do NOT invent duplicate or generic placeholder findings. Each finding must cite its exact page number and quote an exact phrase from the excerpts.

Return a valid JSON object matching this structure:
{{
  "summary": "1-2 sentence executive summary of detected risks",
  "findings": [
    {{
      "category": "Liability",
      "priority": "High",
      "finding": "Contractor assumes unlimited indemnification without mutual protection.",
      "evidence_section": "Page 1 - exact excerpt: \\"indemnify and hold harmless\\"",
      "reason": "Exposes the freelancer to uncapped legal and financial exposure.",
      "action_question": "Can indemnification be mutual and capped at the contract fee?"
    }}
  ]
}}

Contract Excerpts:
{context}

Audit Report JSON:"""

        try:
            response = ollama.chat(
                model=model_choice,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": temperature,
                    "num_ctx": 2048,
                    "num_predict": 850,
                },
                format="json",
            )
        except ollama.ResponseError as err:
            if err.status_code == 404 or "not found" in str(err).lower():
                status_box.update(label="❌ Model not found", state="error")
                st.error(
                    f"🚨 **Model `{model_choice}` not found in Ollama.**\n\n"
                    f"Run this in terminal to install:\n```bash\nollama pull {model_choice}\n```\n"
                    f"Or select an installed model from the sidebar."
                )
            else:
                status_box.update(label="❌ Ollama error", state="error")
                st.error(f"Ollama Error ({err.status_code}): {err}")
            st.stop()
        except Exception as err:
            status_box.update(label="❌ Communication failed", state="error")
            st.error(f"Failed to communicate with Ollama: {err}\n\nPlease ensure Ollama is running (`ollama serve`).")
            st.stop()

        # Step 4: Parse with resilient parser
        status_box.write("📋 Parsing structured risk findings...")
        raw_output = response.get("message", {}).get("content", "")
        try:
            audit_report = parse_audit_report(raw_output)
        except Exception as error:
            status_box.update(label="❌ Parsing error", state="error")
            st.error(f"The local model returned an invalid format: {error}")
            st.code(raw_output)
            st.stop()

        # Step 5: Citation Verification
        status_box.write("✅ Verifying page citations and exact quotes against PDF text...")
        citation_results = validate_report_citations(audit_report.findings, pages)

        elapsed = time.time() - start_time
        status_box.update(label=f"🎉 Audit finished in {elapsed:.1f}s!", state="complete", expanded=False)

        st.session_state["audit_report"] = audit_report
        st.session_state["citation_results"] = citation_results
        st.session_state["relevant_chunks"] = relevant_chunks
        st.session_state["audit_elapsed_sec"] = elapsed
        st.session_state["selected_clause_idx"] = 0
        st.session_state["doc_page_idx"] = 0
        st.rerun()

# ---------------------------------------------------------
# App Header
# ---------------------------------------------------------
st.markdown('<div class="main-header-title">⚖️ VaultDocs AI — Legal Contract Analysis Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-header-subtitle">Local-first contract risk screening • 100% private retrieval & open-weight LLM inference • Zero cloud leakage</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Main UI Layout Rendering
# ---------------------------------------------------------
audit_report: Optional[AuditReport] = st.session_state.get("audit_report")
citation_results: Optional[List[CitationResult]] = st.session_state.get("citation_results")
pages: List[Dict[str, Any]] = st.session_state.get("pages", [])

if not uploaded_file:
    # Empty State Guidance
    st.info("👈 **Upload a contract PDF in the sidebar** to start your secure local audit session.")

    demo_col1, demo_col2, demo_col3 = st.columns([1, 1.5, 1])
    with demo_col1:
        st.markdown("### 📑 1. Clauses List")
        st.caption("Auto-categorized list of detected liability traps with risk scores, priority tags, and confidence bars.")
    with demo_col2:
        st.markdown("### 📄 2. Document Viewer")
        st.caption("Clean page-by-page PDF viewer with excerpt navigation and precise text tracking.")
    with demo_col3:
        st.markdown("### 🔍 3. Clause Inspector")
        st.caption("Deep-dive inspection panel with verified citations, risk drivers, and suggested negotiation moves.")

elif audit_report is None:
    # File loaded, ready to run audit
    st.success(f"✅ **Contract loaded:** `{uploaded_file.name}` ({len(pages)} pages, {len(st.session_state.get('chunks', []))} chunks indexed)")
    st.markdown("Select your audit focus in the sidebar and click **'🚀 Run Risk Screening Audit'** to initiate local inference.")
    if st.button("🚀 Start Local Risk Audit Now", type="primary"):
        st.session_state["audit_report"] = None
        st.rerun()

else:
    # =========================================================
    # 1. TOP SECTION: "Risk Overview"
    # =========================================================
    st.markdown("### 📊 Risk Overview")

    overall_score, risk_status, risk_color = calculate_overall_risk(audit_report.findings)
    critical_high_count = sum(1 for f in audit_report.findings if f.priority.lower() in ["critical", "high"])
    verified_count = sum(1 for c in citation_results if c.verified) if citation_results else 0
    total_findings = len(audit_report.findings)
    ver_pct = int((verified_count / total_findings * 100)) if total_findings > 0 else 100

    # 4 Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Contract Risk Score</div>
                <div class="metric-value" style="color:{risk_color};">Risk: {overall_score}/100</div>
                <div class="metric-sub" style="color:{risk_color};">{risk_status}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Critical & High Risks</div>
                <div class="metric-value" style="color:#ef4444;">{critical_high_count}</div>
                <div class="metric-sub">out of {total_findings} flagged clauses</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Citation Accuracy</div>
                <div class="metric-value" style="color:#16a34a;">{ver_pct}%</div>
                <div class="metric-sub">{verified_count}/{total_findings} verified against PDF</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m4:
        elapsed_val = st.session_state.get("audit_elapsed_sec")
        elapsed_text = f"⚡ {elapsed_val:.1f}s" if elapsed_val else f"{len(pages)} Pgs"
        elapsed_sub = f"local CPU ({model_choice})" if elapsed_val else f"{len(st.session_state.get('chunks', []))} clauses indexed"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Local Screening Speed</div>
                <div class="metric-value" style="color:#6366f1;">{elapsed_text}</div>
                <div class="metric-sub">{elapsed_sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Horizontal Summary Container with Category Tags
    unique_categories = sorted(list(set(f.category for f in audit_report.findings)))
    cat_badges = "".join([f"<span class='badge-pill badge-cat'>🏷️ {html.escape(c)}</span>" for c in unique_categories])

    st.markdown(
        f"""
        <div style="background:rgba(128,128,128,0.05); border:1px solid rgba(128,128,128,0.18); border-radius:10px; padding:12px 18px; margin:12px 0 18px 0;">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; margin-bottom:6px;">
                <span style="font-size:11.5px; font-weight:700; text-transform:uppercase; letter-spacing:0.5px; opacity:0.8;">Top Flagged Risk Categories</span>
                <div>{cat_badges if cat_badges else "<span style='font-size:11px; opacity:0.7;'>None</span>"}</div>
            </div>
            <div style="font-size:13px; line-height:1.6; opacity:0.9;">
                <b>Executive Summary:</b> {html.escape(audit_report.summary)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar Download Option
    markdown_report = generate_markdown_report(audit_report, citation_results)
    st.sidebar.download_button(
        "⬇️ Download Markdown Audit Report",
        data=markdown_report,
        file_name="vaultdocs-audit-report.md",
        mime="text/markdown",
        use_container_width=True,
    )

    # =========================================================
    # 2. THREE-COLUMN MAIN LAYOUT: st.columns([1, 1.5, 1])
    # =========================================================
    col_left, col_center, col_right = st.columns([1, 1.5, 1])

    # Ensure selected index is valid
    selected_idx = st.session_state.get("selected_clause_idx", 0)
    if total_findings > 0:
        selected_idx = max(0, min(total_findings - 1, selected_idx))
    else:
        selected_idx = 0
    st.session_state["selected_clause_idx"] = selected_idx

    # ---------------------------------------------------------
    # LEFT COLUMN: "Clauses List"
    # ---------------------------------------------------------
    with col_left:
        st.markdown("### 📑 Clauses List")

        # Search & Filter controls
        search_kw = st.text_input(
            "Filter clauses",
            placeholder="Search category or keywords...",
            label_visibility="collapsed",
            key="clause_search_kw",
        )
        priority_filter = st.selectbox(
            "Priority",
            ["All Priorities", "Critical", "High", "Medium", "Low"],
            label_visibility="collapsed",
            key="clause_priority_filter",
        )

        filtered_indices = []
        for i, f in enumerate(audit_report.findings):
            if search_kw:
                q = search_kw.lower()
                if q not in f.finding.lower() and q not in f.category.lower() and q not in f.evidence_section.lower():
                    continue
            if priority_filter != "All Priorities":
                if priority_filter.lower() not in f.priority.lower():
                    continue
            filtered_indices.append(i)

        st.caption(f"Showing {len(filtered_indices)} of {total_findings} detected clauses")

        # Scrollable container for detected clauses
        with st.container(height=560):
            if not filtered_indices:
                st.info("No clauses match your search filters.")
            else:
                for idx in filtered_indices:
                    f = audit_report.findings[idx]
                    cit = citation_results[idx] if citation_results and idx < len(citation_results) else None
                    meta = get_priority_meta(f.priority)
                    is_active = (idx == selected_idx)

                    card_border = "2px solid #6366f1" if is_active else "1px solid rgba(128,128,128,0.18)"
                    card_bg = "rgba(99, 102, 241, 0.08)" if is_active else "rgba(128,128,128,0.04)"
                    cit_badge = (
                        "<span class='badge-pill badge-verified'>✅ Verified</span>"
                        if cit and cit.verified
                        else "<span class='badge-pill badge-unverified'>⚠️ Unverified</span>"
                    )

                    st.markdown(
                        f"""
                        <div class="clause-list-item" style="background:{card_bg}; border:{card_border};">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                                <span class="badge-pill badge-cat">{html.escape(f.category[:20])}</span>
                                <span class="badge-pill badge-{meta['label'].lower()}">{meta['label']}</span>
                            </div>
                            <div style="font-weight:600; font-size:12.5px; margin-bottom:4px; line-height:1.4;">
                                {"👉 " if is_active else ""}{html.escape(f.finding[:80])}{"..." if len(f.finding) > 80 else ""}
                            </div>
                            <div style="display:flex; justify-content:space-between; align-items:center; font-size:11px; opacity:0.85; margin-bottom:4px;">
                                <span>Risk: <b>{meta['score']}/100</b></span>
                                <span>{cit_badge}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Confidence progress bar
                    st.progress(meta["score"] / 100)

                    # Clause selection button
                    btn_label = "✅ Selected" if is_active else f"Inspect Clause #{idx + 1}"
                    if st.button(
                        btn_label,
                        key=f"btn_clause_{idx}",
                        use_container_width=True,
                        disabled=is_active,
                    ):
                        st.session_state["selected_clause_idx"] = idx
                        # Auto-jump Document Viewer to cited page if available
                        if cit and cit.page_numbers:
                            target_page = cit.page_numbers[0]
                            st.session_state["doc_page_idx"] = max(0, min(len(pages) - 1, target_page - 1))
                        st.rerun()

    # ---------------------------------------------------------
    # CENTER COLUMN: "Document Viewer"
    # ---------------------------------------------------------
    with col_center:
        st.markdown("### 📄 Document Viewer")

        if pages:
            current_page_idx = st.session_state.get("doc_page_idx", 0)
            current_page_idx = max(0, min(len(pages) - 1, current_page_idx))
            st.session_state["doc_page_idx"] = current_page_idx

            page_data = pages[current_page_idx]
            page_num = page_data["page_number"]
            page_text = page_data.get("content", "")

            # Check if active clause has an excerpt on this specific page
            active_excerpt = None
            if total_findings > 0 and citation_results and selected_idx < len(citation_results):
                cit = citation_results[selected_idx]
                if cit and cit.excerpt and page_num in cit.page_numbers:
                    active_excerpt = cit.excerpt

            # Scrollable contract document viewer box
            with st.container(height=520):
                if active_excerpt:
                    st.markdown(
                        f"""
                        <div class="active-excerpt-callout">
                            <b>📌 Flagged Excerpt on Page {page_num}:</b><br/>
                            <i>"{html.escape(active_excerpt)}"</i>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                escaped_text = html.escape(page_text)

                # Highlight excerpt within the page if matched (with case-insensitive / whitespace tolerance)
                highlighted_text = escaped_text
                if active_excerpt:
                    esc_excerpt = html.escape(active_excerpt)
                    if re.search(re.escape(esc_excerpt), escaped_text, re.IGNORECASE):
                        highlighted_text = re.sub(
                            re.escape(esc_excerpt),
                            lambda m: f"<mark style='background:#fef08a; color:#854d0e; padding:1px 4px; border-radius:3px;'><b>{m.group(0)}</b></mark>",
                            escaped_text,
                            flags=re.IGNORECASE,
                        )
                    else:
                        words = [re.escape(html.escape(w)) for w in active_excerpt.split() if len(w) > 2]
                        if len(words) >= 3:
                            fuzzy_pattern = r"\s+".join(words[:6])
                            if re.search(fuzzy_pattern, escaped_text, re.IGNORECASE):
                                highlighted_text = re.sub(
                                    fuzzy_pattern,
                                    lambda m: f"<mark style='background:#fef08a; color:#854d0e; padding:1px 4px; border-radius:3px;'><b>{m.group(0)}</b></mark>",
                                    escaped_text,
                                    flags=re.IGNORECASE,
                                )

                st.markdown(
                    f"""
                    <div class="doc-viewer-box">
                        <div class="doc-page-header">
                            <span>PAGE {page_num} OF {len(pages)}</span>
                            <span>{len(page_text.split())} WORDS</span>
                        </div>
                        {highlighted_text}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Pagination Controls below viewer
            p_col1, p_col2, p_col3 = st.columns([1, 1.2, 1])
            with p_col1:
                if st.button("⬅️ Previous", disabled=(current_page_idx <= 0), use_container_width=True):
                    st.session_state["doc_page_idx"] = max(0, current_page_idx - 1)
                    st.rerun()

            with p_col2:
                jump_selection = st.selectbox(
                    "Jump to page",
                    options=range(len(pages)),
                    format_func=lambda i: f"Page {pages[i]['page_number']} of {len(pages)}",
                    index=current_page_idx,
                    key="doc_page_jump_select",
                    label_visibility="collapsed",
                )
                if jump_selection != current_page_idx:
                    st.session_state["doc_page_idx"] = jump_selection
                    st.rerun()

            with p_col3:
                if st.button("Next ➡️", disabled=(current_page_idx >= len(pages) - 1), use_container_width=True):
                    st.session_state["doc_page_idx"] = min(len(pages) - 1, current_page_idx + 1)
                    st.rerun()
        else:
            st.info("No document pages available to display.")

    # ---------------------------------------------------------
    # RIGHT COLUMN: "Clause Inspector"
    # ---------------------------------------------------------
    with col_right:
        st.markdown("### 🔍 Clause Inspector")

        if total_findings > 0:
            finding = audit_report.findings[selected_idx]
            cit = citation_results[selected_idx] if citation_results and selected_idx < len(citation_results) else None
            meta = get_priority_meta(finding.priority)

            with st.container(height=560):
                # Header & Category Badges
                cit_badge_html = (
                    f"<span class='badge-pill badge-verified'>✅ Verified ({cit.message if cit else ''})</span>"
                    if cit and cit.verified
                    else f"<span class='badge-pill badge-unverified'>⚠️ Unverified ({cit.message if cit else 'Not verified'})</span>"
                )

                st.markdown(
                    f"""
                    <div style="margin-bottom:8px;">
                        <span class="badge-pill badge-cat">{html.escape(finding.category)}</span>
                        <span class="badge-pill badge-{meta['label'].lower()}">{meta['label']} PRIORITY</span>
                    </div>
                    <div style="margin-bottom:8px;">{cit_badge_html}</div>
                    <div style="font-size:15px; font-weight:700; line-height:1.4; margin-bottom:8px;">
                        {html.escape(finding.finding)}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Quoted Clause Text / Evidence
                pages_cited = ", ".join(map(str, cit.page_numbers)) if cit and cit.page_numbers else "Unspecified"
                raw_excerpt = cit.excerpt if cit and cit.excerpt else finding.evidence_section

                st.markdown(
                    f"""
                    <div class="inspector-label">📜 Contract Excerpt (Cited Page {pages_cited})</div>
                    <div class="inspector-quote-box">
                        "{html.escape(raw_excerpt)}"
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Why Flagged
                st.markdown(
                    f"""
                    <div class="inspector-label">⚠️ Why Flagged (Triage Indicator)</div>
                    <div style="font-size:12.5px; line-height:1.55; margin-bottom:10px;">
                        {html.escape(finding.finding)}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Risk Drivers
                st.markdown(
                    f"""
                    <div class="inspector-label">📊 Risk Drivers & Commercial Impact</div>
                    <div style="font-size:12.5px; line-height:1.55; background:rgba(128,128,128,0.05); padding:10px 12px; border-radius:6px; margin-bottom:10px;">
                        {html.escape(finding.reason)}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Actionable Suggested Moves
                st.markdown(
                    f"""
                    <div class="inspector-label">💡 Suggested Moves & Actionable Question</div>
                    <div style="font-size:12.5px; line-height:1.55; border-left:3px solid #22c55e; padding:6px 10px; margin-bottom:8px;">
                        <b>Question for Counterparty:</b><br/>
                        <i>"{html.escape(finding.action_question)}"</i>
                    </div>
                    <div style="font-size:11.5px; opacity:0.85; margin-top:6px; line-height:1.45;">
                        <b>Human Review Recommendation:</b> Request a balanced clause with reciprocal rights, reasonable notice periods, and capped liability equal to the contract fees.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No clause findings detected for this focus area.")

    # ---------------------------------------------------------
    # Optional Bottom Trail: Retrieved Context Chunks
    # ---------------------------------------------------------
    relevant_chunks = st.session_state.get("relevant_chunks", [])
    if relevant_chunks:
        with st.expander("🔍 Audit Trail (Retrieved Context Chunks)"):
            for c in relevant_chunks:
                st.caption(f"**Page {c['page_number']} Chunk**")
                st.text(c["content"])