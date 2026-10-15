import os
import tempfile
import streamlit as st
import ollama
from pydantic import ValidationError
from services.pdf_parser import PDFParser
from services.clause_splitter import ClauseSplitter
from services.citation_validator import validate_report_citations
from models.schemas import AuditReport
from reports.report_generator import generate_markdown_report

st.set_page_config(page_title="VaultDocs AI — Local Contract Auditor", page_icon="⚖️", layout="wide")

st.title("⚖️ VaultDocs AI: Local-First Contract Risk Screening Assistant")
st.markdown("*Screen freelance service agreements and NDAs locally using open-weight models. Zero cloud data leakage.*")

# Sidebar Configuration
st.sidebar.header("⚙️ Open AI Configuration")
model_choice = st.sidebar.selectbox("Select Local Model (Model Control)", ["gemma:2b", "qwen2.5:3b"])
temperature = st.sidebar.slider("Inference Temperature", 0.0, 0.4, 0.1)

st.sidebar.markdown("---")
st.sidebar.header("📁 Document Ingestion (Data Control)")
uploaded_file = st.sidebar.file_uploader("Upload Client Agreement or NDA (PDF)", type=["pdf"])

@st.cache_resource
def load_embedder(model_name: str):
    """Load the embedding model only after a document has been uploaded."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    pages = PDFParser.extract_text_with_pages(tmp_path)
    chunks = ClauseSplitter.split_into_chunks(pages)
    os.remove(tmp_path)

    # Local Embeddings & FAISS Indexing
    import faiss
    import numpy as np

    embedder = load_embedder("all-MiniLM-L6-v2")
    chunk_texts = [c["content"] for c in chunks]
    embeddings = embedder.encode(chunk_texts, show_progress_bar=False)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))

    st.sidebar.success(f"Indexed {len(chunks)} clauses locally (100% private).")

    # Audit Controls
    audit_focus = st.selectbox(
        "Select Audit Focus Area:",
        [
            "General Risk & Liability Summary",
            "Payment Terms & Late Penalties",
            "Intellectual Property (IP) Rights",
            "Termination & Notice Periods"
        ]
    )

    if st.button("Run Local Contract Audit"):
        query_vector = embedder.encode([audit_focus]).astype("float32")
        k = min(4, len(chunks))
        distances, indices = index.search(query_vector, k)
        
        relevant_chunks = [chunks[i] for i in indices[0]]
        context = "\n\n".join([f"[Page {c['page_number']}]: {c['content']}" for c in relevant_chunks])

        prompt = f"""
        You are a contract-risk screening assistant for freelancers. Analyze these contract excerpts focusing on: {audit_focus}.
        
        RULES:
        1. Flag unfavorable clauses or ambiguity.
        2. Every finding must cite its exact page number and quote an exact excerpt.
        3. Use the exact JSON shape: {{"summary": "...", "findings": [
           {{"category": "...", "priority": "Low|Medium|High|Critical",
           "finding": "...", "evidence_section": "Page N - exact excerpt: \\"...\\",
           "reason": "...", "action_question": "..."}}]}}.
        4. Do not provide formal legal advice; label findings as triage indicators.
        5. Return JSON only. Use an empty findings list if the excerpts contain no relevant risk.

        Contract Excerpts:
        {context}

        Audit Report:
        """

        with st.spinner(f"Running local inference with {model_choice}..."):
            response = ollama.chat(
                model=model_choice,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": temperature},
                format="json",
            )

        try:
            audit_report = AuditReport.model_validate_json(
                response["message"]["content"]
            )
        except (ValidationError, TypeError, KeyError) as error:
            st.error(f"The local model returned an invalid audit report: {error}")
            st.code(response.get("message", {}).get("content", ""))
            st.stop()

        citation_results = validate_report_citations(audit_report.findings, pages)

        st.markdown("### 📋 Audit Findings & Triage Report")
        st.markdown(audit_report.summary)
        for finding, citation in zip(audit_report.findings, citation_results):
            with st.container(border=True):
                st.subheader(f"{finding.category} — {finding.priority}")
                st.write(finding.finding)
                st.markdown(f"**Evidence:** {finding.evidence_section}")
                if citation.verified:
                    st.success(f"✅ Citation verified: {citation.message}")
                else:
                    st.warning(f"⚠️ Unverified citation: {citation.message}")
                st.markdown(f"**Why it matters:** {finding.reason}")
                st.markdown(f"**Question:** {finding.action_question}")

        markdown_report = generate_markdown_report(audit_report, citation_results)
        st.download_button(
            "⬇️ Download Markdown Audit Report",
            data=markdown_report,
            file_name="vaultdocs-audit-report.md",
            mime="text/markdown",
        )

        with st.expander("🔍 Audit Trail (Retrieved Clauses)"):
            for c in relevant_chunks:
                st.caption(f"**Page {c['page_number']}**")
                st.text(c["content"])
else:
    st.info("👈 Upload a contract PDF in the sidebar to start your secure local audit session.")