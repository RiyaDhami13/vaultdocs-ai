# ⚖️ VaultDocs AI 🛡️

A local-first, privacy-focused contract risk screening assistant designed for freelancers and small agencies. VaultDocs AI analyzes client agreements, SOWs, and NDAs locally using open-weight models (`gemma:2b` or `qwen2.5:3b` via Ollama), highlighting unfavorable clauses with exact page citations and zero cloud data leakage.

## Why Open Innovation Matters

- **Data Privacy:** Client contracts stay entirely on your laptop. No third-party API tracking or server logs.
- **Model Control:** Switch seamlessly between open-weight models without modifying application logic.
- **Cost Efficiency:** \$0 per-query runtime cost.

## Setup & Running

1. Install [Ollama](https://ollama.com/) and download your preferred model:
   ```bash
   ollama pull llama3.2:1b   # Recommended for CPU: fastest & lightweight
   # or
   ollama pull gemma:2b      # Google Gemma 2B
   # or
   ollama pull qwen2.5:3b    # In-depth analysis
   ```
2. Install Python dependencies and start Streamlit:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```

## Audit output

The local model is required to return a Pydantic-validated `AuditReport`. Each
finding includes a page citation and quoted excerpt. VaultDocs AI checks those
citations against the extracted PDF text and marks missing or mismatched
citations as unverified. A verified or unverified Markdown report can be
downloaded from the audit results screen.
