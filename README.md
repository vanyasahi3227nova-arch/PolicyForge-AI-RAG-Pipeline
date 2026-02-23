---
title: PolicyForge AI
emoji: 🛡️
colorFrom: blue
colorTo: cyan
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: true
license: mit
short_description: RAG-powered cybersecurity policy generation (NIST 800-53 / ISO 27001)
---

# 🛡️ PolicyForge AI — Cybersecurity Knowledge Engine

> **Live Demo:** [https://huggingface.co/spaces/YOUR_USERNAME/policyforge-ai](https://huggingface.co/spaces/vanyasahi3227nova-arch/policyforge-ai)

A production-grade **Retrieval-Augmented Generation (RAG)** system for generating enterprise cybersecurity policies aligned with **NIST SP 800-53 Rev 5** and **ISO/IEC 27001:2022**.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Gradio UI  (app.py)                                    │
│       │                                                 │
│       ▼                                                 │
│  RAG Pipeline  (rag_pipeline.py)                        │
│       │                                                 │
│       ├── 1. Query → BM25 Retriever (lexical)           │
│       ├── 2. Query → ChromaDB (semantic embeddings)     │
│       ├── 3. Reciprocal Rank Fusion (RRF) merge         │
│       └── 4. Top-k context → HF Inference API (LLM)    │
│                    │                                    │
│                    └── mistralai/Mistral-7B-Instruct    │
│                        (free, serverless, no download)  │
└─────────────────────────────────────────────────────────┘
```

**Key design decisions:**
- **Hybrid retrieval**: BM25 (keyword) + ChromaDB (semantic) fused with RRF for best coverage
- **No local model loading**: Uses HF Inference API for generation → fits free tier RAM
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` loaded locally (only 90 MB)
- **Frameworks**: Pre-loaded with NIST 800-53 and ISO 27001 control summaries at startup

---

## 📁 Repository Structure

```
policyforge-ai/
├── app.py                  # Gradio UI + app entry point
├── rag_pipeline.py         # Core RAG logic (BM25 + ChromaDB + RRF + LLM)
├── ingest.py               # Document ingestion utilities
├── requirements.txt        # Python dependencies
├── data/
│   ├── nist_800_53.txt     # NIST 800-53 control summaries (pre-loaded)
│   └── iso_27001.txt       # ISO 27001 control summaries (pre-loaded)
├── docs/
│   └── architecture.md     # Detailed architecture notes
└── README.md               # This file
```

---

## 🚀 Running Locally

```bash
# 1. Clone
git clone https://github.com/vanyasahi3227nova-arch/policyforge-ai
cd policyforge-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set HuggingFace token (free at huggingface.co/settings/tokens)
export HF_TOKEN="hf_your_token_here"

# 4. Run
python app.py
# → opens at http://localhost:7860
```

---

## 🛠️ Deploying to Hugging Face Spaces (Free)

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
   - SDK: **Gradio**
   - Visibility: **Public**
2. Add your HF token as a Space secret: **Settings → Variables and Secrets → `HF_TOKEN`**
3. Push this repo:
```bash
git remote add space https://huggingface.co/spaces/vanyasahi3227nova-arch/policyforge-ai
git push space main
```
4. Your live URL: `https://huggingface.co/spaces/vanyasahi3227nova-arch/policyforge-ai`

---

## 🔬 RAG Implementation Details

### Hybrid Retrieval
Combines two complementary strategies via **Reciprocal Rank Fusion (RRF)**:

| Retriever | Type | Strength |
|-----------|------|----------|
| BM25 | Lexical / keyword | Exact control IDs (e.g. "AC-2", "A.9.2") |
| ChromaDB | Semantic / vector | Conceptual similarity ("access management") |

RRF formula: `score = Σ 1 / (a + rank + 1)` with `a = 0.4`

### LLM: Mistral-7B-Instruct via HF Inference API
- Free serverless endpoint, no GPU needed on the Space
- Instruction-tuned for structured policy generation
- Reasoning prompt guides step-by-step output

### Embeddings: all-MiniLM-L6-v2
- 90 MB model, CPU-only, loads in ~5 seconds
- 384-dimensional dense embeddings
- Strong performance on domain-specific text

---

## 📊 Example Queries

- *"Generate 10 device management policies for a finance company aligned with NIST 800-53"*
- *"What ISO 27001 controls apply to access management in healthcare?"*
- *"Create an incident response policy for a SaaS startup"*
- *"Compare NIST and ISO 27001 requirements for encryption"*

---

## 🤝 Contributing

PRs welcome! Areas for improvement:
- Add more framework documents (SOC 2, PCI-DSS, HIPAA)
- Improve chunking strategy for longer documents
- Add policy export to PDF/DOCX
- Fine-tune retrieval parameters



---

*Built with LangChain · ChromaDB · sentence-transformers · Gradio · Mistral-7B | Dense Embeddings | BM25 | Reciprocal Rank Function"
