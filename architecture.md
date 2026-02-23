# PolicyForge AI — Architecture Notes

## Why This Architecture for HF Free Tier

The Hugging Face Spaces **free CPU tier** gives you:
- 2 vCPUs, 16 GB RAM
- No GPU
- Persistent storage: none (in-memory only)
- Network: public HTTPS URL, always-on

### The Constraint: No Local LLM

A 7B parameter model (like Gemma-2b-it or Mistral-7B) needs:
- ~14 GB RAM in float16 (or ~7 GB quantized)
- Several minutes to load from disk
- 1–5 seconds per generation on CPU

That's too slow and too memory-heavy for a shared demo. The solution: **HF Inference API** (serverless), which lets you call any hosted model via HTTP without loading weights locally.

### What Runs Locally (on the Space)

| Component | Model | Size | Why local |
|-----------|-------|------|-----------|
| Embeddings | all-MiniLM-L6-v2 | 90 MB | Fast, cheap, used at query time AND indexing |
| ChromaDB | n/a | in-memory | Vector similarity search, no server needed |
| BM25 | rank_bm25 | ~1 MB | Pure Python, zero overhead |
| Gradio UI | n/a | ~50 MB | The web frontend |

### What Runs via API

| Component | Model | Provider |
|-----------|-------|----------|
| Text generation | Mistral-7B-Instruct-v0.3 | HF Inference API (free with token) |

---

## Retrieval Pipeline Detail

```
User query: "Generate access control policies for a healthcare org"
     │
     ├─── BM25 Retriever ──────────────────────────────────────────────┐
     │    Tokenizes query, scores all chunks by BM25(IDF×TF)           │
     │    Returns top-10 by keyword relevance                          │
     │    Best for: exact control IDs ("AC-2", "A.9.2.1")              │
     │                                                                  ├─→ RRF Fusion
     └─── ChromaDB Retriever ─────────────────────────────────────────┘         │
          Embeds query with MiniLM-L6-v2 (384-dim vector)                       │
          Cosine similarity search over indexed chunks                           │
          Returns top-10 by semantic similarity                                  │
          Best for: conceptual queries ("access management")                     │
                                                                                 │
                                              Reciprocal Rank Fusion ◄───────────┘
                                              score = Σ 1/(0.4 + rank + 1)
                                              Deduplicates, re-ranks, returns top-8
                                                          │
                                                          ▼
                                              Build prompt with context
                                              (SYSTEM + REASONING_TEMPLATE)
                                                          │
                                                          ▼
                                              HF Inference API
                                              mistralai/Mistral-7B-Instruct-v0.3
                                              max_new_tokens=700, temp=0.25
                                                          │
                                                          ▼
                                              Policy recommendations + sources
```

---

## Chunking Strategy

```python
RecursiveCharacterTextSplitter(
    chunk_size=900,       # ~200 tokens, fits well in context window
    chunk_overlap=180,    # 20% overlap preserves cross-chunk context
    add_start_index=True  # metadata for source attribution
)
```

Each control entry in the pre-loaded data files is naturally ~100–400 words, so most controls fit in 1–2 chunks. The overlap ensures that control IDs appearing at chunk boundaries are retrievable from both chunks.

---

## Pre-loaded Knowledge Base

At startup, `app.py` calls `pipe.ingest_file()` on two files:

- `data/nist_800_53.txt` — 50+ controls from AC, AT, AU, CM, CP, IA, IR, MA, MP, RA, SC, SI families
- `data/iso_27001.txt`  — All 93 controls from ISO 27001:2022 Annex A (A.5–A.8)

Users can also upload their own PDFs (org policies, audit reports, custom standards) which get chunked and added to the same ChromaDB collection.

---

## Prompt Engineering

The reasoning prompt forces the model to:
1. Ground its answer in the retrieved context (prevents hallucination)
2. Follow a structured reasoning chain
3. Output numbered policy recommendations
4. Include specific control IDs

```
SYSTEM: You are PolicyForge AI... Use ONLY the retrieved context below...
        Context: {top-8 chunks with source metadata}

USER:   {framework_hint} {question}

ASSISTANT reasoning chain:
1. Identify relevant controls from context
2. Connect to organizational needs
3. Derive policy statements

Final Policy Recommendations:
[model output starts here]
```

---

## Scaling Beyond Free Tier

When ready to upgrade:

| Need | Solution |
|------|----------|
| Persistent knowledge base | Add ChromaDB Cloud or Pinecone |
| Faster generation | Upgrade to HF Spaces with A10G GPU |
| Better LLM | Switch to Claude via Anthropic API |
| More users | Add HF Spaces persistent storage + Redis queue |
| PDF upload persistence | Add S3/R2 bucket |
