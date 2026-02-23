"""
PolicyForge AI — RAG Pipeline
Optimised for Hugging Face Spaces free tier:
  - Embeddings: all-MiniLM-L6-v2  (90 MB, CPU, loads locally)
  - Vector store: ChromaDB         (in-memory, no persistence needed)
  - BM25: rank_bm25                (pure Python, no GPU)
  - LLM: Mistral-7B-Instruct       (via HF Inference API — no local weights)
"""

from __future__ import annotations

import os
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import Field, ConfigDict
from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL     = "mistralai/Mistral-7B-Instruct-v0.3"
CHUNK_SIZE    = 900
CHUNK_OVERLAP = 180
RRF_A         = 0.4
TOP_K         = 8

SYSTEM_PROMPT = """\
You are PolicyForge AI, an expert enterprise cybersecurity policy advisor.
You have deep expertise in NIST SP 800-53 Rev 5 and ISO/IEC 27001:2022.

Using ONLY the retrieved context below, generate clear, structured, actionable \
cybersecurity policy recommendations. Always:
- Reference specific control IDs (e.g. NIST AC-2, ISO A.9.2.1)
- Number each policy recommendation
- Keep language professional and implementable
- Tailor recommendations to the organization type mentioned

Context:
{context}
"""

REASONING_TEMPLATE = """\
{system}

User request: {question}

Step-by-step reasoning:
1. Identify the relevant framework controls from the context.
2. Connect those controls to the organization's specific needs.
3. Derive actionable policy statements.

Final Policy Recommendations:
"""


# ──────────────────────────────────────────────────
# RRF Hybrid Retriever
# ──────────────────────────────────────────────────

def reciprocal_rank_fusion(
    result_lists: List[List[Document]], a: float = RRF_A
) -> List[Document]:
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}
    for lst in result_lists:
        for doc in lst:
            k = doc.page_content
            doc_map.setdefault(k, doc)
    for lst in result_lists:
        for rank, doc in enumerate(lst):
            k = doc.page_content
            scores[k] = scores.get(k, 0.0) + 1.0 / (a + rank + 1)
    return sorted(doc_map.values(), key=lambda d: scores.get(d.page_content, 0), reverse=True)


class RRFHybridRetriever(BaseRetriever):
    retrievers: List[BaseRetriever] = Field(...)
    rrf_a: float = Field(default=RRF_A)
    k_merge: int = Field(default=TOP_K)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        results = []
        for r in self.retrievers:
            cfg = {"callbacks": run_manager.get_child() if run_manager else None}
            try:
                results.append(r.invoke(query, config=cfg))
            except Exception as e:
                logger.warning(f"Retriever {r} failed: {e}")
                results.append([])
        return reciprocal_rank_fusion(results, a=self.rrf_a)[: self.k_merge]


# ──────────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────────

class PolicyForgePipeline:
    def __init__(self):
        self.hf_token = os.environ.get("HF_TOKEN", "")
        logger.info("Loading embedding model…")
        self.embed = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
        )
        self.db: Optional[Chroma] = None
        self.bm25: Optional[BM25Retriever] = None
        self.retriever: Optional[RRFHybridRetriever] = None
        self.all_splits: List[Document] = []
        self.loaded_sources: List[str] = []
        self._client: Optional[InferenceClient] = None
        logger.info("Embedding model loaded.")

    # ── LLM Client ─────────────────────────────────

    @property
    def client(self) -> InferenceClient:
        if self._client is None:
            self._client = InferenceClient(
                model=LLM_MODEL,
                token=self.hf_token if self.hf_token else None,
            )
        return self._client

    # ── Ingestion ──────────────────────────────────

    def ingest_text(self, text: str, source_name: str) -> int:
        """Ingest raw text (used for pre-loaded framework data)."""
        doc = Document(page_content=text, metadata={"source": source_name})
        splits = self.splitter.split_documents([doc])
        self._add_splits(splits, source_name)
        return len(splits)

    def ingest_file(self, file_path: str) -> int:
        """Ingest a PDF or .txt file."""
        path = Path(file_path)
        if path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(path))
        else:
            loader = TextLoader(str(path), encoding="utf-8")
        docs = loader.load()
        splits = self.splitter.split_documents(docs)
        self._add_splits(splits, path.stem)
        return len(splits)

    def _add_splits(self, splits: List[Document], source_name: str):
        self.all_splits.extend(splits)
        if source_name not in self.loaded_sources:
            self.loaded_sources.append(source_name)

        # Rebuild ChromaDB
        self.db = Chroma.from_documents(
            self.all_splits, self.embed,
            collection_metadata={"hnsw:space": "cosine"},
        )

        # Rebuild BM25
        texts = [d.page_content for d in self.all_splits]
        self.bm25 = BM25Retriever.from_texts(texts)
        self.bm25.k = TOP_K

        # Build hybrid retriever
        chroma_ret = self.db.as_retriever(search_kwargs={"k": TOP_K})
        self.retriever = RRFHybridRetriever(
            retrievers=[self.bm25, chroma_ret],
            rrf_a=RRF_A, k_merge=TOP_K,
        )
        logger.info(f"Indexed {len(splits)} chunks from '{source_name}'. Total: {len(self.all_splits)}")

    # ── Query ──────────────────────────────────────

    def query(self, question: str, framework: str = "auto") -> Dict[str, Any]:
        if self.retriever is None:
            raise RuntimeError("No documents indexed yet.")

        t0 = time.time()

        # Framework hint
        fw_hint = {
            "nist": "Focus on NIST SP 800-53 Rev 5. Reference specific NIST control IDs.",
            "iso":  "Focus on ISO/IEC 27001:2022. Reference specific ISO Annex A control IDs.",
            "both": "Reference both NIST SP 800-53 and ISO 27001:2022 control IDs.",
            "auto": "",
        }.get(framework.lower(), "")

        augmented_q = f"{fw_hint} {question}".strip()

        # Retrieve
        docs = self.retriever.invoke(augmented_q)
        context = "\n\n---\n\n".join(
            f"[Source: {d.metadata.get('source','?')} | p.{d.metadata.get('page','?')}]\n{d.page_content}"
            for d in docs
        )

        # Build prompt
        prompt = REASONING_TEMPLATE.format(
            system=SYSTEM_PROMPT.format(context=context),
            question=augmented_q,
        )

        # Generate via HF Inference API
        answer = self._generate(prompt)

        # Build source objects
        sources = self._build_sources(docs)
        retrieval_ms = round((time.time() - t0) * 1000)

        return {
            "answer": answer,
            "sources": sources,
            "framework": framework,
            "retrieval_ms": retrieval_ms,
            "chunks_retrieved": len(docs),
        }

    def _generate(self, prompt: str) -> str:
        try:
            response = self.client.text_generation(
                prompt,
                max_new_tokens=700,
                temperature=0.25,
                repetition_penalty=1.1,
                stop_sequences=["</s>", "[INST]", "User request:"],
            )
            # Strip any echoed prompt
            if "Final Policy Recommendations:" in response:
                response = response.split("Final Policy Recommendations:")[-1].strip()
            return response.strip()
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"⚠ Generation failed: {str(e)}\n\nTip: Make sure HF_TOKEN is set as a Space secret."

    def _build_sources(self, docs: List[Document]) -> List[Dict[str, Any]]:
        sources = []
        seen = set()
        for i, doc in enumerate(docs[:6]):
            meta = doc.metadata or {}
            src  = meta.get("source", "Framework Document")
            page = meta.get("page", None)
            key  = f"{src}-{page}-{doc.page_content[:40]}"
            if key in seen:
                continue
            seen.add(key)
            # Confidence based on rank (higher rank = higher confidence)
            confidence = max(60, 97 - i * 5)
            is_nist = "nist" in src.lower() or "800" in src.lower()
            is_iso  = "iso" in src.lower() or "27001" in src.lower()
            fw_type = "NIST 800-53" if is_nist else ("ISO 27001" if is_iso else "Document")
            sources.append({
                "label":      src,
                "page":       page,
                "framework":  fw_type,
                "confidence": confidence,
                "snippet":    doc.page_content[:220],
            })
        return sources

    def status(self) -> Dict[str, Any]:
        return {
            "ready":           self.retriever is not None,
            "total_chunks":    len(self.all_splits),
            "loaded_sources":  self.loaded_sources,
            "embed_model":     EMBED_MODEL,
            "llm_model":       LLM_MODEL,
        }
