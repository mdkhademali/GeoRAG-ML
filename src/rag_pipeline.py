"""Retrieval-Augmented Generation (RAG) pipeline for geospatial documentation.

Pipeline: ingest (PDF/TXT) -> sentence-aware chunking -> embeddings -> FAISS inner-product index
          -> top-k retrieval -> evidence-sufficiency check -> grounded answer with source citations.

Honesty rules implemented here
------------------------------
* Embeddings: Sentence-Transformers when the model can be loaded; otherwise a TF-IDF + LSA fallback. The active
  backend is always reported (``rag.backend_name``) so results are never attributed to the wrong model.
* Answers: extractive by default (sentences copied from retrieved evidence, each tagged with its source).
  A generative LLM is used ONLY if ``ANTHROPIC_API_KEY`` is set in the environment; otherwise no LLM text is produced.
* If retrieved evidence is weak (low similarity or no keyword overlap), the system says so instead of answering.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import faiss
import numpy as np

log = logging.getLogger(__name__)

_STOPWORDS = set("""a an the of in on at to for from by with and or but not is are was were be been being this that these
those it its as into than then so such can could should would may might will shall do does did how what which who whom why
when where whose about between over under using use used via per if else also more most less very""".split())


# --------------------------------------------------------------------------------------
# Ingestion and chunking
# --------------------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id: str
    source: str
    text: str
    page: Optional[int] = None


def read_document(path: Path) -> List[dict]:
    """Return a list of {source, page, text} records for a .txt or .pdf file."""
    path = Path(path)
    if path.suffix.lower() == ".txt":
        return [{"source": path.name, "page": None, "text": path.read_text(encoding="utf-8", errors="ignore")}]
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        recs = []
        for i, page in enumerate(PdfReader(str(path)).pages, start=1):
            try:
                txt = page.extract_text() or ""
            except Exception as exc:  # corrupted page; keep going
                log.warning("Could not extract %s p.%d: %s", path.name, i, exc)
                txt = ""
            if txt.strip():
                recs.append({"source": path.name, "page": i, "text": txt})
        return recs
    raise ValueError(f"Unsupported file type: {path.suffix}")


def ingest(documents_dir: str | Path, extra_files: Iterable[str | Path] = ()) -> List[dict]:
    """Read every PDF/TXT in a directory (non-recursive, README excluded) plus optional extra files."""
    files = sorted(p for p in Path(documents_dir).glob("*") if p.suffix.lower() in (".txt", ".pdf"))
    files += [Path(p) for p in extra_files if Path(p).exists()]
    records: List[dict] = []
    for f in files:
        try:
            records += read_document(f)
        except Exception as exc:
            log.warning("Skipping %s: %s", f, exc)
    if not records:
        raise FileNotFoundError(f"No readable PDF/TXT documents found in {documents_dir}")
    return records


def _split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])", text) if s.strip()]


def chunk_records(records: List[dict], chunk_words: int = 110, overlap_words: int = 25) -> List[Chunk]:
    """Sentence-aware sliding-window chunking with word-level overlap."""
    chunks: List[Chunk] = []
    for rec in records:
        sents = _split_sentences(rec["text"])
        cur: List[str] = []
        n = 0
        counter = 0
        def flush():
            nonlocal cur, n, counter
            if cur:
                cid = f"{rec['source']}#{counter}"
                chunks.append(Chunk(cid, rec["source"], " ".join(cur), rec["page"]))
                counter += 1
                # keep trailing sentences as overlap
                keep, kw = [], 0
                for s in reversed(cur):
                    if kw >= overlap_words:
                        break
                    keep.insert(0, s)
                    kw += len(s.split())
                cur, n = keep, kw
        for s in sents:
            w = len(s.split())
            if n + w > chunk_words and cur:
                flush()
            cur.append(s)
            n += w
        if cur and (not chunks or chunks[-1].text != " ".join(cur)):
            flush()
    return chunks


# --------------------------------------------------------------------------------------
# Embedding backends
# --------------------------------------------------------------------------------------

class DenseBackend:
    """Sentence-Transformers embeddings (e.g. all-MiniLM-L6-v2)."""
    kind = "dense"

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.name = f"sentence-transformers ({model_name})"

    def fit(self, texts: Sequence[str]) -> "DenseBackend":
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self.model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False),
                          dtype="float32")


class LSABackend:
    """Fallback: TF-IDF followed by truncated SVD (latent semantic analysis), L2-normalised."""
    kind = "lsa"

    def __init__(self, n_components: int = 128, seed: int = 42):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import Normalizer

        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english", min_df=1)
        self.n_components, self.seed = n_components, seed
        self.svd, self.norm = None, Normalizer(copy=False)
        self._TruncatedSVD = TruncatedSVD
        self.name = "TF-IDF + LSA fallback (sentence-transformers unavailable)"

    def fit(self, texts: Sequence[str]) -> "LSABackend":
        X = self.tfidf.fit_transform(texts)
        k = max(2, min(self.n_components, X.shape[0] - 1, X.shape[1] - 1))
        self.svd = self._TruncatedSVD(n_components=k, random_state=self.seed).fit(X)
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        Z = self.svd.transform(self.tfidf.transform(texts))
        return np.asarray(self.norm.transform(Z), dtype="float32")


def make_backend(model_name: str, prefer: str = "auto", seed: int = 42):
    """Return (backend, note). ``prefer``: 'auto' | 'dense' | 'lsa'."""
    if prefer in ("auto", "dense"):
        try:
            return DenseBackend(model_name), "dense backend loaded"
        except Exception as exc:
            if prefer == "dense":
                raise
            note = f"Dense backend unavailable ({type(exc).__name__}); using TF-IDF+LSA fallback."
            log.warning(note)
            return LSABackend(seed=seed), note
    return LSABackend(seed=seed), "LSA backend requested"


# --------------------------------------------------------------------------------------
# Retrieval + grounded answering
# --------------------------------------------------------------------------------------

_SYNONYMS = {"calculate": "comput", "calculated": "comput", "calculation": "comput", "calculating": "comput",
             "computed": "comput", "compute": "comput", "computation": "comput", "formula": "comput"}


def _stem(tok: str) -> str:
    if tok in _SYNONYMS:
        return _SYNONYMS[tok]
    for suf in ("ations", "ation", "ings", "ing", "ies", "es", "s"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            return tok[: -len(suf)]
    return tok


def keywords(text: str) -> List[str]:
    toks = re.findall(r"[A-Za-z][A-Za-z0-9\-]+", text.lower())
    return [_stem(t) for t in toks if t not in _STOPWORDS and len(t) > 2]


@dataclass
class Hit:
    rank: int
    score: float
    chunk: Chunk


class GeoRAG:
    """Document index with evidence-aware question answering."""

    def __init__(self, backend, chunks: List[Chunk], min_score: float, min_keyword_overlap: float = 0.34):
        self.backend, self.chunks = backend, chunks
        self.min_score, self.min_overlap = float(min_score), float(min_keyword_overlap)
        texts = [c.text for c in chunks]
        self.backend.fit(texts)
        emb = self.backend.encode(texts)
        self.index = faiss.IndexFlatIP(emb.shape[1])       # inner product == cosine for unit vectors
        self.index.add(emb)
        self.embeddings = emb

    @property
    def backend_name(self) -> str:
        return self.backend.name

    def retrieve(self, query: str, k: int = 4) -> List[Hit]:
        q = self.backend.encode([query])
        scores, idx = self.index.search(q, min(k, len(self.chunks)))
        return [Hit(r + 1, float(s), self.chunks[i]) for r, (s, i) in enumerate(zip(scores[0], idx[0])) if i >= 0]

    def _overlap(self, query: str, text: str) -> float:
        qk = set(keywords(query))
        if not qk:
            return 0.0
        tk = set(keywords(text))
        return len(qk & tk) / len(qk)

    def assess(self, query: str, hits: List[Hit]) -> dict:
        """Decide whether the retrieved evidence is sufficient to answer."""
        if not hits:
            return {"sufficient": False, "reason": "no chunks retrieved", "best_score": 0.0, "overlap": 0.0}
        best = hits[0]
        overlap = max(self._overlap(query, h.chunk.text) for h in hits)
        ok = best.score >= self.min_score and overlap >= self.min_overlap
        reason = "ok" if ok else (f"top similarity {best.score:.2f} < {self.min_score:.2f}"
                                  if best.score < self.min_score else
                                  f"keyword overlap {overlap:.2f} < {self.min_overlap:.2f}")
        return {"sufficient": ok, "reason": reason, "best_score": best.score, "overlap": overlap}

    def _best_sentences(self, query: str, hits: List[Hit], n: int = 3) -> List[dict]:
        cand = []
        for h in hits:
            if h.score < self.min_score * 0.8:
                continue
            for s in _split_sentences(h.chunk.text):
                if len(s.split()) < 5:
                    continue
                cand.append({"sentence": s, "source": h.chunk.source, "chunk_id": h.chunk.chunk_id,
                             "page": h.chunk.page, "chunk_score": h.score})
        if not cand:
            return []
        q = self.backend.encode([query])[0]
        S = self.backend.encode([c["sentence"] for c in cand])
        sim = S @ q
        for c, s in zip(cand, sim):
            c["sent_score"] = float(s) + 0.35 * self._overlap(query, c["sentence"])
        cand.sort(key=lambda c: c["sent_score"], reverse=True)
        seen, uniq = set(), []                       # overlapping chunks repeat sentences; keep one copy
        for c in cand:
            key = re.sub(r"\W+", " ", c["sentence"].lower()).strip()
            if key not in seen:
                seen.add(key)
                uniq.append(c)
        return uniq[:n]

    def answer(self, query: str, k: int = 4, llm_model: Optional[str] = None) -> dict:
        """Answer a question using only retrieved evidence. Returns a dict with status, text, evidence, sources."""
        hits = self.retrieve(query, k)
        verdict = self.assess(query, hits)
        out = {"question": query, "backend": self.backend_name, "assessment": verdict,
               "retrieved": [{"rank": h.rank, "score": round(h.score, 3), "chunk_id": h.chunk.chunk_id,
                              "source": h.chunk.source} for h in hits]}
        if not verdict["sufficient"]:
            out.update(status="insufficient_evidence", generated_by=None, sources=[],
                       answer=("Insufficient evidence in the indexed documents to answer this question "
                               f"({verdict['reason']}). No answer was generated."))
            return out
        evidence = self._best_sentences(query, hits)
        out["evidence"] = evidence
        out["sources"] = sorted({e["source"] for e in evidence})
        generated = self._maybe_generate(query, hits, llm_model)
        if generated:
            out.update(status="answered_llm", generated_by=llm_model, answer=generated)
        else:
            body = " ".join(f"{e['sentence']} [{e['source']}]" for e in evidence)
            out.update(status="answered_extractive", generated_by=None,
                       answer="Evidence-based extract (no LLM used): " + body)
        return out

    def _maybe_generate(self, query: str, hits: List[Hit], llm_model: Optional[str]) -> Optional[str]:
        """Optional LLM step. Runs only when an API key is present in the environment; never hard-coded."""
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key or not llm_model:
            return None
        try:
            import anthropic

            ctx = "\n\n".join(f"[{h.chunk.source}] {h.chunk.text}" for h in hits)
            msg = anthropic.Anthropic(api_key=key).messages.create(
                model=llm_model, max_tokens=500,
                system=("Answer ONLY from the provided context. Cite sources in square brackets using the given "
                        "document names. If the context does not contain the answer, reply exactly: "
                        "'Insufficient evidence in the indexed documents.'"),
                messages=[{"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {query}"}])
            return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
        except Exception as exc:
            log.warning("LLM generation failed (%s); falling back to extractive answer.", exc)
            return None


def build_rag(cfg: dict, root: str | Path = ".", prefer_backend: str = "auto", seed: int = 42) -> GeoRAG:
    """Convenience constructor using the ``rag`` section of config.yaml."""
    rc, root = cfg["rag"], Path(root)
    records = ingest(root / rc["documents_dir"], [root / p for p in rc.get("extra_documents", [])])
    chunks = chunk_records(records, rc["chunk_words"], rc["chunk_overlap_words"])
    backend, note = make_backend(rc["embedding_model"], prefer_backend, seed)
    min_score = rc["min_score_dense"] if backend.kind == "dense" else rc["min_score_lsa"]
    rag = GeoRAG(backend, chunks, min_score, rc["min_keyword_overlap"])
    log.info("RAG index: %d chunks from %d records | %s", len(chunks), len(records), rag.backend_name)
    return rag


def evaluate_retrieval(rag: GeoRAG, in_scope: List[dict], out_of_scope: List[str], k: int = 4) -> dict:
    """Retrieval quality on a hand-written question set. ``in_scope`` items: {'q':..., 'source': 'file.txt'}."""
    hits_at_k, rr, answered = [], [], []
    for item in in_scope:
        hs = rag.retrieve(item["q"], k)
        ranks = [h.rank for h in hs if h.chunk.source == item["source"]]
        hits_at_k.append(bool(ranks))
        rr.append(1.0 / ranks[0] if ranks else 0.0)
        answered.append(rag.answer(item["q"], k)["status"] != "insufficient_evidence")
    refused = [rag.answer(q, k)["status"] == "insufficient_evidence" for q in out_of_scope]
    return {"n_in_scope": len(in_scope), f"hit_at_{k}": float(np.mean(hits_at_k)), "mrr": float(np.mean(rr)),
            "in_scope_answered_rate": float(np.mean(answered)), "n_out_of_scope": len(out_of_scope),
            "out_of_scope_refusal_rate": float(np.mean(refused)) if refused else float("nan")}


def export_chunks(chunks: List[Chunk], path: str | Path) -> None:
    Path(path).write_text(json.dumps([asdict(c) for c in chunks], indent=2), encoding="utf-8")
