# Mini-Index: Pipeline Verification Demo

Proves the semantic indexing pipeline works end-to-end in under 60 seconds.

## Quick Start

```bash
pip install sentence-transformers faiss-cpu
python demo_query.py
```

## What This Contains

| File | Description |
|------|-------------|
| `vectors.index` | FAISS IndexFlatIP (1024-dim, 22 vectors) |
| `chunks.jsonl` | Text chunks (one JSON per line) |
| `metadata.jsonl` | Per-chunk metadata (aligned line-by-line) |
| `summary.json` | Manifest with counts, model info, quality metrics |
| `demo_query.py` | Runnable demo script |
| `sample_docs/` | 20 source documents covering ML, IR, NLP topics |

## What It Proves

1. **Code runs** — not just documentation
2. **FAISS loads** — proper index format, correct dimensions
3. **Embeddings work** — e5-large-v2 produces query vectors
4. **Search returns results** — queries return semantically relevant chunks
5. **Integrity holds** — vector count matches chunk count matches manifest

## Technical Details

- **Embedding model:** intfloat/e5-large-v2 (1024 dimensions)
- **Index type:** FAISS IndexFlatIP (exact inner product search)
- **Documents:** 20 covering neural networks, information retrieval, FAISS, RAG, transformers, embeddings, chunking, evaluation, GPU computing, reproducibility
- **Chunks:** 22 (word-aligned, ~1500 char target, 200 char overlap)
- **Same pipeline as production** — only difference is dataset size
