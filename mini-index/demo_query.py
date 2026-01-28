#!/usr/bin/env python3
"""
Query the mini-index in under 60 seconds — proves pipeline works.

Usage: python demo_query.py
Requires: pip install sentence-transformers faiss-cpu
"""

import os
import sys
import faiss
import json
from sentence_transformers import SentenceTransformer

def main():
    # Ensure we run from the mini-index directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print("\n" + "=" * 60)
    print("Mini-Index Demo: Semantic Search on 20 Documents")
    print("=" * 60 + "\n")

    # Load index
    index = faiss.read_index("vectors.index")
    with open("chunks.jsonl") as f:
        chunks = [json.loads(line) for line in f]

    with open("summary.json") as f:
        summary = json.load(f)

    print(f"Loaded index: {index.ntotal} vectors, {index.d} dimensions")
    print(f"Model: {summary['embedding_model']}")
    print(f"Index type: {summary['index_type']}")
    print(f"Status: {summary['status']}")

    # Verify integrity
    assert index.ntotal == len(chunks), \
        f"Integrity check failed: {index.ntotal} vectors != {len(chunks)} chunks"
    assert index.ntotal == summary["vector_count"], \
        f"Manifest mismatch: {index.ntotal} vectors != {summary['vector_count']} in summary"
    print(f"Integrity check passed: {index.ntotal} vectors == {len(chunks)} chunks")

    # Load model
    print(f"\nLoading embedding model...")
    model = SentenceTransformer(summary["embedding_model"])

    # Test queries
    test_queries = [
        "machine learning and neural networks",
        "semantic search and vector retrieval",
        "how to build a FAISS index",
    ]

    print("\n" + "-" * 60)
    for q in test_queries:
        # Embed with query prefix (per E5 spec)
        qvec = model.encode(f"query: {q}", normalize_embeddings=True)

        # Search
        D, I = index.search(qvec.reshape(1, -1), k=3)

        print(f"\nQuery: '{q}'")
        for rank, (score, idx) in enumerate(zip(D[0], I[0])):
            chunk = chunks[idx]
            text_preview = chunk["text"][:100].replace("\n", " ")
            print(f"  [{rank+1}] score={score:.3f}  {text_preview}...")
            print(f"      doc: {chunk['doc_id']}")

    print("\n" + "-" * 60)
    print("\nPipeline verified:")
    print("  - FAISS index loads correctly")
    print("  - Embedding model produces query vectors")
    print("  - Semantic search returns ranked results")
    print("  - Vector/chunk alignment is intact")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
