#!/usr/bin/env python3
"""
Merge 5 StackExchange split indexes into one final index.

Usage:
    python3 merge_stackexchange_splits.py
"""

import json
import sys
from pathlib import Path
import numpy as np
import faiss
import time

# Paths
# NOTE: Update this path for your local environment
BASE_DIR = Path("./results/indexes")
SPLIT_INDEX_DIRS = [
    BASE_DIR / f"stackexchange_split{i}"
    for i in range(1, 6)
]
FINAL_INDEX_DIR = BASE_DIR / "stackexchange_python"
EMBEDDING_DIM = 1024

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def merge_indexes():
    log("=" * 60)
    log("MERGING STACKEXCHANGE SPLIT INDEXES")
    log("=" * 60)

    all_vectors = []
    all_chunks = []
    all_metadata = []

    # Load each split
    for i, split_dir in enumerate(SPLIT_INDEX_DIRS, 1):
        log(f"Loading split {i} from {split_dir}...")

        index_path = split_dir / "vectors.index"
        chunks_path = split_dir / "chunks.json"
        metadata_path = split_dir / "metadata.jsonl"

        if not index_path.exists():
            log(f"  WARNING: {index_path} not found, skipping split {i}")
            continue

        # Load vectors
        index = faiss.read_index(str(index_path))
        vectors = faiss.rev_swig_ptr(index.get_xb(), index.ntotal * EMBEDDING_DIM)
        vectors = np.array(vectors).reshape(index.ntotal, EMBEDDING_DIM)
        all_vectors.append(vectors)
        log(f"  Loaded {index.ntotal:,} vectors")

        # Load chunks
        if chunks_path.exists():
            with open(chunks_path, 'r') as f:
                chunks = json.load(f)
                all_chunks.extend(chunks)
            log(f"  Loaded {len(chunks):,} chunks")

        # Load metadata
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                for line in f:
                    all_metadata.append(json.loads(line.strip()))
            log(f"  Loaded {len(all_metadata):,} total metadata entries")

    if not all_vectors:
        log("ERROR: No vectors found in any split!")
        sys.exit(1)

    # Concatenate all vectors
    log("")
    log("Concatenating vectors...")
    vectors = np.vstack(all_vectors)
    log(f"Total vectors: {len(vectors):,}")
    log(f"Total chunks: {len(all_chunks):,}")
    log(f"Total metadata: {len(all_metadata):,}")

    # Integrity check
    if len(vectors) != len(all_chunks) or len(vectors) != len(all_metadata):
        log("INTEGRITY CHECK: FAILED")
        log(f"  Mismatch: vectors={len(vectors)}, chunks={len(all_chunks)}, metadata={len(all_metadata)}")
        sys.exit(1)
    log("INTEGRITY CHECK: PASSED")

    # Build final FAISS index
    log("")
    log("Building final FAISS IndexFlatIP...")
    final_index = faiss.IndexFlatIP(EMBEDDING_DIM)
    final_index.add(vectors.astype(np.float32))
    log(f"FAISS index: {final_index.ntotal:,} vectors")

    # Write to final directory
    FINAL_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    log("")
    log("Writing merged index...")

    # Write vectors.index
    tmp_index = FINAL_INDEX_DIR / "vectors.index.tmp"
    faiss.write_index(final_index, str(tmp_index))
    tmp_index.rename(FINAL_INDEX_DIR / "vectors.index")
    log(f"  Wrote vectors.index")

    # Write chunks.json
    tmp_chunks = FINAL_INDEX_DIR / "chunks.json.tmp"
    with open(tmp_chunks, 'w') as f:
        json.dump(all_chunks, f)
    tmp_chunks.rename(FINAL_INDEX_DIR / "chunks.json")
    log(f"  Wrote chunks.json")

    # Write metadata.jsonl
    tmp_meta = FINAL_INDEX_DIR / "metadata.jsonl.tmp"
    with open(tmp_meta, 'w') as f:
        for m in all_metadata:
            f.write(json.dumps(m) + "\n")
    tmp_meta.rename(FINAL_INDEX_DIR / "metadata.jsonl")
    log(f"  Wrote metadata.jsonl")

    # Write summary
    summary = {
        "dataset_name": "stackexchange_python",
        "raw_dataset_path": "./datasets/stackexchange_python",
        "total_vectors": final_index.ntotal,
        "embedding_model": "intfloat/e5-large-v2",
        "embedding_dimension": EMBEDDING_DIM,
        "index_type": "IndexFlatIP",
        "split_merge": True,
        "num_splits": 5,
        "date_completed": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "integrity_status": "VERIFIED"
    }
    tmp_summary = FINAL_INDEX_DIR / "summary.json.tmp"
    with open(tmp_summary, 'w') as f:
        json.dump(summary, f, indent=2)
    tmp_summary.rename(FINAL_INDEX_DIR / "summary.json")
    log(f"  Wrote summary.json")

    log("")
    log("=" * 60)
    log("MERGE COMPLETE")
    log("=" * 60)
    log(f"  Output: {FINAL_INDEX_DIR}")
    log(f"  Total vectors: {final_index.ntotal:,}")
    log("=" * 60)

if __name__ == "__main__":
    merge_indexes()
