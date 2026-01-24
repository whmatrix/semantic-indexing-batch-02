#!/usr/bin/env python3
"""
Memory-efficient merge of 5 StackExchange split indexes into one final index.
Uses streaming to avoid loading all data into RAM at once.

Usage:
    python3 merge_stackexchange_splits_efficient.py
"""

import json
import sys
from pathlib import Path
import numpy as np
import faiss
import time
import gc

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
    log("MERGING STACKEXCHANGE SPLIT INDEXES (MEMORY-EFFICIENT)")
    log("=" * 60)

    # Create output directory
    FINAL_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize final FAISS index
    log("Creating final FAISS IndexFlatIP...")
    final_index = faiss.IndexFlatIP(EMBEDDING_DIM)

    # Open output files for streaming write
    tmp_meta = FINAL_INDEX_DIR / "metadata.jsonl.tmp"
    tmp_chunks = FINAL_INDEX_DIR / "chunks.json.tmp"

    all_chunks = []
    total_vectors = 0

    # Process each split one at a time
    for i, split_dir in enumerate(SPLIT_INDEX_DIRS, 1):
        log(f"Processing split {i} from {split_dir}...")

        index_path = split_dir / "vectors.index"
        chunks_path = split_dir / "chunks.json"
        metadata_path = split_dir / "metadata.jsonl"

        if not index_path.exists():
            log(f"  WARNING: {index_path} not found, skipping split {i}")
            continue

        # Load and add vectors to index
        log(f"  Loading vectors...")
        split_index = faiss.read_index(str(index_path))
        vectors = faiss.rev_swig_ptr(split_index.get_xb(), split_index.ntotal * EMBEDDING_DIM)
        vectors = np.array(vectors).reshape(split_index.ntotal, EMBEDDING_DIM).astype(np.float32)

        log(f"  Adding {split_index.ntotal:,} vectors to final index...")
        final_index.add(vectors)
        total_vectors += split_index.ntotal

        # Free memory
        del vectors
        del split_index
        gc.collect()

        log(f"  Added {total_vectors:,} vectors total so far")

        # Append chunks
        log(f"  Loading chunks...")
        if chunks_path.exists():
            with open(chunks_path, 'r') as f:
                chunks = json.load(f)
                all_chunks.extend(chunks)
            log(f"  Loaded {len(chunks):,} chunks")
            del chunks
            gc.collect()

        # Stream metadata (append to file)
        log(f"  Streaming metadata...")
        if metadata_path.exists():
            with open(metadata_path, 'r') as src:
                with open(tmp_meta, 'a') as dst:
                    for line in src:
                        dst.write(line)
            log(f"  Metadata appended")

        log(f"  Split {i} complete, memory freed")
        log("")

    log(f"Total vectors in final index: {final_index.ntotal:,}")
    log(f"Total chunks collected: {len(all_chunks):,}")

    # Count metadata lines
    metadata_count = 0
    with open(tmp_meta, 'r') as f:
        for _ in f:
            metadata_count += 1
    log(f"Total metadata entries: {metadata_count:,}")

    # Integrity check
    if final_index.ntotal != len(all_chunks) or final_index.ntotal != metadata_count:
        log("INTEGRITY CHECK: FAILED")
        log(f"  Mismatch: vectors={final_index.ntotal}, chunks={len(all_chunks)}, metadata={metadata_count}")
        sys.exit(1)
    log("INTEGRITY CHECK: PASSED")

    log("")
    log("Writing final files...")

    # Write vectors.index
    log("  Writing vectors.index...")
    tmp_index = FINAL_INDEX_DIR / "vectors.index.tmp"
    faiss.write_index(final_index, str(tmp_index))
    tmp_index.rename(FINAL_INDEX_DIR / "vectors.index")
    log(f"  Wrote vectors.index ({final_index.ntotal:,} vectors)")

    # Write chunks.json
    log("  Writing chunks.json...")
    chunks_tmp = FINAL_INDEX_DIR / "chunks.json.tmp.2"
    with open(chunks_tmp, 'w') as f:
        json.dump(all_chunks, f)
    chunks_tmp.rename(tmp_chunks)
    tmp_chunks.rename(FINAL_INDEX_DIR / "chunks.json")
    log(f"  Wrote chunks.json")

    # Rename metadata
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
