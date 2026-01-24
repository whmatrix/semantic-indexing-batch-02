#!/usr/bin/env python3
"""
Full indexer for ArXiv ML Abstracts dataset.

This script:
1. Scans ArXiv JSONL files for ML/CS abstracts
2. Extracts title + abstract + categories
3. Chunks into ~1500 char segments (though abstracts are usually short)
4. Embeds with e5-large-v2
5. Builds FAISS IndexFlatIP

Dataset: ./datasets/arxiv_ml_abstracts
Index:   ./results/indexes/arxiv_ml_abstracts
Work:    ./results/work_dirs/arxiv_ml_abstracts

Usage:
    python3 index_arxiv_ml_abstracts.py --dry-run
    python3 index_arxiv_ml_abstracts.py
"""

import argparse
import gc
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path
from queue import Queue, Empty
from typing import Set, Tuple

import faiss
import numpy as np
import psutil
import torch
from sentence_transformers import SentenceTransformer

# =============================================================================
# CONFIGURATION
# =============================================================================

DATASET_NAME = "arxiv_ml_abstracts"
# NOTE: Update these paths for your local environment
DATASET_DIR = Path("./datasets/arxiv_ml_abstracts")
INDEX_DIR = Path("./results/indexes/arxiv_ml_abstracts")
NVME_WORK_DIR = Path("./results/work_dirs/arxiv_ml_abstracts")

EMBEDDING_MODEL = "intfloat/e5-large-v2"
EMBEDDING_DIM = 1024
CHUNK_SIZE = 1500

NUM_CPU_WORKERS = 12
QUEUE_SOFT_MAX = 50000

# BATCH SIZE LOCKED AT 1300
BATCH_START = 1300
BATCH_DELTA_OOM = 100
MIN_FREE_VRAM_GB = 3

RAM_MIN_FREE_STARTUP_GB = 12
RAM_HIGH_THRESHOLD = 0.90
RAM_NORMAL_THRESHOLD = 0.70

CHECKPOINT_INTERVAL = 1_000_000
SHARD_SIZE = 200_000
ROW_BATCH_SIZE = 5000

# =============================================================================
# GLOBALS
# =============================================================================

shutdown_requested = False
producer_paused = False
pause_lock = threading.Lock()

def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def signal_handler(signum, frame):
    global shutdown_requested
    log(f"Signal {signum} received, requesting graceful shutdown...")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# =============================================================================
# RAM BALANCER
# =============================================================================

def get_ram_usage() -> float:
    return psutil.virtual_memory().percent / 100.0

def get_free_ram_gb() -> float:
    return psutil.virtual_memory().available / (1024**3)

def wait_for_ram_startup():
    log(f"[RAM_BALANCER] Checking available RAM before startup...")
    free_gb = get_free_ram_gb()
    if free_gb < RAM_MIN_FREE_STARTUP_GB:
        log(f"[RAM_BALANCER] Only {free_gb:.1f} GB free, waiting...")
        while get_free_ram_gb() < RAM_MIN_FREE_STARTUP_GB:
            time.sleep(5)
    log(f"[RAM_BALANCER] Free RAM: {get_free_ram_gb():.1f} GB - OK")

def check_ram_pressure() -> bool:
    global producer_paused
    usage = get_ram_usage()
    with pause_lock:
        if usage > RAM_HIGH_THRESHOLD and not producer_paused:
            producer_paused = True
            log(f"[RAM_BALANCER] Pausing producers (RAM={usage*100:.1f}%)")
            return True
        elif usage < RAM_NORMAL_THRESHOLD and producer_paused:
            producer_paused = False
            log(f"[RAM_BALANCER] Resuming producers (RAM={usage*100:.1f}%)")
    return producer_paused

# =============================================================================
# GPU BALANCER
# =============================================================================

def get_gpu_memory() -> tuple:
    if not torch.cuda.is_available():
        return 0, 0
    torch.cuda.synchronize()
    used = torch.cuda.memory_allocated() / (1024**3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    return used, total - used

def gpu_cleanup():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

# =============================================================================
# TEXT EXTRACTION
# =============================================================================

def clean_latex(text: str) -> str:
    """Remove common LaTeX commands from text."""
    if not text:
        return ""

    # Remove common LaTeX commands
    replacements = [
        (r'\emph{', ''),
        (r'\textbf{', ''),
        (r'\textit{', ''),
        (r'\cite{', '['),
        (r'\ref{', '['),
        ('}', ''),
        (r'\$', ''),
        (r'$$', ''),
    ]

    for old, new in replacements:
        text = text.replace(old, new)

    # Remove standalone backslash commands
    import re
    text = re.sub(r'\\[a-zA-Z]+', ' ', text)

    # Normalize whitespace
    text = ' '.join(text.split())

    return text

def extract_text_from_paper(paper: dict) -> str:
    """Extract text from an ArXiv paper record (RedPajama format)."""
    parts = []

    # Check if this is RedPajama format (has "text" and "meta")
    if "text" in paper and "meta" in paper:
        meta = paper.get("meta", {})
        text = paper.get("text", "").strip()

        if not text:
            return None

        # Extract arxiv_id from meta
        arxiv_id = meta.get("arxiv_id", "")
        if arxiv_id:
            parts.append(f"[ARXIV_ID] {arxiv_id}")

        # Extract year/month from meta
        yymm = meta.get("yymm", "")
        if yymm:
            parts.append(f"[DATE] {yymm}")

        # Clean and truncate text (use first ~5000 chars as "abstract")
        text = clean_latex(text)
        # Take first section or first 5000 chars as a summary
        if len(text) > 5000:
            text = text[:5000] + "..."

        parts.append(f"[CONTENT]\n{text}")

    else:
        # Standard ArXiv format
        # ArXiv ID (optional)
        arxiv_id = paper.get("id", "") or paper.get("arxiv_id", "")
        if arxiv_id:
            parts.append(f"[ARXIV_ID] {arxiv_id}")

        # Categories
        categories = paper.get("categories", "") or paper.get("category", "")
        if categories:
            if isinstance(categories, list):
                cat_str = ", ".join(str(c) for c in categories if c)
            else:
                cat_str = str(categories)
            if cat_str:
                parts.append(f"[CATEGORIES] {cat_str}")

        # Title
        title = paper.get("title", "").strip()
        if title:
            title = clean_latex(title)
            parts.append(f"[TITLE] {title}")

        # Authors (optional)
        authors = paper.get("authors", "") or paper.get("author", "")
        if authors:
            if isinstance(authors, list):
                author_str = ", ".join(str(a) for a in authors[:5] if a)  # First 5 authors
            else:
                author_str = str(authors)
            if author_str:
                author_str = clean_latex(author_str)
                parts.append(f"[AUTHORS] {author_str}")

        # Abstract
        abstract = paper.get("abstract", "").strip()
        if not abstract:
            return None

        abstract = clean_latex(abstract)
        parts.append(f"[ABSTRACT]\n{abstract}")

    full_text = "\n\n".join(parts)
    return full_text if full_text.strip() else None

def split_text(text: str, max_len: int = CHUNK_SIZE) -> list:
    """Split text into word-aligned chunks."""
    if len(text) <= max_len:
        return [text]

    chunks = []
    words = text.split()
    current = []
    current_len = 0

    for word in words:
        if current_len + len(word) + 1 > max_len and current:
            chunks.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += len(word) + 1

    if current:
        chunks.append(" ".join(current))

    return chunks

def discover_jsonl_files(dataset_dir: Path) -> list:
    """Find all JSONL files in the dataset directory."""
    files = sorted(dataset_dir.glob("*.jsonl"))
    log(f"Found {len(files)} JSONL files")
    return files

# =============================================================================
# STREAMING INDEX BUILDER
# =============================================================================

class StreamingIndexBuilder:
    def __init__(self, work_dir: Path):
        self.work_dir = work_dir
        self.progress_path = work_dir / "progress.json"
        self.chunks_path = work_dir / "chunks.jsonl"
        self.metadata_path = work_dir / "metadata.jsonl"
        self.vectors_dir = work_dir / "vector_shards"

        self.vectors_count = 0
        self.shard_index = 0
        self.shard_vectors = 0
        self.rows_processed = 0

        self.current_shard = []

    def save_checkpoint(self):
        data = {
            "vectors_count": self.vectors_count,
            "shard_index": self.shard_index,
            "shard_vectors": self.shard_vectors,
            "rows_processed": self.rows_processed,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        }
        tmp_path = self.progress_path.with_suffix('.tmp')
        with open(tmp_path, 'w') as f:
            json.dump(data, f, indent=2)
        tmp_path.rename(self.progress_path)

    def flush_shard(self):
        if not self.current_shard:
            return
        shard_path = self.vectors_dir / f"shard_{self.shard_index:04d}.npy"
        arr = np.array(self.current_shard, dtype=np.float32)
        np.save(shard_path, arr)
        log(f"[SHARD] Wrote {shard_path.name}: {len(self.current_shard):,} vectors")
        self.current_shard = []
        self.shard_index += 1
        self.shard_vectors = 0

    def add_vectors(self, vectors: np.ndarray):
        for vec in vectors:
            self.current_shard.append(vec)
            self.shard_vectors += 1
            self.vectors_count += 1

            if self.shard_vectors >= SHARD_SIZE:
                self.flush_shard()

# =============================================================================
# PRODUCER
# =============================================================================

def producer_thread(queue: Queue, jsonl_files: list, dry_run: bool, dry_run_limit: int):
    """Read JSONL files and produce chunks."""
    global shutdown_requested, producer_paused

    total_items = 0
    skipped_empty = 0

    log(f"[PRODUCER] Starting full index scan...")

    for file_idx, jsonl_path in enumerate(jsonl_files):
        if shutdown_requested:
            break

        log(f"[PRODUCER] Processing {jsonl_path.name} ({file_idx+1}/{len(jsonl_files)})")

        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for row_idx, line in enumerate(f):
                    if shutdown_requested:
                        break

                    # RAM pressure check
                    while producer_paused and not shutdown_requested:
                        time.sleep(0.5)

                    # Queue size check
                    while queue.qsize() >= QUEUE_SOFT_MAX and not shutdown_requested:
                        time.sleep(0.1)

                    try:
                        paper = json.loads(line.strip())
                    except:
                        continue

                    # Extract text
                    text = extract_text_from_paper(paper)
                    if not text:
                        skipped_empty += 1
                        continue

                    # Split into chunks (though most abstracts fit in one chunk)
                    chunks = split_text(text, CHUNK_SIZE)

                    for chunk_idx, chunk in enumerate(chunks):
                        if len(chunk.strip()) < 20:
                            continue

                        metadata = {
                            "source": jsonl_path.name,
                            "file_index": file_idx,
                            "row": row_idx,
                            "chunk": chunk_idx,
                            "arxiv_id": paper.get("id", "") or paper.get("arxiv_id", "") or "",
                            "title": paper.get("title", "")[:100]
                        }

                        queue.put((chunk, metadata))
                        total_items += 1

                    # Dry-run limit
                    if dry_run and total_items >= dry_run_limit:
                        log(f"[PRODUCER] Dry-run limit reached: {total_items} items")
                        queue.put(None)
                        return

        except Exception as e:
            log(f"[PRODUCER] Error processing {jsonl_path.name}: {e}")
            continue

    queue.put(None)
    log(f"[PRODUCER] Finished: {total_items:,} items produced")
    log(f"[PRODUCER] Skipped: {skipped_empty:,} empty papers")

# =============================================================================
# FINALIZE - Build FAISS index
# =============================================================================

def finalize_index(builder: StreamingIndexBuilder):
    """Build final FAISS index from shards."""
    log("")
    log("=" * 60)
    log("FINALIZING INDEX")
    log("=" * 60)

    # Load all shards
    shard_files = sorted(builder.vectors_dir.glob("shard_*.npy"))
    if not shard_files:
        log("ERROR: No vector shards found!")
        sys.exit(1)

    all_vectors = []
    for sf in shard_files:
        arr = np.load(sf)
        all_vectors.append(arr)
        log(f"[FINALIZE] Loaded {sf.name}: {len(arr):,} vectors")

    vectors = np.vstack(all_vectors)
    log(f"[FINALIZE] Total vectors: {len(vectors):,}")

    # Load chunks and metadata
    chunks = []
    metadata = []

    with open(builder.chunks_path, 'r') as f:
        for line in f:
            try:
                chunks.append(json.loads(line.strip()))
            except:
                continue

    with open(builder.metadata_path, 'r') as f:
        for line in f:
            try:
                metadata.append(json.loads(line.strip()))
            except:
                continue

    log(f"[FINALIZE] Total chunks: {len(chunks):,}")
    log(f"[FINALIZE] Total metadata: {len(metadata):,}")

    # Integrity check
    if len(vectors) != len(chunks) or len(vectors) != len(metadata):
        log("INTEGRITY CHECK: FAILED")
        log(f"  Mismatch: vectors={len(vectors)}, chunks={len(chunks)}, metadata={len(metadata)}")
        sys.exit(1)
    log("INTEGRITY CHECK: PASSED")

    # Build FAISS index
    log("Building FAISS IndexFlatIP...")
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(vectors.astype(np.float32))
    log(f"FAISS index: {index.ntotal:,} vectors")

    # Write to INDEX_DIR (atomic)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Write vectors.index
    tmp_index = INDEX_DIR / "vectors.index.tmp"
    faiss.write_index(index, str(tmp_index))
    tmp_index.rename(INDEX_DIR / "vectors.index")
    log(f"[FINALIZE] Wrote vectors.index")

    # Write chunks.json
    tmp_chunks = INDEX_DIR / "chunks.json.tmp"
    with open(tmp_chunks, 'w') as f:
        json.dump(chunks, f)
    tmp_chunks.rename(INDEX_DIR / "chunks.json")
    log(f"[FINALIZE] Wrote chunks.json")

    # Write metadata.jsonl
    tmp_meta = INDEX_DIR / "metadata.jsonl.tmp"
    with open(tmp_meta, 'w') as f:
        for m in metadata:
            f.write(json.dumps(m) + "\n")
    tmp_meta.rename(INDEX_DIR / "metadata.jsonl")
    log(f"[FINALIZE] Wrote metadata.jsonl")

    # Write summary
    summary = {
        "dataset_name": DATASET_NAME,
        "raw_dataset_path": str(DATASET_DIR),
        "total_vectors": index.ntotal,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": EMBEDDING_DIM,
        "index_type": "IndexFlatIP",
        "resume_mode": False,
        "date_completed": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "integrity_status": "VERIFIED"
    }
    tmp_summary = INDEX_DIR / "summary.json.tmp"
    with open(tmp_summary, 'w') as f:
        json.dump(summary, f, indent=2)
    tmp_summary.rename(INDEX_DIR / "summary.json")
    log(f"[FINALIZE] Wrote summary.json")

    return index.ntotal

# =============================================================================
# MAIN
# =============================================================================

def run_indexer(dry_run: bool = False):
    global shutdown_requested

    log("=" * 60)
    log(f"{DATASET_NAME.upper()} FULL INDEXER")
    log("=" * 60)
    log(f"Dataset: {DATASET_DIR}")
    log(f"Index: {INDEX_DIR}")
    log(f"Work dir: {NVME_WORK_DIR}")
    log(f"Batch size: {BATCH_START} (LOCKED)")
    log(f"Mode: {'DRY-RUN' if dry_run else 'PRODUCTION'}")

    # Setup directories
    NVME_WORK_DIR.mkdir(parents=True, exist_ok=True)
    (NVME_WORK_DIR / "vector_shards").mkdir(exist_ok=True)

    # Discover JSONL files
    jsonl_files = discover_jsonl_files(DATASET_DIR)
    if not jsonl_files:
        log("ERROR: No JSONL files found!")
        sys.exit(1)

    if dry_run:
        # Dry-run: just scan and count
        log("")
        log("=" * 60)
        log("DRY-RUN MODE - Scanning dataset")
        log("=" * 60)

        total_papers = 0
        total_chunks_est = 0

        for file_idx, jsonl_path in enumerate(jsonl_files):
            log(f"[DRY-RUN] Scanning {jsonl_path.name} ({file_idx+1}/{len(jsonl_files)})")
            try:
                with open(jsonl_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            paper = json.loads(line.strip())
                            text = extract_text_from_paper(paper)
                            if text:
                                total_papers += 1
                                # Most abstracts fit in one chunk
                                chunks_est = max(1, len(text) // CHUNK_SIZE)
                                total_chunks_est += chunks_est
                        except:
                            continue
            except Exception as e:
                log(f"[DRY-RUN] Error scanning {jsonl_path.name}: {e}")

        log("")
        log("=" * 60)
        log("DRY-RUN COMPLETE")
        log("=" * 60)
        log(f"  Total papers found: {total_papers:,}")
        log(f"  Estimated chunks: {total_chunks_est:,}")
        log(f"  Estimated vectors: {total_chunks_est:,}")
        log("  DRY_RUN_SUCCESS")
        log("=" * 60)
        return

    # Initialize builder
    builder = StreamingIndexBuilder(NVME_WORK_DIR)

    # RAM check
    wait_for_ram_startup()

    # GPU check
    log("[GPU_BALANCER] Pre-load VRAM reset...")
    gpu_cleanup()
    time.sleep(2)
    log(f"[GPU_BALANCER] Starting with LOCKED batch size: {BATCH_START}")

    # Load model
    log(f"Loading {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    model = model.half().cuda()
    log(f"Model loaded on cuda:0 (fp16)")

    # Open output files
    chunks_file = open(builder.chunks_path, 'w', encoding='utf-8')
    metadata_file = open(builder.metadata_path, 'w', encoding='utf-8')

    # Start producer
    queue = Queue()
    dry_run_limit = 5 * BATCH_START if dry_run else float('inf')

    producer = threading.Thread(
        target=producer_thread,
        args=(queue, jsonl_files, dry_run, dry_run_limit)
    )
    producer.start()

    # Embedding loop
    log("Starting streaming embedding loop...")
    batch_texts = []
    batch_meta = []
    current_batch_size = BATCH_START  # LOCKED at 1300
    total_embedded = 0
    last_checkpoint_vectors = 0

    while not shutdown_requested:
        check_ram_pressure()

        try:
            item = queue.get(timeout=1.0)
        except Empty:
            if not producer.is_alive():
                break
            continue

        if item is None:
            break

        text, meta = item
        batch_texts.append(f"passage: {text}")
        batch_meta.append(meta)

        if len(batch_texts) >= current_batch_size:
            try:
                with torch.no_grad():
                    embeddings = model.encode(
                        batch_texts,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        show_progress_bar=False
                    )

                # Write chunks and metadata
                for txt, m in zip(batch_texts, batch_meta):
                    chunks_file.write(json.dumps({"text": txt}) + "\n")
                    metadata_file.write(json.dumps(m) + "\n")

                # Add vectors
                builder.add_vectors(embeddings)
                builder.rows_processed += len(batch_texts)
                total_embedded += len(batch_texts)

                # Progress logging
                log(
                    f"[BATCH] {total_embedded:,} embedded, "
                    f"batch_size={current_batch_size}, "
                    f"vectors={builder.vectors_count:,}"
                )

                # Checkpoint
                if builder.vectors_count - last_checkpoint_vectors >= CHECKPOINT_INTERVAL:
                    builder.save_checkpoint()
                    last_checkpoint_vectors = builder.vectors_count
                    log(f"[CHECKPOINT] Saved at {builder.vectors_count:,} vectors")

                batch_texts = []
                batch_meta = []

            except torch.cuda.OutOfMemoryError:
                log(f"[GPU_BALANCER] OOM! Reducing batch {current_batch_size} -> {current_batch_size - BATCH_DELTA_OOM}")
                current_batch_size = max(100, current_batch_size - BATCH_DELTA_OOM)
                gpu_cleanup()
                # Re-queue items
                for t, m in zip(batch_texts, batch_meta):
                    queue.put((t.replace("passage: ", ""), m))
                batch_texts = []
                batch_meta = []

    # Process remaining
    if batch_texts and not shutdown_requested:
        try:
            with torch.no_grad():
                embeddings = model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )

            for txt, m in zip(batch_texts, batch_meta):
                chunks_file.write(json.dumps({"text": txt}) + "\n")
                metadata_file.write(json.dumps(m) + "\n")

            builder.add_vectors(embeddings)
            builder.rows_processed += len(batch_texts)
            total_embedded += len(batch_texts)
        except Exception as e:
            log(f"[ERROR] Final batch: {e}")

    # Cleanup
    producer.join(timeout=10)
    chunks_file.close()
    metadata_file.close()

    # Flush remaining shard
    builder.flush_shard()
    builder.save_checkpoint()

    if shutdown_requested:
        log("[SHUTDOWN] Graceful shutdown complete")
        return

    # Finalize index
    total_vectors = finalize_index(builder)

    log("")
    log("=" * 60)
    log("INDEXING COMPLETE")
    log(f"  Output: {INDEX_DIR}")
    log(f"  Total vectors: {total_vectors:,}")
    log("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Scan and estimate without embedding")
    args = parser.parse_args()

    run_indexer(dry_run=args.dry_run)
