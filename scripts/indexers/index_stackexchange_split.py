#!/usr/bin/env python3
"""
Split indexer for StackExchange - processes a subset of files.
Accepts --start-file and --end-file parameters to process a range.

Usage:
    python3 index_stackexchange_split.py --split-id 1 --start-file 0 --end-file 15
"""

import argparse
import sys

# Import everything from the main indexer
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Read the main indexer and modify it
exec(open('/home/wade/CLAUDE/index_stackexchange_python.py').read())

# Override the discover_data_files function to filter by range
original_discover = discover_data_files

def discover_data_files_filtered(dataset_dir, start_idx, end_idx):
    """Find data files in specified range."""
    all_files = original_discover(dataset_dir)
    filtered = all_files[start_idx:end_idx]
    log(f"[SPLIT] Processing files {start_idx} to {end_idx-1} ({len(filtered)} files)")
    return filtered

# Override main to accept split parameters
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-id", type=int, required=True, help="Split ID (1-5)")
    parser.add_argument("--start-file", type=int, required=True, help="Start file index")
    parser.add_argument("--end-file", type=int, required=True, help="End file index (exclusive)")
    parser.add_argument("--dry-run", action="store_true", help="Scan and estimate without embedding")
    args = parser.parse_args()

    # Modify work directory to be split-specific
    NVME_WORK_DIR = Path(f"/home/wade/nvme_work/stackexchange_python_split{args.split_id}")
    DATASET_NAME = f"stackexchange_python_split{args.split_id}"

    # Override globals
    globals()['NVME_WORK_DIR'] = NVME_WORK_DIR
    globals()['DATASET_NAME'] = DATASET_NAME

    # Override discover function
    original_run = run_indexer

    def run_indexer_split(dry_run=False):
        global shutdown_requested

        log("=" * 60)
        log(f"{DATASET_NAME.upper()} SPLIT INDEXER")
        log("=" * 60)
        log(f"Dataset: {DATASET_DIR}")
        log(f"Index: {INDEX_DIR}")
        log(f"Work dir: {NVME_WORK_DIR}")
        log(f"Split: Files {args.start_file} to {args.end_file-1}")
        log(f"Batch size: {BATCH_START} (LOCKED)")
        log(f"Mode: {'DRY-RUN' if dry_run else 'PRODUCTION'}")

        # Setup directories
        NVME_WORK_DIR.mkdir(parents=True, exist_ok=True)
        (NVME_WORK_DIR / "vector_shards").mkdir(exist_ok=True)

        # Discover data files in range
        data_files = discover_data_files_filtered(DATASET_DIR, args.start_file, args.end_file)
        if not data_files:
            log("ERROR: No data files in specified range!")
            sys.exit(1)

        # Continue with rest of indexer...
        # (Copy the rest of run_indexer function but use data_files variable)

        if dry_run:
            log("")
            log("=" * 60)
            log("DRY-RUN MODE - Scanning dataset")
            log("=" * 60)

            total_rows = 0
            total_chunks_est = 0

            for file_idx, data_path in enumerate(data_files):
                log(f"[DRY-RUN] Scanning {data_path.name} ({file_idx+1}/{len(data_files)})")
                try:
                    if data_path.suffix == ".parquet":
                        meta = pq.read_metadata(data_path)
                        file_rows = meta.num_rows
                        total_rows += file_rows
                        total_chunks_est += int(file_rows * 1.5)
                except Exception as e:
                    log(f"[DRY-RUN] Error scanning {data_path.name}: {e}")

            log("")
            log("=" * 60)
            log("DRY-RUN COMPLETE")
            log("=" * 60)
            log(f"  Total Q&A pairs found: {total_rows:,}")
            log(f"  Estimated chunks: {total_chunks_est:,}")
            log(f"  Estimated vectors: {total_chunks_est:,}")
            log("  DRY_RUN_SUCCESS")
            log("=" * 60)
            return

        # Production mode - call original with our filtered files
        # Temporarily replace discover function
        import types
        saved_discover = globals()['discover_data_files']
        globals()['discover_data_files'] = lambda x: data_files

        # Run the original indexer logic
        original_run(dry_run=False)

        # Restore
        globals()['discover_data_files'] = saved_discover

    run_indexer_split(dry_run=args.dry_run)
