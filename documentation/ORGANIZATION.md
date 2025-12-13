# Portfolio Indexing Projects - Organization Guide

## Overview

This document describes the organizational structure and file management principles for the Portfolio Indexing Projects directory within the Professional & Clear collection.

**Last Updated**: 2025-12-13
**Status**: Production Complete ✅

---

## Directory Organization Principles

### 1. Self-Contained Structure
The entire portfolio is self-contained within a single directory tree, making it portable and easy to understand:

```
PORTFOLIO_INDEXING_PROJECTS/
├── datasets/          # Raw data (self-contained copies)
├── scripts/           # All processing code
├── results/           # Final outputs
└── documentation/     # Guides and specifications
```

### 2. Separation of Concerns

**datasets/** - Raw source data only
- No processing happens here
- Read-only after initial setup
- Complete copies, not symlinks (for portability)

**scripts/** - All executable code
- Organized by function (indexers, preparation, wrappers, merge)
- Each script is standalone and documented
- Clear naming conventions (verb_noun pattern)

**results/** - All outputs
- indexes/ - Final production-ready FAISS indexes
- work_dirs/ - Temporary processing artifacts
- Clear separation between final and intermediate results

**documentation/** - Human-readable guides
- README files
- Organization specs
- Usage examples

### 3. Consistent Naming Conventions

**Datasets**:
- `wiki_featured` - Wikipedia Featured Articles
- `arxiv_ml_abstracts` - ArXiv ML/CS Papers
- `stackexchange_python` - StackExchange Python Q&A

**Scripts**:
- `index_<dataset>.py` - Main indexing scripts
- `prepare_<dataset>.sh` - Dataset preparation
- `RUN_PORTFOLIO_<DATASET>_<MODE>.sh` - Execution wrappers
- `merge_<dataset>_<variant>.py` - Merge utilities

**Modes**:
- `DRYRUN` - Pre-flight validation (read-only, fast)
- `PRODUCTION` - Full indexing operation

---

## Detailed Structure

### datasets/

Raw source datasets copied for self-containment:

```
datasets/
├── wiki_featured/                     # 491 MB
│   ├── wiki_shard_0.jsonl
│   ├── wiki_shard_1.jsonl
│   ├── wiki_shard_2.jsonl
│   ├── wiki_shard_3.jsonl
│   └── wiki_shard_4.jsonl
│
├── arxiv_ml_abstracts/                # 7.0 GB
│   ├── arxiv_*.jsonl (9 files)
│   └── [RedPajama format JSONL files]
│
└── stackexchange_python/              # 19 GB
    └── train-*-of-00075-*.parquet (75 files)
```

**Note**: These are actual file copies (not symlinks) to make the portfolio self-contained and portable.

### scripts/

#### indexers/
Core indexing engines implementing UAIO protocol:

```
indexers/
├── index_wiki_featured.py              # Wikipedia indexer
├── index_arxiv_ml_abstracts.py         # ArXiv indexer (RedPajama format)
├── index_stackexchange_python.py       # StackExchange full indexer
├── index_stackexchange_split1.py       # StackExchange split 1/5 (files 0-14)
├── index_stackexchange_split2.py       # StackExchange split 2/5 (files 15-29)
├── index_stackexchange_split3.py       # StackExchange split 3/5 (files 30-44)
├── index_stackexchange_split4.py       # StackExchange split 4/5 (files 45-59)
└── index_stackexchange_split5.py       # StackExchange split 5/5 (files 60-74)
```

**Key Features**:
- UAIO protocol compliant
- DRY_RUN and PRODUCTION modes
- Signal handling (SIGINT/SIGTERM)
- Checkpointing every 1M vectors
- Atomic file writes (.tmp swaps)

#### preparation/
Dataset setup scripts:

```
preparation/
├── prepare_wiki_featured.sh            # Concatenate wiki shards
├── prepare_arxiv_ml.sh                 # Symlink ArXiv files
├── prepare_stackexchange_python.sh     # Symlink StackExchange parquet files
└── PREPARE_ALL_PORTFOLIO_DATASETS.sh   # Run all preparation scripts
```

**Purpose**: Transform raw data into indexer-ready format

#### wrappers/
Convenience scripts for common operations:

```
wrappers/
├── RUN_PORTFOLIO_WIKI_DRYRUN.sh
├── RUN_PORTFOLIO_WIKI_PRODUCTION.sh
├── RUN_PORTFOLIO_ARXIV_DRYRUN.sh
├── RUN_PORTFOLIO_ARXIV_PRODUCTION.sh
├── RUN_PORTFOLIO_STACKEXCHANGE_DRYRUN.sh
├── RUN_PORTFOLIO_STACKEXCHANGE_PRODUCTION.sh
├── RUN_PORTFOLIO_ALL_DRYRUN.sh
└── RUN_PORTFOLIO_ALL_PRODUCTION.sh
```

**Pattern**:
```bash
#!/bin/bash
cd "$(dirname "$0")/../indexers"
DRY_RUN=1 python3 index_<dataset>.py
```

#### merge/
Split-merge utilities:

```
merge/
├── merge_stackexchange_splits.py           # Original (memory-intensive)
└── merge_stackexchange_splits_efficient.py # Optimized (streaming)
```

**merge_stackexchange_splits_efficient.py** features:
- Processes one split at a time
- Memory-efficient streaming
- Garbage collection between splits
- Avoids OOM on large datasets

### results/

#### indexes/
Final production-ready FAISS indexes:

```
indexes/
├── wiki_featured/
│   ├── vectors.index          # 1.4 GB - FAISS IndexFlatIP
│   ├── chunks.json            # 201 MB - Text chunks
│   ├── metadata.jsonl         # 58 MB - Per-vector metadata
│   └── summary.json           # 393 B - Index statistics
│
├── arxiv_ml_abstracts/
│   ├── vectors.index          # 1.9 GB
│   ├── chunks.json            # 387 MB
│   ├── metadata.jsonl         # 85 MB
│   └── summary.json
│
├── stackexchange_python/
│   ├── vectors.index          # 29 GB
│   ├── chunks.json            # 5.8 GB
│   ├── metadata.jsonl         # 1.3 GB
│   └── summary.json
│
└── stackexchange_split{1-5}/  # Individual splits (archived)
    └── [same structure as above]
```

**summary.json format**:
```json
{
  "dataset_name": "stackexchange_python",
  "total_vectors": 7513263,
  "embedding_model": "intfloat/e5-large-v2",
  "embedding_dimension": 1024,
  "index_type": "IndexFlatIP",
  "date_completed": "2025-12-13T04:52:54",
  "integrity_status": "VERIFIED"
}
```

#### work_dirs/
Temporary processing artifacts (created during indexing):

```
work_dirs/
├── wiki_featured/
│   └── vector_shards/
│       └── shard_*.npy (200K vectors per shard)
├── arxiv_ml_abstracts/
│   └── vector_shards/
└── stackexchange_python/
    └── vector_shards/
```

**Note**: These are temporary and can be deleted after indexing completes.

---

## File Naming Standards

### Scripts

**Pattern**: `<action>_<target>_<variant>.ext`

Examples:
- `index_wiki_featured.py` - Index the wiki_featured dataset
- `prepare_arxiv_ml.sh` - Prepare the arxiv_ml dataset
- `merge_stackexchange_splits_efficient.py` - Merge StackExchange with efficiency optimizations

### Datasets

**Pattern**: `<source>_<subset>`

- `wiki_featured` - Wikipedia featured articles
- `arxiv_ml_abstracts` - ArXiv ML/CS abstracts
- `stackexchange_python` - StackExchange Python Q&A

### Index Output

**Pattern**: Same as dataset name

- Consistency: Dataset name matches index directory name
- Traceability: Easy to map raw data to indexed results

---

## Path Management

### Absolute Paths in Scripts

Scripts contain hardcoded paths for clarity and reproducibility:

```python
DATASET_DIR = Path("/media/wade/gork/training_datasets/misc_datasets/wiki_featured")
INDEX_DIR = Path("/media/wade/gork/indexed/wiki_featured")
WORK_DIR = Path("/home/wade/nvme_work/wiki_featured")
```

**Rationale**:
- Clear documentation of expected locations
- No ambiguity about file locations
- Easy to adapt for different environments

### Professional & Clear Folder

This self-contained copy uses local paths:

```python
BASE = "/home/wade/CLAUDE/__UNIVERSAL_PROTOCOL_CORE__/Professional & Clear/PORTFOLIO_INDEXING_PROJECTS"
DATASET_DIR = Path(f"{BASE}/datasets/wiki_featured")
INDEX_DIR = Path(f"{BASE}/results/indexes/wiki_featured")
```

**Note**: Original scripts reference `/media/wade/gork` paths as reference implementations.

---

## Operational Workflow

### 1. Preparation Phase
```bash
cd scripts/preparation
./PREPARE_ALL_PORTFOLIO_DATASETS.sh
```

Creates dataset-ready structure from raw files.

### 2. Dry-Run Phase
```bash
cd scripts/wrappers
./RUN_PORTFOLIO_ALL_DRYRUN.sh
```

Validates datasets and estimates vectors without indexing.

### 3. Production Phase
```bash
# Option A: Sequential
./RUN_PORTFOLIO_WIKI_PRODUCTION.sh
./RUN_PORTFOLIO_ARXIV_PRODUCTION.sh
./RUN_PORTFOLIO_STACKEXCHANGE_PRODUCTION.sh

# Option B: Parallel (staggered starts)
./RUN_PORTFOLIO_WIKI_PRODUCTION.sh &
sleep 60 && ./RUN_PORTFOLIO_ARXIV_PRODUCTION.sh &
sleep 60 && ./RUN_PORTFOLIO_STACKEXCHANGE_PRODUCTION.sh &
```

### 4. Merge Phase (for split datasets)
```bash
cd scripts/merge
python3 merge_stackexchange_splits_efficient.py
```

Combines multiple split indexes into unified index.

---

## Storage Management

### Disk Usage

**datasets/** - 26.5 GB total:
- wiki_featured: 491 MB
- arxiv_ml_abstracts: 7.0 GB
- stackexchange_python: 19 GB

**results/indexes/** - ~70 GB total:
- wiki_featured: ~1.7 GB
- arxiv_ml_abstracts: ~2.4 GB
- stackexchange_python: ~36 GB
- stackexchange_split{1-5}: ~30 GB (can be deleted after merge)

**work_dirs/** - Temporary (delete after completion):
- Variable size depending on processing stage
- Automatically cleaned on successful completion

### Cleanup Recommendations

After successful indexing:
1. ✅ Keep: `datasets/`, `results/indexes/`, `scripts/`
2. ⚠️ Optional: `results/indexes/stackexchange_split{1-5}/` (already merged)
3. ❌ Delete: `results/work_dirs/` (temporary processing artifacts)

---

## Portability Considerations

### Self-Contained Design

This portfolio is designed to be portable:

1. **All dependencies local**: Raw data, scripts, results in one tree
2. **No external symlinks**: All files are actual copies
3. **Clear documentation**: README + ORGANIZATION files
4. **Reproducible paths**: Base path can be adjusted

### Adapting to New Environment

To deploy in a new environment:

1. Copy entire `PORTFOLIO_INDEXING_PROJECTS/` directory
2. Update base paths in scripts (if needed)
3. Verify Python dependencies: `pip install -r requirements.txt`
4. Run dry-run to validate: `./scripts/wrappers/RUN_PORTFOLIO_ALL_DRYRUN.sh`

---

## Version Control

### Git Integration

Recommended `.gitignore` patterns:
```
# Temporary files
results/work_dirs/
*.tmp
*.log

# Large binaries (use Git LFS)
datasets/
results/indexes/*/vectors.index
results/indexes/*/chunks.json

# Keep small files
!results/indexes/*/summary.json
!results/indexes/*/metadata.jsonl
```

### Archiving

For long-term archival:
- **Essential**: `scripts/`, `documentation/`, `results/indexes/*/summary.json`
- **Optional**: `datasets/` (can re-download)
- **Omit**: `results/work_dirs/`, `results/indexes/stackexchange_split*/`

---

## Best Practices

### 1. Always Run Dry-Run First
```bash
# GOOD
./RUN_PORTFOLIO_WIKI_DRYRUN.sh
./RUN_PORTFOLIO_WIKI_PRODUCTION.sh

# BAD
./RUN_PORTFOLIO_WIKI_PRODUCTION.sh  # Skip dry-run
```

### 2. Monitor Resource Usage
```bash
# GPU
watch -n 1 nvidia-smi

# RAM
watch -n 1 free -h

# Disk
df -h /home/wade/nvme_work
```

### 3. Stagger Parallel Starts
```bash
# GOOD: Wait for embedding to start
indexer1 & sleep 60
indexer2 & sleep 60
indexer3 &

# BAD: All at once
indexer1 & indexer2 & indexer3 &
```

### 4. Verify After Completion
```bash
# Check summary
cat results/indexes/wiki_featured/summary.json

# Verify integrity
python3 -c "
import faiss, json
idx = faiss.read_index('results/indexes/wiki_featured/vectors.index')
meta_count = sum(1 for _ in open('results/indexes/wiki_featured/metadata.jsonl'))
print(f'Vectors: {idx.ntotal:,}, Metadata: {meta_count:,}, Match: {idx.ntotal == meta_count}')
"
```

---

## Troubleshooting

### Common Issues

**Issue**: OOM during merge
**Solution**: Use `merge_stackexchange_splits_efficient.py` instead of standard merge

**Issue**: Disk space exhausted
**Solution**: Clean up `work_dirs/` and old split indexes

**Issue**: Vector count mismatch
**Solution**: Check dry-run logs - estimates are approximate, production is exact

---

## Maintenance

### Regular Tasks

**Weekly**:
- Check disk space in `results/` and `work_dirs/`
- Review log files for errors

**Monthly**:
- Verify index integrity
- Update documentation if process changes

**As Needed**:
- Re-run indexing if source data updates
- Archive old indexes before re-indexing

---

## Contact & Support

For questions about this organizational structure:
- Reference: Universal Protocol Core documentation
- Location: `/home/wade/CLAUDE/__UNIVERSAL_PROTOCOL_CORE__/`

---

**Document Version**: 1.0
**Last Updated**: 2025-12-13
**Maintained By**: Professional & Clear Portfolio Collection
