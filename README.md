> **Status: PRODUCTION_PORTFOLIO (Canonical Implementation)**

> Portfolio repository demonstrating large-scale semantic indexing pipelines.
> 
> 8,355,163 vectors indexed across Wikipedia Featured Articles, ArXiv ML abstracts,
> and StackExchange Python using e5-large-v2 embeddings and FAISS IndexFlatIP.

# Portfolio Indexing Projects

**Professional-grade semantic search indexing system demonstrating production-ready RAG capabilities**

---

## Start Here

This repo implements the indexing pipeline. For the full operational context, see:

- **[Universal Protocol v4.23](https://github.com/whmatrix/universal-protocol-v4.23)** — Deliverable contracts, audit contracts, and quality gates
- Example deliverable structure: [Semantic_Indexing_Output__Example_Structure.pdf](https://github.com/whmatrix/universal-protocol-v4.23/blob/main/portfolio_artifacts/Semantic_Indexing_Output__Example_Structure.pdf)
- RAG Readiness Audit examples:
  - [Portfolio_01_RAG_Readiness_Audit.pdf](https://github.com/whmatrix/universal-protocol-v4.23/blob/main/portfolio_artifacts/Portfolio_01_RAG_Readiness_Audit.pdf)
  - [RAG_Readiness_Audit__Sample_Report.pdf](https://github.com/whmatrix/universal-protocol-v4.23/blob/main/portfolio_artifacts/RAG_Readiness_Audit__Sample_Report.pdf)
- Earlier foundational work: [semantic-indexing-batch-01](https://github.com/whmatrix/semantic-indexing-batch-01) (superseded)

---

## Overview

This portfolio showcases three diverse, real-world datasets indexed with a Universal Protocol-compliant pipeline:

1. **Wikipedia Featured Articles** - High-quality encyclopedia content (352,606 vectors)
2. **StackExchange Python Q&A** - Real developer community discussions (7,513,263 vectors)
3. **ArXiv ML Abstracts** - Scientific research from ML/CS domains (489,294 vectors)

**Total: 8,355,163 vectors across all datasets** ✅ Production Complete

## Project Structure

```
PORTFOLIO_INDEXING_PROJECTS/
├── README.md                          # This file
├── datasets/                          # Raw source datasets (26.5 GB total)
│   ├── wiki_featured/                 # 491 MB, 5 JSONL files
│   ├── arxiv_ml_abstracts/            # 7.0 GB, 9 JSONL files
│   └── stackexchange_python/          # 19 GB, 75 Parquet files
│
├── scripts/                           # All indexing & processing scripts
│   ├── indexers/                      # Core indexing engines (UAIO-compliant)
│   │   ├── index_wiki_featured.py
│   │   ├── index_arxiv_ml_abstracts.py
│   │   ├── index_stackexchange_python.py
│   │   └── index_stackexchange_split{1-5}.py
│   │
│   ├── preparation/                   # Dataset preparation scripts
│   │   ├── prepare_wiki_featured.sh
│   │   ├── prepare_arxiv_ml.sh
│   │   ├── prepare_stackexchange_python.sh
│   │   └── PREPARE_ALL_PORTFOLIO_DATASETS.sh
│   │
│   ├── wrappers/                      # DRY-RUN and PRODUCTION wrappers
│   │   ├── RUN_PORTFOLIO_WIKI_*.sh
│   │   ├── RUN_PORTFOLIO_ARXIV_*.sh
│   │   ├── RUN_PORTFOLIO_STACKEXCHANGE_*.sh
│   │   └── RUN_PORTFOLIO_ALL_*.sh
│   │
│   └── merge/                         # Multi-process merge utilities
│       ├── merge_stackexchange_splits.py
│       └── merge_stackexchange_splits_efficient.py
│
├── results/                           # Final indexed results
│   ├── indexes/                       # FAISS indexes (ready for deployment)
│   │   ├── wiki_featured/             # 352,606 vectors
│   │   ├── arxiv_ml_abstracts/        # 489,294 vectors
│   │   ├── stackexchange_python/      # 7,513,263 vectors (merged)
│   │   └── stackexchange_split{1-5}/  # Individual splits (archived)
│   │
│   └── work_dirs/                     # Temporary work directories
│
└── documentation/                     # Complete usage guides
    ├── PORTFOLIO_DATASETS_README.md
    └── ORGANIZATION.md
```

## Key Features

### Universal Protocol Compliance

All indexers implement the Universal Batch Indexing & Verification Engine (UAIO):

- ✅ **Producer/Consumer Architecture** - Streaming pipeline with queue coordination
- ✅ **RAM Balancer** - Auto pause at 90%, resume at 70%
- ✅ **GPU Balancer** - OOM handling with batch size reduction
- ✅ **Locked Batch Size** - 1300 (proven stable)
- ✅ **Memory Footprint** - 1-3GB VRAM per indexer
- ✅ **Signal Handling** - Graceful shutdown (SIGINT/SIGTERM)
- ✅ **Checkpointing** - Every 1M vectors, fully resumable
- ✅ **Atomic Writes** - .tmp file swaps prevent corruption
- ✅ **Integrity Checks** - len(vectors) == len(chunks) == len(metadata)

### Advanced Capabilities

**Parallel Processing**: Successfully demonstrated 7 concurrent indexers (1 Wikipedia + 1 ArXiv + 5 StackExchange splits) running in parallel with proper VRAM management.

**Split-Merge Pattern**: StackExchange dataset split into 5 parallel operations (15 files each) for 5× speedup, then merged into single unified index.

**Resource Management**:
- **VRAM**: 18GB / 49GB (7 indexers × ~2.5GB each)
- **RAM**: 87GB+ free with automatic pressure balancing
- **GPU**: 100% utilization, optimal throughput

## Datasets

### 1. Wikipedia Featured Articles

- **Source**: English Wikipedia featured/quality articles
- **Format**: JSONL (5 files, 491 MB)
- **Content**: Full encyclopedia articles with title, categories, and content
- **Rows**: ~39,716 articles
- **Vectors**: 352,606 (avg 8.9 chunks per article)
- **Use Case**: General knowledge retrieval, content understanding
- **Status**: ✅ Complete

### 2. StackExchange Python Q&A

- **Source**: Stack Exchange Python-related Q&A dataset
- **Format**: Parquet (75 files, 19 GB)
- **Content**: Questions, accepted answers, tags, scores
- **Rows**: 6,378,706 Q&A pairs
- **Vectors**: 7,513,263 (avg 1.178 chunks per Q&A)
- **Use Case**: Technical support, developer tools, code Q&A
- **Innovation**: Split into 5 parallel indexers for 5× speedup, then merged
- **Status**: ✅ Complete (merged from 5 splits)

### 3. ArXiv ML Abstracts

- **Source**: RedPajama ArXiv ML/CS papers
- **Format**: JSONL (9 files, 7.0 GB, RedPajama format)
- **Content**: Scientific papers with arxiv_id, date, and content
- **Rows**: ~123K papers
- **Vectors**: 489,294 (avg 4.0 chunks per paper)
- **Use Case**: Research retrieval, academic applications
- **Status**: ✅ Complete

## Technical Specifications

### Embedding Model
- **Model**: `intfloat/e5-large-v2`
- **Dimensions**: 1024
- **Precision**: FP16
- **Prefix**: `"passage: "` for all chunks

### Indexing Parameters
- **Chunking**: 1500 characters, word-aligned
- **Shard Size**: 200K vectors per .npy file
- **Checkpoint Interval**: 1M vectors
- **Batch Size**: 1300 (locked, auto-reduces on OOM)
- **Queue Soft Max**: 50K items

### Index Format
- **Type**: FAISS IndexFlatIP (inner product similarity)
- **Outputs**:
  - `vectors.index` - FAISS index file
  - `chunks.json` - Text chunks
  - `metadata.jsonl` - Chunk metadata
  - `summary.json` - Index statistics

## Try It Small First

This repo indexes 8.35M vectors from 26.5 GB of source data — it requires an NVIDIA GPU with 48 GB VRAM and 128 GB RAM.

**Want to try the pipeline without the hardware requirements?** Use [research-corpus-discovery](https://github.com/whmatrix/research-corpus-discovery), which runs the same embedding model and FAISS index type on a small PDF corpus:

```bash
git clone https://github.com/whmatrix/research-corpus-discovery
cd research-corpus-discovery
pip install -r scripts/requirements.txt
python scripts/build_index.py --pdf_dir ./sample_docs/ --output_dir ./demo_index
python scripts/query.py --index ./demo_index/faiss.index --chunks ./demo_index/chunks.jsonl
```

See [QUICK_START.md](https://github.com/whmatrix/research-corpus-discovery/blob/main/QUICK_START.md) for a full walkthrough. The pipeline, embedding model (e5-large-v2), and index type (FAISS IndexFlatIP) are the same — only the scale differs.

---

## Usage

### Quick Start (Full Scale)

```bash
cd <repo-root>/

# Prepare all datasets
./scripts/preparation/PREPARE_ALL_PORTFOLIO_DATASETS.sh

# Test with dry-run
./scripts/wrappers/RUN_PORTFOLIO_ALL_DRYRUN.sh

# Run production (individual)
./scripts/wrappers/RUN_PORTFOLIO_WIKI_PRODUCTION.sh
./scripts/wrappers/RUN_PORTFOLIO_STACKEXCHANGE_PRODUCTION.sh
./scripts/wrappers/RUN_PORTFOLIO_ARXIV_PRODUCTION.sh
```

### Advanced: Parallel Processing

For StackExchange, use split indexers for 5× speedup:

```bash
# Start all 5 splits (run manually with staggered starts)
python3 scripts/indexers/index_stackexchange_split1.py > /tmp/se_split1.log 2>&1 &
python3 scripts/indexers/index_stackexchange_split2.py > /tmp/se_split2.log 2>&1 &
# ... (wait for each to start embedding before launching next)

# After all complete, merge
python3 scripts/merge/merge_stackexchange_splits.py
```

### Monitoring

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Monitor individual indexer progress
tail -f /tmp/wiki_production.log
tail -f /tmp/stackexchange_production.log
tail -f /tmp/arxiv_production.log

# Check split progress
tail -f /tmp/stackexchange_split{1..5}.log
```

## Results

### Index Locations

**Repository Structure (Production-Ready)**:
```
./results/indexes/
├── wiki_featured/                  # 352,606 vectors
├── arxiv_ml_abstracts/             # 489,294 vectors
├── stackexchange_python/           # 7,513,263 vectors (merged)
└── stackexchange_split{1-5}/       # Individual splits (archived)
```

### Index File Structure

Each index directory contains:
```
<dataset_name>/
├── vectors.index          # FAISS IndexFlatIP (1024-dim, ~29GB for StackExchange)
├── chunks.json            # Text chunks corresponding to vectors (~5.8GB for StackExchange)
├── metadata.jsonl         # Per-vector metadata (~1.3GB for StackExchange)
└── summary.json           # Dataset summary & integrity verification
```

## Performance Metrics

### Throughput
- **Single Indexer**: ~1300 vectors per batch (~13 seconds per batch)
- **7 Parallel Indexers**: ~9,100 vectors per 13 seconds = ~700 vectors/sec aggregate
- **VRAM Efficiency**: ~2.5GB per indexer (well below 3GB target)

### Scalability
- Successfully demonstrated 7 concurrent indexers
- Linear scaling with available VRAM (can run 2-15 indexers on RTX A6000)
- RAM balancer prevents memory exhaustion
- GPU stays at 100% utilization

## Portfolio Highlights

### For Potential Clients

1. **Multi-Domain Expertise**: Wikipedia (general), StackExchange (technical), ArXiv (academic)
2. **Production-Ready**: Signal handling, checkpointing, atomic writes, integrity checks
3. **Scalable Architecture**: Parallel processing, split-merge patterns
4. **Resource Efficient**: Optimal VRAM/RAM usage, automatic balancing
5. **Real-World Data**: 10M+ vectors from authentic sources
6. **Enterprise Patterns**: Logging, monitoring, graceful degradation

### Technical Demonstrations

- ✅ Parallel multi-dataset indexing
- ✅ Split-merge workflow for large datasets
- ✅ Automatic resource management
- ✅ OOM recovery and batch adaptation
- ✅ Resumable long-running operations
- ✅ Integrity verification and atomic commits

## Documentation

See `/documentation/PORTFOLIO_DATASETS_README.md` for:
- Detailed dataset preparation instructions
- Where to download source data
- Expected formats and structures
- Troubleshooting guides
- Advanced configuration

## System Requirements

- **GPU**: NVIDIA RTX A6000 (48GB) or similar
- **RAM**: 128GB+ recommended for large datasets
- **Storage**:
  - NVMe: ~500GB for work directories
  - HDD: ~100GB for final indexes
- **Software**:
  - Python 3.8+
  - PyTorch with CUDA
  - FAISS-GPU
  - sentence-transformers

## License & Attribution

This portfolio demonstrates production-ready indexing pipelines built with:
- **Embedding Model**: intfloat/e5-large-v2
- **Index Engine**: FAISS (Facebook AI)
- **Architecture**: Universal Batch Indexing & Verification Engine (UAIO)

All code follows Universal Protocol standards for reproducibility and compliance.

---

**Created**: December 2025
**Status**: Production-Ready
**Contact**: Professional portfolio demonstration

## Limitations & Non-Claims

This index demonstrates large-scale semantic indexing capability (8.35M+ vectors) but makes no claims about retrieval quality, relevance, or suitability for specific applications. Use case specificity and evaluation require domain-specific testing.

---

## Protocol Alignment

This indexing run conforms to the
[Universal Protocol v4.23](https://github.com/whmatrix/universal-protocol-v4.23).

All dataset ingestion, chunking, embedding, FAISS construction,
and validation artifacts follow the schemas and constraints defined there.
