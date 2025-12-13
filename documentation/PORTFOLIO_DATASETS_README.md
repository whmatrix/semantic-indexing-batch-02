# Portfolio Indexing Scripts

This directory contains three production-ready indexing scripts for building impressive portfolio demonstrations of semantic search and RAG capabilities.

## Overview

These scripts index three diverse, real-world datasets:

1. **Wikipedia Featured Articles** - High-quality encyclopedia content
2. **StackExchange Python Q&A** - Real developer community discussions
3. **ArXiv ML Abstracts** - Scientific paper abstracts from machine learning research

All scripts are **100% compliant** with the Universal Batch Indexing & Verification Engine (UAIO) protocol.

## Dataset Preparation

### 1. Wikipedia Featured Articles

**What it is:**
- High-quality Wikipedia articles marked as "Featured" or from specific categories (Science, Technology, etc.)
- Demonstrates ability to handle diverse, well-written human text
- Immediately recognizable to potential clients

**Where to get it:**

```bash
# Option 1: Download from Wikimedia dumps
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2

# Option 2: Use wikiextractor to filter featured articles
pip install wikiextractor
python -m wikiextractor.WikiExtractor \
  --json --filter_disambig_pages \
  --processes 8 \
  enwiki-latest-pages-articles.xml.bz2 \
  -o wiki_featured_output/

# Convert to JSONL format with featured article filtering
# (you can write a simple script to filter by category or featured status)
```

**Expected format (JSONL):**

```json
{"title": "Article Title", "text": "Article content...", "categories": ["Science", "Physics"], "is_featured": true}
```

**Where to place:**

```bash
/media/wade/gork/training_datasets/misc_datasets/wiki_featured/
```

### 2. StackExchange Python Q&A

**What it is:**
- Real Q&A pairs from Stack Overflow / Stack Exchange
- Filtered for Python-related questions
- Shows ability to handle structured technical content

**Where to get it:**

```bash
# Option 1: Stack Exchange Data Dump (archive.org)
# Download from: https://archive.org/details/stackexchange

# Option 2: Kaggle mirrors
# Visit: https://www.kaggle.com/datasets?search=stackoverflow

# Option 3: Use Stack Exchange API for recent data
# https://api.stackexchange.com/docs

# Filter for Python tag and convert to JSONL or Parquet
```

**Expected format (JSONL or Parquet):**

```json
{
  "title": "How to sort a dictionary by value?",
  "tags": ["python", "dictionary", "sorting"],
  "question": "I have a dict and I want to sort it by values...",
  "accepted_answer": "You can use sorted()...",
  "answers": ["Another approach...", "Or try this..."]
}
```

**Where to place:**

```bash
/media/wade/gork/training_datasets/misc_datasets/stackexchange_python/
```

### 3. ArXiv ML Abstracts

**What it is:**
- Scientific paper abstracts from ArXiv
- Filtered for Machine Learning / Computer Science categories
- Demonstrates technical language understanding

**Where to get it:**

```bash
# Option 1: ArXiv bulk data
# Visit: https://arxiv.org/help/bulk_data

# Option 2: Use ArXiv API
# https://arxiv.org/help/api/index

# Example: Download CS.LG (Machine Learning) category
# Filter for categories: cs.LG, cs.AI, cs.CV, stat.ML

# Option 3: Kaggle ArXiv dataset
# https://www.kaggle.com/datasets/Cornell-University/arxiv
```

**Expected format (JSONL):**

```json
{
  "id": "2103.12345",
  "title": "A Novel Approach to...",
  "abstract": "We propose a new method...",
  "categories": ["cs.LG", "cs.AI"],
  "authors": ["Author One", "Author Two"]
}
```

**Where to place:**

```bash
/media/wade/gork/training_datasets/misc_datasets/arxiv_ml_abstracts/
```

## Usage

### Quick Start - Test All Datasets

```bash
# Run dry-run tests for all three datasets
cd /home/wade/CLAUDE
./RUN_PORTFOLIO_ALL_DRYRUN.sh
```

This will scan all datasets and estimate:
- Total rows found
- Expected chunks
- Expected vectors
- No GPU usage, very fast

### Individual Dataset Testing

```bash
# Wikipedia Featured Articles
./RUN_PORTFOLIO_WIKI_DRYRUN.sh

# StackExchange Python Q&A
./RUN_PORTFOLIO_STACKEXCHANGE_DRYRUN.sh

# ArXiv ML Abstracts
./RUN_PORTFOLIO_ARXIV_DRYRUN.sh
```

### Production Runs

**Run all three datasets:**

```bash
./RUN_PORTFOLIO_ALL_PRODUCTION.sh
```

**Run individually:**

```bash
# Wikipedia (recommended first - most recognizable)
./RUN_PORTFOLIO_WIKI_PRODUCTION.sh

# StackExchange (real application scenario)
./RUN_PORTFOLIO_STACKEXCHANGE_PRODUCTION.sh

# ArXiv (high technical value)
./RUN_PORTFOLIO_ARXIV_PRODUCTION.sh
```

## Output Structure

After successful indexing, each dataset will have:

```
/media/wade/gork/indexed/wiki_featured/
├── vectors.index       # FAISS IndexFlatIP
├── chunks.json         # Text chunks
├── metadata.jsonl      # Metadata for each vector
└── summary.json        # Index statistics

/media/wade/gork/indexed/stackexchange_python/
├── vectors.index
├── chunks.json
├── metadata.jsonl
└── summary.json

/media/wade/gork/indexed/arxiv_ml_abstracts/
├── vectors.index
├── chunks.json
├── metadata.jsonl
└── summary.json
```

## Technical Specifications

All scripts implement the Universal Protocol:

### Core Features
- **Producer/Consumer Architecture**: Streaming pipeline with queue-based coordination
- **RAM Balancer**: Automatic pause at 90%, resume at 70%
- **GPU Balancer**: OOM handling with batch size reduction
- **Batch Size**: Locked at 1300 (proven stable)
- **Embedding Model**: intfloat/e5-large-v2 (fp16)
- **Chunking**: 1500 characters, word-aligned
- **Index Type**: FAISS IndexFlatIP (1024 dimensions)
- **Sharding**: 200K vectors per shard
- **Checkpointing**: Every 1M vectors
- **Signal Handling**: Graceful shutdown on SIGINT/SIGTERM

### Memory Footprint
- **VRAM per indexer**: 1-3GB total (including model)
- **RAM requirement**: 12GB free at startup
- **Parallelism**: Can run 2-4 indexers simultaneously

### Data Integrity
- Atomic file writes using `.tmp` swap
- Integrity checks: `len(vectors) == len(chunks) == len(metadata)`
- Progress tracking with resumable checkpoints
- Full logging with timestamps

## Why These Datasets?

### Portfolio Value

1. **Domain Diversity**: Encyclopedia (Wikipedia) + Technical Q&A (Stack Exchange) + Academic (ArXiv)
2. **Real-World Recognition**: All three are immediately recognizable to clients
3. **Use Case Coverage**:
   - Wikipedia → general knowledge retrieval
   - StackExchange → technical support / developer tools
   - ArXiv → research / scientific applications
4. **Production-Ready**: Demonstrate enterprise-grade indexing capabilities

### Client Impression

- **Wikipedia**: "Can handle large, varied, real-world text"
- **StackExchange**: "Understands technical Q&A and code discussions"
- **ArXiv**: "Can process complex scientific language"

All three together show **multi-domain expertise** with **production-quality infrastructure**.

## Troubleshooting

### Dataset Not Found

```bash
# Check if dataset directories exist
ls /media/wade/gork/training_datasets/misc_datasets/
```

Create missing directories and populate with data following the "Dataset Preparation" section above.

### Out of Memory

The scripts are designed to handle OOM automatically:
- GPU OOM: Batch size reduces automatically
- RAM pressure: Producers pause automatically

If problems persist, check:
```bash
# GPU memory
nvidia-smi

# System RAM
free -h

# Disk space on NVMe
df -h /home/wade/nvme_work/
```

### Failed Integrity Check

If `vectors != chunks != metadata`, the script will abort to prevent corruption.

Common causes:
- Disk full during writing
- Process killed without graceful shutdown
- Corrupted source data

Solution:
```bash
# Clear work directory and restart
rm -rf /home/wade/nvme_work/[dataset_name]
./RUN_PORTFOLIO_[DATASET]_PRODUCTION.sh
```

## Advanced Usage

### Parallel Indexing

You can run multiple datasets in parallel if you have VRAM headroom:

```bash
# Terminal 1
./RUN_PORTFOLIO_WIKI_PRODUCTION.sh &

# Terminal 2
./RUN_PORTFOLIO_STACKEXCHANGE_PRODUCTION.sh &

# Monitor VRAM
watch -n 1 nvidia-smi
```

**Rule**: Ensure `free_vram >= (#indexers × 3GB) + 4GB buffer`

### Resuming After Interruption

All scripts support graceful shutdown (Ctrl+C). To resume:

```bash
# Just run the same production script again
# The script will detect existing work and continue from checkpoint
./RUN_PORTFOLIO_WIKI_PRODUCTION.sh
```

## Files Created

### Scripts
- `index_wiki_featured.py` - Wikipedia indexer
- `index_stackexchange_python.py` - StackExchange indexer
- `index_arxiv_ml_abstracts.py` - ArXiv indexer

### Wrapper Scripts
- `RUN_PORTFOLIO_WIKI_DRYRUN.sh` / `RUN_PORTFOLIO_WIKI_PRODUCTION.sh`
- `RUN_PORTFOLIO_STACKEXCHANGE_DRYRUN.sh` / `RUN_PORTFOLIO_STACKEXCHANGE_PRODUCTION.sh`
- `RUN_PORTFOLIO_ARXIV_DRYRUN.sh` / `RUN_PORTFOLIO_ARXIV_PRODUCTION.sh`

### Master Scripts
- `RUN_PORTFOLIO_ALL_DRYRUN.sh` - Test all three datasets
- `RUN_PORTFOLIO_ALL_PRODUCTION.sh` - Index all three datasets

## Next Steps

1. **Prepare datasets**: Follow "Dataset Preparation" section to download and format data
2. **Test with dry-run**: Run `./RUN_PORTFOLIO_ALL_DRYRUN.sh`
3. **Review estimates**: Check estimated vector counts
4. **Run production**: Execute `./RUN_PORTFOLIO_ALL_PRODUCTION.sh`
5. **Verify outputs**: Check `/media/wade/gork/indexed/` for completed indexes

## Support

For issues or questions, refer to:
- `/home/wade/CLAUDE/__UNIVERSAL_PROTOCOL_CORE__/The Universal Batch Indexing & Verification Engine.MD`
- `/home/wade/CLAUDE/__UNIVERSAL_PROTOCOL_CORE__/EXAMPLE SCRIPT.MD`
