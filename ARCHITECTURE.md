# Architecture: Semantic Indexing Batch 02

This document describes the architecture for indexing 8.35M vectors across three datasets using e5-large-v2 embeddings and FAISS.

## Data Processing Pipeline

```mermaid
graph LR
    A[Acquire Data] --> B[Chunk Text]
    B --> C[Embed with e5-large-v2]
    C --> D[Index via FAISS IndexFlatIP]
    D --> E[Store to Disk]
    E --> F[Query Interface]

    subgraph Config
        G[batch_size=1300]
        H[checkpoint every 1M vectors]
        I[1024-dim FP16]
    end
```

## Query Pipeline

```mermaid
graph LR
    Q[User Query] --> P["Prefix: query:"]
    P --> E[e5-large-v2 Encode]
    E --> S[FAISS Inner Product Search]
    S --> R[Ranked Results]
    R --> M[Metadata Lookup]
    M --> O[Return Top-K with Sources]
```

## Split-Merge Pattern for StackExchange

StackExchange Python contains 7.5M vectors and is processed using a split-merge pattern with 5 parallel indexers.

```mermaid
graph TD
    SRC[StackExchange Python 7.5M] --> SP[Split into 5 Shards]
    SP --> I1[Indexer 1 - 1.5M]
    SP --> I2[Indexer 2 - 1.5M]
    SP --> I3[Indexer 3 - 1.5M]
    SP --> I4[Indexer 4 - 1.5M]
    SP --> I5[Indexer 5 - 1.5M]
    I1 --> MG[Merge into Single Index]
    I2 --> MG
    I3 --> MG
    I4 --> MG
    I5 --> MG
    MG --> FI[Final StackExchange Index]
```

## Component Responsibilities

| Component | Responsibility |
|---|---|
| `acquire` | Download and validate raw datasets (Wikipedia FA, StackExchange Python, ArXiv ML) |
| `chunk` | Split documents into embedding-ready text segments |
| `embed` | Encode chunks using e5-large-v2 (1024-dim, FP16) with `passage:` prefix |
| `index` | Build FAISS IndexFlatIP with checkpointing every 1M vectors |
| `store` | Persist index files and metadata mappings to disk |
| `query` | Accept user queries with `query:` prefix, return ranked results via inner product search |
| `split-merge` | Parallelize StackExchange indexing across 5 workers, then merge into a single index |
| `checkpoint` | Save intermediate state every 1M vectors for fault tolerance |

## Scale Reference

| Dataset | Vectors | Notes |
|---|---|---|
| Wikipedia Featured Articles | 352K | High-quality encyclopedic content |
| StackExchange Python | 7.5M | Split-merge across 5 parallel indexers |
| ArXiv ML Abstracts | 489K | Machine learning paper abstracts |
| **Total** | **8.35M** | 1024-dim FP16, FAISS IndexFlatIP |

## Embedding Configuration

- **Model**: e5-large-v2
- **Dimensions**: 1024
- **Precision**: FP16
- **Batch size**: 1,300
- **Index type**: FAISS IndexFlatIP (inner product / cosine similarity on normalized vectors)
- **Checkpointing**: Every 1,000,000 vectors
