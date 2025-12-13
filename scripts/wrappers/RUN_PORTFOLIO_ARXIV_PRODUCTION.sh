#!/bin/bash
# Production run for ArXiv ML Abstracts indexing
# Full embedding and FAISS index creation

set -e

echo "========================================="
echo "ArXiv ML Abstracts - PRODUCTION"
echo "========================================="
echo ""
echo "WARNING: This will run full indexing with GPU embedding"
echo "Press Ctrl+C within 5 seconds to cancel..."
sleep 5

python3 /home/wade/CLAUDE/index_arxiv_ml_abstracts.py 2>&1 | tee /tmp/arxiv_ml_production.log

echo ""
echo "PRODUCTION RUN COMPLETE"
echo "Output index: /media/wade/gork/indexed/arxiv_ml_abstracts"
echo "Log saved to: /tmp/arxiv_ml_production.log"
