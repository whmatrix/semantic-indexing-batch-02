#!/bin/bash
# Master production script for all three portfolio datasets
# Runs full indexing with GPU embedding for all datasets

set -e

echo "========================================="
echo "PORTFOLIO DATASETS - ALL PRODUCTION RUNS"
echo "========================================="
echo ""
echo "This will index 3 datasets with full GPU embedding:"
echo "  1. Wikipedia Featured Articles"
echo "  2. StackExchange Python Q&A"
echo "  3. ArXiv ML Abstracts"
echo ""
echo "WARNING: This will take significant time and GPU resources"
echo "Press Ctrl+C within 10 seconds to cancel..."
sleep 10

echo ""
echo "========================================="
echo "DATASET 1/3: Wikipedia Featured Articles"
echo "========================================="
python3 ./scripts/indexers/index_wiki_featured.py 2>&1 | tee /tmp/wiki_featured_production.log

echo ""
echo "========================================="
echo "DATASET 2/3: StackExchange Python Q&A"
echo "========================================="
python3 ./scripts/indexers/index_stackexchange_python.py 2>&1 | tee /tmp/stackexchange_python_production.log

echo ""
echo "========================================="
echo "DATASET 3/3: ArXiv ML Abstracts"
echo "========================================="
python3 ./scripts/indexers/index_arxiv_ml_abstracts.py 2>&1 | tee /tmp/arxiv_ml_production.log

echo ""
echo "========================================="
echo "ALL PRODUCTION RUNS COMPLETE"
echo "========================================="
echo ""
echo "Output indexes:"
echo "  ./results/indexes/wiki_featured"
echo "  ./results/indexes/stackexchange_python"
echo "  ./results/indexes/arxiv_ml_abstracts"
echo ""
echo "Logs saved to:"
echo "  /tmp/wiki_featured_production.log"
echo "  /tmp/stackexchange_python_production.log"
echo "  /tmp/arxiv_ml_production.log"
