#!/bin/bash
# Production run for StackExchange Python Q&A indexing
# Full embedding and FAISS index creation

set -e

echo "========================================="
echo "StackExchange Python Q&A - PRODUCTION"
echo "========================================="
echo ""
echo "WARNING: This will run full indexing with GPU embedding"
echo "Press Ctrl+C within 5 seconds to cancel..."
sleep 5

python3 ./scripts/indexers/index_stackexchange_python.py 2>&1 | tee /tmp/stackexchange_python_production.log

echo ""
echo "PRODUCTION RUN COMPLETE"
echo "Output index: ./results/indexes/stackexchange_python"
echo "Log saved to: /tmp/stackexchange_python_production.log"
