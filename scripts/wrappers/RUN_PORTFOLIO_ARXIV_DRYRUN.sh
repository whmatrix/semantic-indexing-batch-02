#!/bin/bash
# Dry-run for ArXiv ML Abstracts indexing
# Tests the indexing pipeline without loading the model or embedding

set -e

echo "========================================="
echo "ArXiv ML Abstracts - DRY RUN"
echo "========================================="
echo ""

python3 ./scripts/indexers/index_arxiv_ml_abstracts.py --dry-run

echo ""
echo "DRY RUN COMPLETE"
echo "To run production: ./RUN_PORTFOLIO_ARXIV_PRODUCTION.sh"
