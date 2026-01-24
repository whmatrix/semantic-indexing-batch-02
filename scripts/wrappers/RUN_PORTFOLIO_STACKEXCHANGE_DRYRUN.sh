#!/bin/bash
# Dry-run for StackExchange Python Q&A indexing
# Tests the indexing pipeline without loading the model or embedding

set -e

echo "========================================="
echo "StackExchange Python Q&A - DRY RUN"
echo "========================================="
echo ""

python3 ./scripts/indexers/index_stackexchange_python.py --dry-run

echo ""
echo "DRY RUN COMPLETE"
echo "To run production: ./RUN_PORTFOLIO_STACKEXCHANGE_PRODUCTION.sh"
