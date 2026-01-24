#!/bin/bash
# Dry-run for Wikipedia Featured Articles indexing
# Tests the indexing pipeline without loading the model or embedding

set -e

echo "========================================="
echo "Wikipedia Featured Articles - DRY RUN"
echo "========================================="
echo ""

python3 ./scripts/indexers/index_wiki_featured.py --dry-run

echo ""
echo "DRY RUN COMPLETE"
echo "To run production: ./RUN_PORTFOLIO_WIKI_PRODUCTION.sh"
