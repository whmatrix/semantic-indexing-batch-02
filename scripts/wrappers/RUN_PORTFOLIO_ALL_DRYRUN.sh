#!/bin/bash
# Master dry-run script for all three portfolio datasets
# Tests all indexing pipelines without embedding

set -e

echo "========================================="
echo "PORTFOLIO DATASETS - ALL DRY RUNS"
echo "========================================="
echo ""
echo "This will test 3 datasets:"
echo "  1. Wikipedia Featured Articles"
echo "  2. StackExchange Python Q&A"
echo "  3. ArXiv ML Abstracts"
echo ""
echo "Starting in 3 seconds..."
sleep 3

echo ""
echo "========================================="
echo "DATASET 1/3: Wikipedia Featured Articles"
echo "========================================="
./RUN_PORTFOLIO_WIKI_DRYRUN.sh

echo ""
echo "========================================="
echo "DATASET 2/3: StackExchange Python Q&A"
echo "========================================="
./RUN_PORTFOLIO_STACKEXCHANGE_DRYRUN.sh

echo ""
echo "========================================="
echo "DATASET 3/3: ArXiv ML Abstracts"
echo "========================================="
./RUN_PORTFOLIO_ARXIV_DRYRUN.sh

echo ""
echo "========================================="
echo "ALL DRY RUNS COMPLETE"
echo "========================================="
echo ""
echo "To run production for all datasets:"
echo "  ./RUN_PORTFOLIO_ALL_PRODUCTION.sh"
echo ""
echo "Or run individually:"
echo "  ./RUN_PORTFOLIO_WIKI_PRODUCTION.sh"
echo "  ./RUN_PORTFOLIO_STACKEXCHANGE_PRODUCTION.sh"
echo "  ./RUN_PORTFOLIO_ARXIV_PRODUCTION.sh"
