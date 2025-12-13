#!/bin/bash
# Master script to prepare all three portfolio datasets

set -e

echo "========================================="
echo "PORTFOLIO DATASETS - PREPARATION"
echo "========================================="
echo ""
echo "This will prepare all three datasets:"
echo "  1. StackExchange Python Q&A (symlink existing parquet files)"
echo "  2. ArXiv ML Abstracts (symlink RedPajama arxiv files)"
echo "  3. Wikipedia Articles (convert wiki shards to JSONL)"
echo ""
echo "Starting in 3 seconds..."
sleep 3

echo ""
echo "========================================="
echo "DATASET 1/3: StackExchange Python Q&A"
echo "========================================="
./prepare_stackexchange_python.sh

echo ""
echo "========================================="
echo "DATASET 2/3: ArXiv ML Abstracts"
echo "========================================="
./prepare_arxiv_ml.sh

echo ""
echo "========================================="
echo "DATASET 3/3: Wikipedia Articles"
echo "========================================="
./prepare_wiki_featured.sh

echo ""
echo "========================================="
echo "ALL DATASETS PREPARED"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Test with dry-run: ./RUN_PORTFOLIO_ALL_DRYRUN.sh"
echo "  2. Run production: ./RUN_PORTFOLIO_ALL_PRODUCTION.sh"
echo ""
echo "Dataset locations:"
echo "  /media/wade/gork/training_datasets/misc_datasets/stackexchange_python/"
echo "  /media/wade/gork/training_datasets/misc_datasets/arxiv_ml_abstracts/"
echo "  /media/wade/gork/training_datasets/misc_datasets/wiki_featured/"
