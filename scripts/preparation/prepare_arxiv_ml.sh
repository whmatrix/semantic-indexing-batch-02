#!/bin/bash
# Prepare ArXiv ML dataset by symlinking RedPajama arxiv files

set -e

SOURCE_DIR="/media/wade/gork/text_datasets/redpajama"
TARGET_DIR="/media/wade/gork/training_datasets/misc_datasets/arxiv_ml_abstracts"

echo "========================================="
echo "Preparing ArXiv ML Abstracts Dataset"
echo "========================================="
echo ""

# Check if source exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "ERROR: Source directory not found: $SOURCE_DIR"
    exit 1
fi

# Create target directory
mkdir -p "$TARGET_DIR"

# Create symlinks to arxiv JSONL files
echo "Creating symlinks to ArXiv JSONL files..."
for file in "$SOURCE_DIR"/arxiv_*.jsonl; do
    if [ -f "$file" ]; then
        ln -sf "$file" "$TARGET_DIR/"
    fi
done

# Count files
FILECOUNT=$(ls -1 "$TARGET_DIR"/*.jsonl 2>/dev/null | wc -l)

echo ""
echo "========================================="
echo "PREPARATION COMPLETE"
echo "========================================="
echo "Source: $SOURCE_DIR"
echo "Target: $TARGET_DIR"
echo "Files linked: $FILECOUNT JSONL files"
echo ""
echo "These are ArXiv papers from the RedPajama dataset."
echo "They include papers from multiple categories including ML, CS, etc."
echo ""
echo "Ready to run:"
echo "  ./RUN_PORTFOLIO_ARXIV_DRYRUN.sh"
