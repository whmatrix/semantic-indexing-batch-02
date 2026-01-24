#!/bin/bash
# Prepare StackExchange Python dataset by symlinking existing data

set -e

# NOTE: Update these paths for your local environment
SOURCE_DIR="./source_data/stackexchange"  # Local source data
TARGET_DIR="./datasets/stackexchange_python"

echo "========================================="
echo "Preparing StackExchange Python Dataset"
echo "========================================="
echo ""

# Check if source exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "ERROR: Source directory not found: $SOURCE_DIR"
    exit 1
fi

# Create target directory
mkdir -p "$TARGET_DIR"

# Create symlinks to all parquet files
echo "Creating symlinks to parquet files..."
ln -sf "$SOURCE_DIR"/*.parquet "$TARGET_DIR/" 2>/dev/null || true

# Count files
FILECOUNT=$(ls -1 "$TARGET_DIR"/*.parquet 2>/dev/null | wc -l)

echo ""
echo "========================================="
echo "PREPARATION COMPLETE"
echo "========================================="
echo "Source: $SOURCE_DIR"
echo "Target: $TARGET_DIR"
echo "Files linked: $FILECOUNT parquet files"
echo ""
echo "The existing StackExchange dataset contains Q&A from multiple sites."
echo "The indexer will process all Q&A pairs - you can filter for Python"
echo "content during querying or modify the indexer to pre-filter."
echo ""
echo "Ready to run:"
echo "  ./RUN_PORTFOLIO_STACKEXCHANGE_DRYRUN.sh"
