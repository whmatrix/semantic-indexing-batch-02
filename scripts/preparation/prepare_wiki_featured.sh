#!/bin/bash
# Prepare Wikipedia Featured Articles dataset by symlinking wiki shards

set -e

SOURCE_DIR="/media/wade/gork/text_datasets/wiki/wiki_shards"
TARGET_DIR="/media/wade/gork/training_datasets/misc_datasets/wiki_featured"

echo "========================================="
echo "Preparing Wikipedia Featured Articles"
echo "========================================="
echo ""

# Check if source exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "ERROR: Source directory not found: $SOURCE_DIR"
    exit 1
fi

# Create target directory
mkdir -p "$TARGET_DIR"

# Concatenate wiki shard files into JSONL files
# Using first few shard directories (AA-AE) for a good sample
echo "Concatenating wiki shards to JSONL..."

for shard_dir in AA AB AC AD AE; do
    shard_path="$SOURCE_DIR/$shard_dir"
    if [ ! -d "$shard_path" ]; then
        continue
    fi

    echo "Processing shard $shard_dir..."
    output_file="$TARGET_DIR/wikipedia_${shard_dir}.jsonl"

    # Concatenate all wiki_* files from this shard directory
    cat "$shard_path"/wiki_* > "$output_file"

    # Count lines
    line_count=$(wc -l < "$output_file")
    echo "  Created $output_file with $line_count articles"
done

# Total count
total_lines=$(cat "$TARGET_DIR"/*.jsonl | wc -l)

echo ""
echo "========================================="
echo "PREPARATION COMPLETE"
echo "========================================="
echo "Source: $SOURCE_DIR"
echo "Target: $TARGET_DIR"
echo "Total articles: $total_lines"
echo ""
echo "Wikipedia articles are ready in JSONL format."
echo ""
echo "Ready to run:"
echo "  ./RUN_PORTFOLIO_WIKI_DRYRUN.sh"
