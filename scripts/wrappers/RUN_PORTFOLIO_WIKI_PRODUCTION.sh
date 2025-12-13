#!/bin/bash
# Production run for Wikipedia Featured Articles indexing
# Full embedding and FAISS index creation

set -e

echo "========================================="
echo "Wikipedia Featured Articles - PRODUCTION"
echo "========================================="
echo ""
echo "WARNING: This will run full indexing with GPU embedding"
echo "Press Ctrl+C within 5 seconds to cancel..."
sleep 5

python3 /home/wade/CLAUDE/index_wiki_featured.py 2>&1 | tee /tmp/wiki_featured_production.log

echo ""
echo "PRODUCTION RUN COMPLETE"
echo "Output index: /media/wade/gork/indexed/wiki_featured"
echo "Log saved to: /tmp/wiki_featured_production.log"
