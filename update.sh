#!/bin/bash
# Moltbook incremental archive update
# Usage: ./update.sh [--full]

cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:/opt/homebrew/bin:$PATH"
source ~/.zshenv 2>/dev/null  # Load AWS creds

echo "=== Update started: $(date) ==="

if [ "$1" == "--full" ]; then
    echo "Running full archive..."
    uv run python scraper/fetch_all_posts.py
    uv run python scraper/fetch_all_comments.py
else
    echo "Running incremental update..."
    uv run python scraper/incremental_update.py
fi

echo ""
echo "Archive: https://moltbook-archive-319933937176.s3.us-east-2.amazonaws.com/data/"
