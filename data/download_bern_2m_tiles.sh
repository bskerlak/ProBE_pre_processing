#!/bin/bash

# Configuration
CSV_FILE="ch.swisstopo.swissalti3d-grosses-rechteck.csv"
TARGET_DIR="bern_2m_tiles"

# Create directory if it doesn't exist
mkdir -p "$TARGET_DIR"

echo "🚀 Starting synchronized download..."

# 1. Read the CSV (first column only)
# 2. Extract the filename from the URL
# 3. Check if that filename already exists in bern_2m_tiles
cut -d ',' -f 1 "$CSV_FILE" | while read -r url; do
    filename=$(basename "$url")
    
    if [ -f "$TARGET_DIR/$filename" ]; then
        echo "✅ Skipping: $filename (Already exists)"
    else
        echo "📥 Downloading: $filename ..."
        # -L follows redirects, -C - resumes partials, -o saves to target dir
        curl -L -C - "$url" -o "$TARGET_DIR/$filename"
    fi
done

echo "✨ All files synchronized in $TARGET_DIR"