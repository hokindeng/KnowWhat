#!/bin/bash

# Check if a directory path was provided
if [ $# -eq 1 ]; then
    TARGET_DIR="$1"
    # Check if the provided directory exists
    if [ ! -d "$TARGET_DIR" ]; then
        echo "Error: Directory '$TARGET_DIR' does not exist!"
        exit 1
    fi
else
    echo "Usage: $0 path/to/directory"
    echo "This script flattens a directory by moving all files to the top level."
    exit 1
fi

# Change to the target directory
cd "$TARGET_DIR" || { echo "Failed to change to directory '$TARGET_DIR'"; exit 1; }

# Find all files recursively and move them to the current directory
find . -type f -not -path "./.*" | while read -r file; do
    # Get just the filename without the path
    filename=$(basename "$file")
    
    # Handle filename conflicts by adding a unique identifier
    if [ -f "./$filename" ] && [ "$file" != "./$filename" ]; then
        extension="${filename##*.}"
        basename="${filename%.*}"
        filename="${basename}_$(date +%s).$extension"
    fi
    
    # Move the file if it's not already in the current directory
    if [ "$file" != "./$filename" ]; then
        mv "$file" "./$filename"
    fi
done

# Remove all empty directories
find . -type d -not -path "." -delete

echo "Directory '$TARGET_DIR' has been flattened successfully."