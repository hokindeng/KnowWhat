#!/bin/bash

# Check if correct number of arguments was provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 /path/to/filtered_results_folder"
    echo "This script cleans participant data folders by:"
    echo "1. Identifying duplicate trials (more than 1 file per shape/size combination)"
    echo "2. Keeping only the most recent file for each combination"
    echo "3. Moving duplicates to an 'archived' subfolder"
    exit 1
fi

RESULTS_DIR="$1"

# Check if results directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    echo "Error: Results directory '$RESULTS_DIR' does not exist!"
    exit 1
fi

# Create an array of all required shape types
shapes=("C" "Z" "cross" "spiral" "square" "triangle")
grid_sizes=("5x5" "7x7")

# Get participant directories (each subdirectory is a participant)
participant_dirs=$(find "$RESULTS_DIR" -mindepth 1 -maxdepth 1 -type d)

echo "Cleaning participant data folders in $RESULTS_DIR..."

for participant_dir in $participant_dirs; do
    participant=$(basename "$participant_dir")
    echo "Processing participant: $participant"
    
    # Create archive directory for this participant
    archive_dir="$participant_dir/archived"
    mkdir -p "$archive_dir"
    
    # Track how many files were archived
    archived_count=0
    
    # Process each shape/size combination
    for shape in "${shapes[@]}"; do
        for size in "${grid_sizes[@]}"; do
            # Find all files for this specific combination
            files=$(find "$participant_dir" -maxdepth 1 -name "${participant}_${size}_${shape}_*.json" | sort)
            file_count=$(echo "$files" | wc -l)
            
            # Skip if no files or just one file
            if [ "$file_count" -le 1 ]; then
                continue
            fi
            
            echo "  Found $file_count files for $shape ($size)"
            
            # Keep the most recent file (last in the sorted list)
            latest_file=$(echo "$files" | tail -n 1)
            
            # Move all other files to the archive directory
            for file in $files; do
                if [ "$file" != "$latest_file" ]; then
                    echo "    Moving $(basename "$file") to archive"
                    mv "$file" "$archive_dir/"
                    archived_count=$((archived_count + 1))
                else
                    echo "    Keeping $(basename "$file")"
                fi
            done
        done
    done
    
    # Report results
    if [ "$archived_count" -gt 0 ]; then
        echo "  Archived $archived_count duplicate files for $participant"
    else
        # If no files were archived, remove the empty archive directory
        rmdir "$archive_dir" 2>/dev/null
        echo "  No duplicates found for $participant"
    fi
done

echo "Done. Cleaned up duplicate trial files." 