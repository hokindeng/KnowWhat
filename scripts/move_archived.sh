#!/bin/bash

# Check if correct number of arguments was provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 /path/to/filtered_results_folder"
    echo "This script consolidates all participant archived folders into a central archive directory."
    echo "It preserves participant IDs by creating subdirectories within the central archive."
    exit 1
fi

RESULTS_DIR="$1"

# Check if results directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    echo "Error: Results directory '$RESULTS_DIR' does not exist!"
    exit 1
fi

# Create central archive directory
CENTRAL_ARCHIVE="$RESULTS_DIR/archived"
mkdir -p "$CENTRAL_ARCHIVE"

echo "Consolidating archived files from participant folders in $RESULTS_DIR..."
echo "Moving to central archive at $CENTRAL_ARCHIVE"

# Initialize counters
total_participants=0
total_files_moved=0

# Find all participant directories that have an 'archived' subfolder
for participant_dir in $(find "$RESULTS_DIR" -mindepth 1 -maxdepth 1 -type d); do
    participant=$(basename "$participant_dir")
    archived_dir="$participant_dir/archived"
    
    # Skip the central archive directory itself
    if [ "$participant" = "archived" ]; then
        continue
    fi
    
    # Check if this participant has an archived folder
    if [ -d "$archived_dir" ]; then
        echo "Processing archived files for participant: $participant"
        
        # Create participant subfolder in central archive
        participant_archive="$CENTRAL_ARCHIVE/$participant"
        mkdir -p "$participant_archive"
        
        # Count files to move
        file_count=$(find "$archived_dir" -type f | wc -l)
        
        if [ "$file_count" -gt 0 ]; then
            # Move all files from participant's archived folder to central archive
            echo "  Moving $file_count files to central archive"
            mv "$archived_dir"/* "$participant_archive/"
            
            # Update counters
            total_participants=$((total_participants + 1))
            total_files_moved=$((total_files_moved + file_count))
            
            # Remove the now-empty archived folder
            rmdir "$archived_dir"
        else
            echo "  No files found in archived folder"
            # Remove empty archived directory
            rmdir "$archived_dir"
        fi
    fi
done

echo "Summary:"
echo "- Processed archives for $total_participants participants"
echo "- Moved $total_files_moved files to central archive"
echo "Done. All archived files have been consolidated in $CENTRAL_ARCHIVE" 