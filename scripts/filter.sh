#!/bin/bash

# Check if correct number of arguments was provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 /path/to/source_json_folder /path/to/output_folder"
    echo "This script analyzes task completion from JSON files in the source folder,"
    echo "generates a report, and moves files for the most complete participant (or all complete ones)"
    echo "to the output folder."
    exit 1
fi

SOURCE_DIR="$1"
TARGET_DIR="$2"

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory '$SOURCE_DIR' does not exist!"
    exit 1
fi

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Create an array of all required shape types
shapes=("C" "Z" "cross" "spiral" "square" "triangle")
grid_sizes=("5x5" "7x7")
# Total number of unique task types (shape x grid_size)
total_task_types=$((${#shapes[@]} * ${#grid_sizes[@]})) # Should be 12
# Required files per task type
files_per_task_type=1
required_total_files=$((total_task_types * files_per_task_type)) # Should be 12

# Extract unique participant IDs from filenames in the source directory
# Uses basename to get filename, then awk to get ID before first '_'
participants=$(find "$SOURCE_DIR" -maxdepth 1 -name "*.json" -type f -exec basename {} \; | awk -F'_' '{print $1}' | sort -u)

if [ -z "$participants" ]; then
    echo "No .json files found in $SOURCE_DIR. Exiting."
    exit 0
fi

echo "Analyzing participant task completion from $SOURCE_DIR..."
echo "Output will be in $TARGET_DIR"

# Initialize report file
report_file="$TARGET_DIR/completion_report.txt"
echo "TASK COMPLETION REPORT" > "$report_file"
echo "Source Directory: $SOURCE_DIR" >> "$report_file"
echo "Generated: $(date)" >> "$report_file"
echo "=====================================" >> "$report_file"
echo "" >> "$report_file"

highest_task_types_completed=0 # Track completion of task *types* (e.g., C_5x5)
highest_total_files=0       # Track total number of files

# --- New: Define completion threshold and list for moved participants ---
MIN_COMPLETION_PERCENTAGE=70
moved_participants_details=()
any_participant_met_threshold=false
# --- End New ---

for participant in $participants; do
    # Skip empty participant IDs just in case (should be filtered by awk/sort -u)
    if [ -z "$participant" ]; then
        continue
    fi

    echo "Analyzing $participant..."
    
    echo "Participant: $participant" >> "$report_file"
    echo "------------------------------------" >> "$report_file"
    
    task_types_done_by_participant=0
    task_matrix_report=""
    
    for shape in "${shapes[@]}"; do
        for size in "${grid_sizes[@]}"; do
            # Count files for this specific task type (participant_size_shape_*.json)
            count=$(find "$SOURCE_DIR" -maxdepth 1 -name "${participant}_${size}_${shape}_*.json" -type f | wc -l)
            task_matrix_report+="$shape ($size): $count/$files_per_task_type completed\\n"
            
            if [ "$count" -ge "$files_per_task_type" ]; then
                task_types_done_by_participant=$((task_types_done_by_participant + 1))
            fi
        done
    done
    
    # Count total files for this participant in the source directory
    current_participant_total_files=$(find "$SOURCE_DIR" -maxdepth 1 -name "${participant}_*.json" -type f | wc -l)
    
    completion_percentage=$((task_types_done_by_participant * 100 / total_task_types))
    
    echo -e "$task_matrix_report" >> "$report_file"
    echo "Task types completed: $task_types_done_by_participant/$total_task_types ($completion_percentage%)" >> "$report_file"
    echo "Total files found: $current_participant_total_files/$required_total_files" >> "$report_file" # Now shows against expected total
    echo "" >> "$report_file"
    
    # Update who is the "most complete"
    # Prioritize by task types completed, then by total files as a tie-breaker
    if [ "$task_types_done_by_participant" -gt "$highest_task_types_completed" ]; then
        highest_task_types_completed=$task_types_done_by_participant
        highest_total_files=$current_participant_total_files
        most_complete_participant=$participant
    elif [ "$task_types_done_by_participant" -eq "$highest_task_types_completed" ] && [ "$current_participant_total_files" -gt "$highest_total_files" ]; then
        highest_total_files=$current_participant_total_files
        most_complete_participant=$participant
    fi
    
    # --- Modified: Check against completion threshold for moving files ---
    if [ "$completion_percentage" -ge "$MIN_COMPLETION_PERCENTAGE" ]; then
        participant_output_dir="$TARGET_DIR/$participant"
        mkdir -p "$participant_output_dir"
        echo "✓ $participant - Met $MIN_COMPLETION_PERCENTAGE% threshold ($completion_percentage% completed). Moving files to $participant_output_dir..."
        # Move all files for this participant to their dedicated subdirectory
        find "$SOURCE_DIR" -maxdepth 1 -name "${participant}_*.json" -type f -exec mv {} "$participant_output_dir" \;
        any_participant_met_threshold=true
        moved_participants_details+=("$participant ($completion_percentage% task types, $current_participant_total_files files)")
    else
        echo "✗ $participant - Did not meet $MIN_COMPLETION_PERCENTAGE% threshold ($completion_percentage% completed). Files not moved based on this criterion."
    fi
    # --- End Modified ---
done

# Add summary section to report
echo "=====================================" >> "$report_file"
echo "SUMMARY" >> "$report_file"
echo "=====================================" >> "$report_file"

# --- Modified: Summary based on threshold ---
if [ "$any_participant_met_threshold" = true ]; then
    echo "Participants meeting the $MIN_COMPLETION_PERCENTAGE% task completion threshold had their files moved to individual subfolders in $TARGET_DIR:" >> "$report_file"
    for item in "${moved_participants_details[@]}"; do
        echo "- $item" >> "$report_file"
    done
else
    echo "No participants met the $MIN_COMPLETION_PERCENTAGE% task completion threshold. No files were moved based on this criterion." >> "$report_file"
    # Still report the most complete participant for informational purposes
    if [ -n "$most_complete_participant" ] && [ "$highest_task_types_completed" -gt 0 ]; then
        most_complete_percentage_info=$((highest_task_types_completed * 100 / total_task_types))
        echo "Overall most complete participant (files not moved): $most_complete_participant ($most_complete_percentage_info% task types, $highest_total_files files)." >> "$report_file"
    elif [ -n "$most_complete_participant" ]; then # Case where a participant exists but completed 0 task types
        echo "Overall most complete participant (files not moved): $most_complete_participant (0% task types, $highest_total_files files)." >> "$report_file"
    else
        echo "No valid participants found or no tasks completed by any participant." >> "$report_file"
    fi
fi
# --- End Modified ---

echo "Done. Report generated at $report_file" 