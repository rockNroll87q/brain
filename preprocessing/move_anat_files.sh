#!/usr/bin/env bash

# This script automatically finds anat files based on an identified and moves the files from
# DATA_FOLDER/raw to DATA_FOLDER/imgs/anat as per data folder structure. 

# Additional functionality of the script:
# - Track the movement of the files from src -> dest and store 
#   a record of the two paths in a file_movement_tracking.csv file saved in DATA_FOLDER/raw.
# - If each subject has a unique folder but the same filename "anat_T1w.nii.gz", the script
#   will append the subject folder name as prefix to the moved filename e.g.

# ├── DATA_FOLDER/raw
# │   ├── 28326_1_MR
# │   │   └── anat
# │   │       └── NIfTI-1
# │   │           └── anat.nii.gz
# │   ├── 28327_1_MR
# │   │   └── anat
# │   │       └── NIfTI-1
# │   │           └── anat.nii.gz
# │   ├── 28328_1_MR
# │   │   └── anat
# │   │       └── NIfTI-1
# │   │           └── anat.nii.gz
# │   ├── 28329_1_MR
# │   │   └── anat
# │   │       └── NIfTI-1
# │   │           └── anat.nii.gz


# It will create filenames 28326_1_MR_anat.nii.gz, 28327_1_MR_anat.nii.gz, 28328_1_MR_anat.nii.gz etc
# in DATA_FOLDER/imgs/anat

# Usage: find_and_move_anat.sh <DATA_FOLDER> <IDENTIFIER>
#./move_anat_files.sh /analyse/Project0406/TheOneSeg/data/dataset_name/ .nii.gz

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <DATA_FOLDER> <IDENTIFIER>"
  exit 1
fi

# Variables
DATA_DIR="$1"                               # Base data folder
ID="$2"                                     # Identifier suffix (e.g. '_T1w.nii.gz')
SRC="${DATA_DIR%/}/raw"                     # Hard‑coded raw subdir
DEST="${DATA_DIR%/}/imgs/anat"              # Hard‑coded anat subdir
LOGFILE="$SRC/watcher.log"                  # Watcher log file
CSV="$SRC/file_movement_tracking.csv"       # Movement CSV

mkdir -p "$DEST"                            # Create DEST if needed

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  # This script’s folder

# 1) Start watcher
echo "Starting file‑move watcher… (logs in $LOGFILE)"
python3 "$SCRIPT_DIR/track_and_map_files.py" \
  --watch-dir "$DATA_DIR" \
  --logdir    "$SRC" \
  >>"$LOGFILE" 2>&1 &
WATCHER_PID=$!

sleep 1                                     # Give watcher a moment
if ! ps -p $WATCHER_PID >/dev/null; then    # Check it really started
  echo "Watcher failed to start. See $LOGFILE."
  exit 1
fi

# 2) Gather matching files
mapfile -d '' paths < <(find "$SRC" -type f -name "*${ID}" -print0)
total=${#paths[@]}

if [ "$total" -eq 0 ]; then                 # If none, stop watcher & exit
  echo "No files matching '*${ID}' in $SRC."
  kill -TERM $WATCHER_PID
  exit 0
fi

echo "Found $total file(s)."

# 3) Detect duplicate basenames
declare -A name_count
for path in "${paths[@]}"; do
  base=$(basename "$path")
  name_count["$base"]=$((name_count["$base"]+1))
 done

# 4) Helper to find unique directory prefix
find_unique_prefix() {
  local target="$1"  # full path
  local dir=$(dirname "$target")
  while [[ "$dir" != "/" && "$dir" != "." ]]; do
    local dname=$(basename "$dir")
    local count=0
    for f in "${paths[@]}"; do
      if [[ "$f" != "$target" && "$f" == */"${dname}"/* ]]; then
        count=$((count+1))
      fi
    done
    if [[ "$count" -eq 0 ]]; then
      echo "$dname"
      return
    fi
    dir=$(dirname "$dir")
  done
  echo ""
}

# 5) Build new_name_map and validate prefix for duplicates
declare -A new_name_map
for path in "${paths[@]}"; do
  base=$(basename "$path")
  if [ "${name_count["$base"]}" -gt 1 ]; then
    prefix=$(find_unique_prefix "$path")
    if [ -z "$prefix" ]; then
      echo "Error: could not determine unique prefix for duplicate filename '$base'. Aborting."
      kill -TERM $WATCHER_PID
      exit 1
    fi
    new_name_map["$path"]="${prefix}_${base}"
  else
    new_name_map["$path"]="$base"
  fi
done

# 6) Move files with progress
count=0
for path in "${paths[@]}"; do
  count=$((count+1))
  newname="${new_name_map[$path]}"
  echo "[$count/$total] Moving: $path → $DEST/$newname"
  mv "$path" "$DEST/$newname"
done

# 7) Terminate watcher
echo "All files moved — terminating watcher (PID $WATCHER_PID)…"
kill -TERM $WATCHER_PID
for i in {1..50}; do
  if ! kill -0 $WATCHER_PID 2>/dev/null; then break; fi
  sleep 0.1
done
if kill -0 $WATCHER_PID 2>/dev/null; then
  echo "Watcher didn’t exit; sending SIGKILL."
  kill -KILL $WATCHER_PID
fi

# 8) Final CSV check
echo ""
if [ -f "$CSV" ]; then
  echo "Tracking CSV created at: $CSV"
else
  echo "No CSV found at $CSV. See $LOGFILE for errors."
fi

echo "Done! Moved $count file(s) to '$DEST'."
