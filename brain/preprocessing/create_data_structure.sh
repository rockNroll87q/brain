#!/usr/bin/env bash
#
# create_dirs.sh
# Usage: ./create_dirs.sh [START_DIR]
#
# Creates the following structure under START_DIR (or cwd if none provided),
# with group 'segment' and permissions rwx for user+group ,
# and setgid so new files/dirs inherit the 'segment' group:
#
# ├── src
# ├── imgs
# │   ├── anat
# │   └── derivative
# ├── non-imgs
# │   ├── derivative
# │   └── raw
# └── raw
# └── csv

set -euo pipefail

usage() {
  cat <<EOF
Usage: $(basename "$0") [START_DIR]
  START_DIR  Directory under which to create the folder tree.
             If omitted, uses the current directory.
EOF
  exit 1
}

# Parse args
if [[ "${1:-}" =~ ^(-h|--help)$ ]]; then
  usage
elif [[ $# -gt 1 ]]; then
  echo "Error: Too many arguments." >&2
  usage
fi

START_DIR="${1:-$(pwd)}"

# Ensure START_DIR exists
if [[ ! -d "$START_DIR" ]]; then
  echo "Error: '$START_DIR' is not a directory." >&2
  exit 1
fi

# Ensure group exists
if ! getent group segment >/dev/null; then
  echo "Error: group 'segment' does not exist on this system." >&2
  exit 1
fi

# Relative directories to create
dirs=(
  src
  imgs/anat
  imgs/derivative
  non-imgs/derivative
  non-imgs/raw
  raw
  csv
)

# Create with correct mode and group
for d in "${dirs[@]}"; do
  fullpath="$START_DIR/$d"
  # -d: create directories, -m 2770 sets u=rwx, g=rwx + setgid; o=—
  install -d -m 2770 -g segment "$fullpath"
done

echo "Directory tree created under '$START_DIR' with group=segment and rwx for user+group."
