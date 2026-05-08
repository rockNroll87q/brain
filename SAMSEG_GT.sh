#!/bin/bash

# !!! This is a CPU intensive script !!!

# Script that generates:
# - SAMSEG segmentation (with --lesion flag)
# - Saves seg.mgz into the subject's derivative directory
#   i.e. DERIV_DIR/<t1_filename>/seg.mgz
# - Deletes all other SAMSEG output

# Iterates over all .nii.gz files in T1W_DIR
# To test on a few subjects first, set N_TEST to a small number (e.g. 3),
# then set to 0 to run all.

# To run:
# /usr/bin/bash
# $ bash ./SAMSEG_GT.sh

# ── Paths ─────────────────────────────────────────────────────────────────────
T1W_DIR="/analyse/Project0406/TheOneSeg/data/TheOneData/OpenMind/imgs/anat/T1w/"
export DERIV_DIR="/analyse/Project0406/TheOneSeg/data/TheOneData/OpenMind/imgs/derivative/T1w/"
export LOGS_DIR="${DERIV_DIR}logs/"
export TMP_FOLDER="${DERIV_DIR}_tmp_samseg/"

PARALLEL="/analyse/Project0403/cortical_thickness/packages/parallel/parallel-20191222/src/parallel"

# ── Settings ──────────────────────────────────────────────────────────────────
N_THREADS=2               # number of subjects to run in parallel
export N_THREADS_SAMSEG=2 # threads per subject (e.g. 3 × 4 = 12 cores)
N_TEST=0                  # set to 0 to run all; set to e.g. 3 to test first

# ── Setup ─────────────────────────────────────────────────────────────────────
mkdir -p "$DERIV_DIR"
mkdir -p "$TMP_FOLDER"
mkdir -p "$LOGS_DIR"

# ── Collect T1 paths ──────────────────────────────────────────────────────────
mapfile -t T1_PATHS < <(ls "${T1W_DIR}"*.nii.gz 2>/dev/null)

echo "Total T1s found: ${#T1_PATHS[@]}"

if [ "$N_TEST" -gt 0 ]; then
    echo "TEST MODE: running first $N_TEST subjects only"
    T1_PATHS=("${T1_PATHS[@]:0:$N_TEST}")
fi

# ── SAMSEG function ────────────────────────────────────────────────────────────
samseg_function(){
    # $1: T1 path

    I_SUBJ_FILENAME=$(basename "$1")
    I_SUBJ_FILENAME="${I_SUBJ_FILENAME%.nii.gz}"
    # Match existing derivative dir regardless of _256iso suffix
    I_SUBJ_BASENAME="${I_SUBJ_FILENAME%_256iso}"
    if [ -d "${DERIV_DIR}${I_SUBJ_FILENAME}/" ]; then
        I_SUBJ_OUT_DIR="${DERIV_DIR}${I_SUBJ_FILENAME}/"
    elif [ -d "${DERIV_DIR}${I_SUBJ_BASENAME}/" ]; then
        I_SUBJ_OUT_DIR="${DERIV_DIR}${I_SUBJ_BASENAME}/"
    else
        I_SUBJ_OUT_DIR="${DERIV_DIR}${I_SUBJ_FILENAME}/"
    fi
    I_TMP_DIR="${TMP_FOLDER}${I_SUBJ_FILENAME}/"
    COMPLETION_FILE="${I_SUBJ_OUT_DIR}samseg_seg.mgz"

    if [ -f "$COMPLETION_FILE" ]; then
        echo "Skipping (already done): $I_SUBJ_FILENAME"
        return
    fi

    if [ -d "$I_TMP_DIR" ]; then
        echo "Skipping (in progress): $I_SUBJ_FILENAME"
        return
    fi

    echo "Processing: $I_SUBJ_FILENAME"
    echo "  T1: $1"
    SECONDS=0

    mkdir -p "$I_SUBJ_OUT_DIR"
    mkdir -p "$I_TMP_DIR"

    run_samseg \
        --input "$1" \
        --output "$I_TMP_DIR" \
        --lesion \
        --threads $N_THREADS_SAMSEG \
        > "${LOGS_DIR}${I_SUBJ_FILENAME}_samseg_log.txt" 2>&1

    # Move only seg.mgz into the subject derivative dir, delete everything else; rename as samseg_seg.mgz
    if [ -f "${I_TMP_DIR}seg.mgz" ]; then
        mv "${I_TMP_DIR}seg.mgz" "${I_SUBJ_OUT_DIR}samseg_seg.mgz"
        rm -r "$I_TMP_DIR"
        echo "$((SECONDS/3600))-hours"
        echo "$((SECONDS/3600))-hours" > "${LOGS_DIR}${I_SUBJ_FILENAME}_time.txt"
    else
        echo "ERROR: seg.mgz not produced for $I_SUBJ_FILENAME — check log"
    fi
}
export -f samseg_function

# ── Run ───────────────────────────────────────────────────────────────────────
printf '%s\n' "${T1_PATHS[@]}" | $PARALLEL --jobs $N_THREADS "samseg_function {}"

echo "ALL DONE"
