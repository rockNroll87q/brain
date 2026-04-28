#!/bin/bash

# !!! This is a CPU intensive script !!!

# Script that generates:
# - Segmentation files with FS (T1 + T2 pial refinement)
# - Level sets with FS, then fuse them with a script
# - Pial and WM surfaces
# - Thickness overlay
# - orig.mgz (in native space)
# - distance sets

# Reads T1/T2 pairs from a CSV (columns: t1w_filename, t2w_filename)
# To test on a few subjects first, set N_TEST to a small number (e.g. 3),
# then set to 0 to run all.

# To run:
# $ bash ./FreeSurfer_GT_T2.sh

# ── Paths ─────────────────────────────────────────────────────────────────────
CSV_FILE="/analyse/Project0406/TheOneSeg/data/OpenMind/analyses/T2w_with_fullbrain_T1w.csv"
T1W_DIR="/analyse/Project0406/TheOneSeg/data/TheOneData/OpenMind/imgs/anat/T1w/"
T2W_DIR="/analyse/Project0406/TheOneSeg/data/TheOneData/OpenMind/imgs/anat/T2w/"

export OUT_DIR="/analyse/Project0406/TheOneSeg/data/TheOneData/OpenMind/imgs/derivative/T2w/"
export LOGS_DIR="/analyse/Project0406/TheOneSeg/data/OpenMind/T1_t2_GT_test/logs/"

export LEVEL_SET_SCRIPT="/analyse/Project0403/cortical_thickness/src/DeepThickness/CD/DeepThickness/scripts/level_set_fusion.py"
export DISTANCE_SET_SCRIPT="/analyse/Project0403/cortical_thickness/src/DeepThickness/CD/DeepThickness/scripts/distance_set_generator.py"
export SEGMENTATION_MASKS_SCRIPT="/analyse/Project0403/cortical_thickness/src/DeepThickness/CD/DeepThickness/scripts/segmentation_masks_generator.py"
export OUT_IDENTIFIER="rh.thickness"

export TMP_FOLDER=$OUT_DIR"_tmp_freeSurfer/"
export PYTHON_RUN="/analyse/Project0403/cortical_thickness/demo/venvs/deepthickness/bin/python"
PARALLEL="/analyse/Project0403/cortical_thickness/packages/parallel/parallel-20191222/src/parallel"

# ── Settings ──────────────────────────────────────────────────────────────────
N_THREADS=3           # number of subjects to run in parallel
export N_MULTI_OPENMP=4   # openmp threads per subject
export LEVEL_SETS_CAP_VALUE=5
N_TEST=0             # set to 0 to run all subjects; set to e.g. 3 to test first 3

# ── Setup ─────────────────────────────────────────────────────────────────────
export SUBJECTS_DIR=$TMP_FOLDER

mkdir -p "$OUT_DIR"
mkdir -p "$TMP_FOLDER"
mkdir -p "$LOGS_DIR"

# ── Read CSV into pairs ────────────────────────────────────────────────────────
# CSV columns: dataset,subject,session,t1w_filename,t2w_filename,t2w_is_full_brain
# Build list of "T1_PATH::T2_PATH" pairs, skipping header
PAIRS=()
while IFS=',' read -r dataset subject session t1w_fname t2w_fname t2w_full; do
    [[ "$dataset" == "dataset" ]] && continue   # skip header
    [[ -z "$t1w_fname" || -z "$t2w_fname" ]] && continue
    T1_PATH="${T1W_DIR}${t1w_fname}"
    T2_PATH="${T2W_DIR}${t2w_fname}"
    PAIRS+=("${T1_PATH}::${T2_PATH}")
done < "$CSV_FILE"

echo "Total pairs in CSV: ${#PAIRS[@]}"

# Limit to N_TEST if set
if [ "$N_TEST" -gt 0 ]; then
    echo "TEST MODE: running first $N_TEST subjects only"
    PAIRS=("${PAIRS[@]:0:$N_TEST}")
fi

# ── GT estimation function ─────────────────────────────────────────────────────
gt_estimation_function(){
    # $1: T1 path
    # $2: T2 path
    # $3: FULL_FILENAME_OUT (completion check)
    # $4: I_SUBJ_FILENAME (subject ID)

    I_SUBJ_FILENAME=$4
    I_OUT_TMP_FOLDER=$TMP_FOLDER$I_SUBJ_FILENAME"/"

    if [ ! -d $I_OUT_TMP_FOLDER ]; then
        if [ ! -f $3 ]; then

            echo "Processing: $I_SUBJ_FILENAME"
            echo "  T1: $1"
            echo "  T2: $2"
            SECONDS=0

            I_SUBJ_OUT_DIR=$OUT_DIR$I_SUBJ_FILENAME"/"
            mkdir -p "$I_SUBJ_OUT_DIR"

            # Run FS recon-all with T2 pial refinement
            recon-all \
                -all \
                -expert "/analyse/Project0235/segmentator/src/utils/FS_options.opts" \
                -i "$1" \
                -T2 "$2" \
                -T2pial \
                -subjid $I_SUBJ_FILENAME \
                -openmp $N_MULTI_OPENMP \
                > $LOGS_DIR$I_SUBJ_FILENAME"_FS_out_log.txt" 2>&1

            # Convert orig.mgz to native space
            mri_vol2vol --mov $I_OUT_TMP_FOLDER"mri/orig.mgz" --targ $I_OUT_TMP_FOLDER"mri/rawavg.mgz" --regheader --o $I_SUBJ_OUT_DIR"orig_NS.mgz" --no-save-reg --trilin > $I_OUT_TMP_FOLDER"conv_orig.txt"

            # Convert aseg and aparc+aseg to native space
            mri_vol2vol --mov $I_OUT_TMP_FOLDER"mri/aseg.mgz" --targ $I_OUT_TMP_FOLDER"mri/rawavg.mgz" --regheader --o $I_SUBJ_OUT_DIR"FS_aseg_NS.mgz" --no-save-reg --nearest > $I_OUT_TMP_FOLDER"conv_aseg.txt"
            mri_vol2vol --mov $I_OUT_TMP_FOLDER"mri/aparc+aseg.mgz" --targ $I_OUT_TMP_FOLDER"mri/rawavg.mgz" --regheader --o $I_SUBJ_OUT_DIR"FS_aparc+aseg_NS.mgz" --no-save-reg --nearest > $I_OUT_TMP_FOLDER"conv_aparc.txt"

            # Skull-stripped brain in native space
            mri_vol2vol --mov $I_OUT_TMP_FOLDER"mri/brainmask.mgz" --targ $I_OUT_TMP_FOLDER"mri/rawavg.mgz" --regheader --o $I_SUBJ_OUT_DIR"FS_brainmask_NS.mgz" --no-save-reg --nearest > $I_OUT_TMP_FOLDER"conv_brainmask.txt"

            # Binarised brain mask in native space
            mri_binarize --i $I_SUBJ_OUT_DIR"FS_brainmask_NS.mgz" --min 1 --o $I_SUBJ_OUT_DIR"FS_brainmask_mask_NS.mgz" > $I_OUT_TMP_FOLDER"bin_brainmask.txt"

            # Deface and save in native space
            DEFACE_DIR="$I_OUT_TMP_FOLDER/mri"
            mkdir -p "$DEFACE_DIR"
            (
                cd "$DEFACE_DIR" || exit 1
                mri_deface "orig.mgz" \
                    "$FREESURFER_HOME/average/talairach_mixed_with_skull.gca" \
                    "$FREESURFER_HOME/average/face.gca" \
                    "orig_defaced.mgz" \
                    > "deface.txt" 2>&1
            )
            mri_vol2vol --mov $I_OUT_TMP_FOLDER"mri/orig_defaced.mgz" --targ $I_OUT_TMP_FOLDER"mri/rawavg.mgz" --regheader --o $I_SUBJ_OUT_DIR"FS_orig_defaced_NS.mgz" --no-save-reg --trilin > $I_OUT_TMP_FOLDER"conv_orig_defaced.txt"

            # Level sets
            mris_volmask --save_distance --cap_distance $LEVEL_SETS_CAP_VALUE $I_SUBJ_FILENAME > "/tmp/out.txt"

            # Convert level sets to native space
            mri_vol2vol --mov $I_OUT_TMP_FOLDER"mri/lh.dpial.ribbon.mgz" --targ $I_OUT_TMP_FOLDER"mri/rawavg.mgz" --regheader --o $I_SUBJ_OUT_DIR"FS_lh.dpial.ribbon_NS.mgz" --no-save-reg --trilin > $I_OUT_TMP_FOLDER"conv_lh_dpial.txt"
            mri_vol2vol --mov $I_OUT_TMP_FOLDER"mri/lh.dwhite.ribbon.mgz" --targ $I_OUT_TMP_FOLDER"mri/rawavg.mgz" --regheader --o $I_SUBJ_OUT_DIR"FS_lh.dwhite.ribbon_NS.mgz" --no-save-reg --trilin > $I_OUT_TMP_FOLDER"conv_lh_dwhite.txt"
            mri_vol2vol --mov $I_OUT_TMP_FOLDER"mri/rh.dpial.ribbon.mgz" --targ $I_OUT_TMP_FOLDER"mri/rawavg.mgz" --regheader --o $I_SUBJ_OUT_DIR"FS_rh.dpial.ribbon_NS.mgz" --no-save-reg --trilin > $I_OUT_TMP_FOLDER"conv_rh_dpial.txt"
            mri_vol2vol --mov $I_OUT_TMP_FOLDER"mri/rh.dwhite.ribbon.mgz" --targ $I_OUT_TMP_FOLDER"mri/rawavg.mgz" --regheader --o $I_SUBJ_OUT_DIR"FS_rh.dwhite.ribbon_NS.mgz" --no-save-reg --trilin > $I_OUT_TMP_FOLDER"conv_rh_dwhite.txt"

            # Merge level sets across hemispheres
            $PYTHON_RUN $LEVEL_SET_SCRIPT $I_SUBJ_OUT_DIR"FS_lh.dpial.ribbon_NS.mgz" $I_SUBJ_OUT_DIR"FS_rh.dpial.ribbon_NS.mgz" --output $I_SUBJ_OUT_DIR"FS_dpial.ribbon_NS.mgz"
            $PYTHON_RUN $LEVEL_SET_SCRIPT $I_SUBJ_OUT_DIR"FS_lh.dwhite.ribbon_NS.mgz" $I_SUBJ_OUT_DIR"FS_rh.dwhite.ribbon_NS.mgz" --output $I_SUBJ_OUT_DIR"FS_dwhite.ribbon_NS.mgz"

            # Convert surfaces to native space
            tkregister2 --mov $I_OUT_TMP_FOLDER"mri/rawavg.mgz" --targ $I_OUT_TMP_FOLDER"mri/orig.mgz" --reg $I_OUT_TMP_FOLDER"mri/register.native.dat" --noedit --regheader > "/tmp/out.txt"
            for HEMI in lh rh; do
                for SURFACE in white pial; do
                    mri_surf2surf --sval-xyz $SURFACE --hemi $HEMI \
                        --reg $I_OUT_TMP_FOLDER"mri/register.native.dat" \
                        $I_OUT_TMP_FOLDER"mri/rawavg.mgz" \
                        --tval $I_SUBJ_OUT_DIR$HEMI"."$SURFACE".native" \
                        --tval-xyz $I_OUT_TMP_FOLDER"mri/rawavg.mgz" \
                        --s $I_SUBJ_FILENAME > "/tmp/out.txt"
                done
            done

            # Compute distance sets
            $PYTHON_RUN $DISTANCE_SET_SCRIPT \
                --lh_surf $I_SUBJ_OUT_DIR"lh.pial.native" \
                --rh_surf $I_SUBJ_OUT_DIR"rh.pial.native" \
                --lh_opposite_level_set $I_SUBJ_OUT_DIR"FS_lh.dwhite.ribbon_NS.mgz" \
                --rh_opposite_level_set $I_SUBJ_OUT_DIR"FS_rh.dwhite.ribbon_NS.mgz" \
                --orig $I_SUBJ_OUT_DIR"orig_NS.mgz" \
                --output $I_SUBJ_OUT_DIR"pial_distance_set_NS.nii.gz" \
                --dilation_iters 1
            $PYTHON_RUN $DISTANCE_SET_SCRIPT \
                --lh_surf $I_SUBJ_OUT_DIR"lh.white.native" \
                --rh_surf $I_SUBJ_OUT_DIR"rh.white.native" \
                --lh_opposite_level_set $I_SUBJ_OUT_DIR"FS_lh.dpial.ribbon_NS.mgz" \
                --rh_opposite_level_set $I_SUBJ_OUT_DIR"FS_rh.dpial.ribbon_NS.mgz" \
                --orig $I_SUBJ_OUT_DIR"orig_NS.mgz" \
                --output $I_SUBJ_OUT_DIR"wm_distance_set_NS.nii.gz" \
                --dilation_iters 1

            # Move thickness overlay
            mv $I_OUT_TMP_FOLDER"surf/"*".thickness" "$I_SUBJ_OUT_DIR"

            # Delete tmp files (uncomment when happy with output)
            rm -r $I_OUT_TMP_FOLDER

            echo $[$SECONDS/60/60]"-hours"
            echo $[$SECONDS/60/60]"-hours" > $LOGS_DIR$I_SUBJ_FILENAME"_time.txt"

        fi
    fi
}
export -f gt_estimation_function

grab_and_run_function(){
    # $1: "T1_PATH::T2_PATH"
    T1_PATH="${1%%::*}"
    T2_PATH="${1##*::}"

    I_SUBJ_FILENAME=$(basename "$T1_PATH")
    I_SUBJ_FILENAME="${I_SUBJ_FILENAME%.nii.gz}"
    FULL_FILENAME_OUT="${OUT_DIR}${I_SUBJ_FILENAME}/${OUT_IDENTIFIER}"

    gt_estimation_function "$T1_PATH" "$T2_PATH" "$FULL_FILENAME_OUT" "$I_SUBJ_FILENAME"
}
export -f grab_and_run_function

# ── Run ───────────────────────────────────────────────────────────────────────
printf '%s\n' "${PAIRS[@]}" | $PARALLEL --jobs $N_THREADS "grab_and_run_function {}"

echo "ALL DONE"
