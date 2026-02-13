#!/bin/bash

# !!! This is a CPU intensive script !!!

# Script that generates:
# - Segmentation files with FS
# - Level sets with FS, then fuse them with a script
# - Pial and WM surfaces
# - Thickness overlay
# - orig.mgz (in native space)
# - distance sets

# To run on the grid with:
# $ cd '/analyse/Project0403/cortical_thickness/src/DeepThickness/CD/DeepThickness/scripts/'
# $ bash ./FreeSurfer_GT.sh

# Export paths
SUBJ_NII_DIR="/analyse/Project0406/TheOneSeg/data/test/T1/"
export OUT_DIR="/analyse/Project0406/TheOneSeg/output/testing/HR/"
export LOGS_DIR="/analyse/Project0406/TheOneSeg/output/testing/HR/logs/"
export LEVEL_SET_SCRIPT="/analyse/Project0403/cortical_thickness/src/DeepThickness/CD/DeepThickness/scripts/level_set_fusion.py"
export DISTANCE_SET_SCRIPT="/analyse/Project0403/cortical_thickness/src/DeepThickness/CD/DeepThickness/scripts/distance_set_generator.py"
export SEGMENTATION_MASKS_SCRIPT="/analyse/Project0403/cortical_thickness/src/DeepThickness/CD/DeepThickness/scripts/segmentation_masks_generator.py"
export OUT_IDENTIFIER="rh.thickness"

# The following is the SUBJECTS_DIR temporary foder where FS will put each subject data
export TMP_FOLDER=$OUT_DIR"_tmp_freeSurfer/"          # Inside the OUT_DIR: runnable on multiple machine
export PYTHON_RUN="/analyse/Project0403/cortical_thickness/demo/venvs/deepthickness/bin/python"
PARALLEL="/analyse/Project0403/cortical_thickness/packages/parallel/parallel-20191222/src/parallel"

N_THREADS=4
export N_MULTI_OPENMP=4
export LEVEL_SETS_CAP_VALUE=5

# change FS local variable (output dir)
export SUBJECTS_DIR=$TMP_FOLDER

# Create out and tmp folder if they do not exist
if [ ! -d $OUT_DIR ]; then
    mkdir $OUT_DIR
fi
if [ ! -d $TMP_FOLDER ]; then
    mkdir $TMP_FOLDER
fi
if [ ! -d $LOGS_DIR ]; then
    mkdir $LOGS_DIR
fi

gt_estimation_function(){
    # Function to obtain the FS's GT (and the relabeled)
    # $1: $i_subj (aka T1)
    # $2: $FULL_FILENAME_OUT
    # $3: $I_SUBJ_FILENAME

    I_SUBJ_FILENAME=$3
    I_OUT_TMP_FOLDER=$TMP_FOLDER$I_SUBJ_FILENAME"/"
    
    # If the temp output folder does not exist then calculate the brainMask (means doing it by someone)
    if [ ! -d $I_OUT_TMP_FOLDER ]; then
    
        # If output does not exist then calculate the brainMask (means already done it)
        if [ ! -f $2 ]; then

            echo $1
            SECONDS=0
            
            # Create out if they do not exist
            I_SUBJ_OUT_DIR=$OUT_DIR$I_SUBJ_FILENAME"/"
            if [ ! -d "$I_SUBJ_OUT_DIR" ]; then
                mkdir "$I_SUBJ_OUT_DIR"
            fi

            Run FS command to obtain everything
            recon-all -all -expert "/analyse/Project0235/segmentator/src/utils/FS_options.opts" -i $1 -subjid $I_SUBJ_FILENAME -openmp $N_MULTI_OPENMP > $LOGS_DIR$I_SUBJ_FILENAME"_FS_out_log.txt"

            Convert orig.mgz to native space and save it
            mri_vol2vol --mov $I_OUT_TMP_FOLDER"mri/orig.mgz" --targ $I_OUT_TMP_FOLDER"mri/rawavg.mgz" --regheader --o $I_SUBJ_OUT_DIR"orig_NS.mgz" --no-save-reg --trilin > "/tmp/out.txt"

            # Keep these files: 'aseg.mgz' and 'aparc+aseg.mgz' (converted in native space)
            # Convert FS's results back to native space
            mri_vol2vol --mov $I_OUT_TMP_FOLDER"mri/aseg.mgz" --targ $I_OUT_TMP_FOLDER"mri/rawavg.mgz" --regheader --o $I_SUBJ_OUT_DIR"FS_aseg_NS.mgz" --no-save-reg --nearest > $I_OUT_TMP_FOLDER"conv_aseg.txt"
            mri_vol2vol --mov $I_OUT_TMP_FOLDER"mri/aparc+aseg.mgz" --targ $I_OUT_TMP_FOLDER"mri/rawavg.mgz" --regheader --o $I_SUBJ_OUT_DIR"FS_aparc+aseg_NS.mgz" --no-save-reg --nearest > $I_OUT_TMP_FOLDER"conv_aparc.txt"

            # NEW # Skull-stripped and defaced files
            # Save FreeSurfer skull-stripped brain (brainmask.mgz) in native space
            mri_vol2vol --mov $I_OUT_TMP_FOLDER"mri/brainmask.mgz" --targ $I_OUT_TMP_FOLDER"mri/rawavg.mgz" --regheader --o $I_SUBJ_OUT_DIR"FS_brainmask_NS.mgz" --no-save-reg --nearest > $I_OUT_TMP_FOLDER"conv_brainmask.txt"

            # Save a binarised brain mask (native space)
            mri_binarize --i $I_SUBJ_OUT_DIR"FS_brainmask_NS.mgz" --min 1 --o $I_SUBJ_OUT_DIR"FS_brainmask_mask_NS.mgz" > $I_OUT_TMP_FOLDER"bin_brainmask.txt"

            # Deface the input anatomical and save it (native space)
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

            # Compute the level sets of the subject between -5 and 5
            mris_volmask --save_distance --cap_distance $LEVEL_SETS_CAP_VALUE $I_SUBJ_FILENAME  > "/tmp/out.txt"

            # Convert level sets to native space
            mri_vol2vol --mov $I_OUT_TMP_FOLDER"mri/lh.dpial.ribbon.mgz" --targ $I_OUT_TMP_FOLDER"mri/rawavg.mgz" --regheader --o $I_SUBJ_OUT_DIR"FS_lh.dpial.ribbon_NS.mgz" --no-save-reg --trilin > $I_OUT_TMP_FOLDER"conv_lh_dpial.txt"  > "/tmp/out.txt"
            mri_vol2vol --mov $I_OUT_TMP_FOLDER"mri/lh.dwhite.ribbon.mgz" --targ $I_OUT_TMP_FOLDER"mri/rawavg.mgz" --regheader --o $I_SUBJ_OUT_DIR"FS_lh.dwhite.ribbon_NS.mgz" --no-save-reg --trilin > $I_OUT_TMP_FOLDER"conv_lh_dwhite.txt"  > "/tmp/out.txt"
            mri_vol2vol --mov $I_OUT_TMP_FOLDER"mri/rh.dpial.ribbon.mgz" --targ $I_OUT_TMP_FOLDER"mri/rawavg.mgz" --regheader --o $I_SUBJ_OUT_DIR"FS_rh.dpial.ribbon_NS.mgz" --no-save-reg --trilin > $I_OUT_TMP_FOLDER"conv_rh_dpial.txt"  > "/tmp/out.txt"
            mri_vol2vol --mov $I_OUT_TMP_FOLDER"mri/rh.dwhite.ribbon.mgz" --targ $I_OUT_TMP_FOLDER"mri/rawavg.mgz" --regheader --o $I_SUBJ_OUT_DIR"FS_rh.dwhite.ribbon_NS.mgz" --no-save-reg --trilin > $I_OUT_TMP_FOLDER"conv_rh_dwhite.txt"  > "/tmp/out.txt"

            # Merge the level sets of the two hemispheres
            $PYTHON_RUN $LEVEL_SET_SCRIPT $I_SUBJ_OUT_DIR"FS_lh.dpial.ribbon_NS.mgz" $I_SUBJ_OUT_DIR"FS_rh.dpial.ribbon_NS.mgz" --output $I_SUBJ_OUT_DIR"FS_dpial.ribbon_NS.mgz"  > "/tmp/out.txt"
            $PYTHON_RUN $LEVEL_SET_SCRIPT $I_SUBJ_OUT_DIR"FS_lh.dwhite.ribbon_NS.mgz" $I_SUBJ_OUT_DIR"FS_rh.dwhite.ribbon_NS.mgz" --output $I_SUBJ_OUT_DIR"FS_dwhite.ribbon_NS.mgz" > "/tmp/out.txt"

            # Convert surfaces to native space
            tkregister2 --mov $I_OUT_TMP_FOLDER"mri/rawavg.mgz" --targ $I_OUT_TMP_FOLDER"mri/orig.mgz" --reg $I_OUT_TMP_FOLDER"mri/register.native.dat" --noedit --regheader  > "/tmp/out.txt"
            for HEMI in lh rh
            do
                for SURFACE in white pial
                do
                    mri_surf2surf --sval-xyz $SURFACE --hemi $HEMI --reg $I_OUT_TMP_FOLDER"mri/register.native.dat" $I_OUT_TMP_FOLDER"mri/rawavg.mgz" --tval $I_SUBJ_OUT_DIR$HEMI"."$SURFACE".native" --tval-xyz $I_OUT_TMP_FOLDER"mri/rawavg.mgz" --s $I_SUBJ_FILENAME  > "/tmp/out.txt"
                done
            done

            # Compute distance sets
            $PYTHON_RUN $DISTANCE_SET_SCRIPT --lh_surf $I_SUBJ_OUT_DIR"lh.pial.native" --rh_surf $I_SUBJ_OUT_DIR"rh.pial.native" --lh_opposite_level_set $I_SUBJ_OUT_DIR"FS_lh.dwhite.ribbon_NS.mgz" --rh_opposite_level_set $I_SUBJ_OUT_DIR"FS_rh.dwhite.ribbon_NS.mgz" --orig $I_SUBJ_OUT_DIR"orig_NS.mgz" --output $I_SUBJ_OUT_DIR"pial_distance_set_NS.nii.gz" --dilation_iters 1
            $PYTHON_RUN $DISTANCE_SET_SCRIPT --lh_surf $I_SUBJ_OUT_DIR"lh.white.native" --rh_surf $I_SUBJ_OUT_DIR"rh.white.native" --lh_opposite_level_set $I_SUBJ_OUT_DIR"FS_lh.dpial.ribbon_NS.mgz" --rh_opposite_level_set $I_SUBJ_OUT_DIR"FS_rh.dpial.ribbon_NS.mgz" --orig $I_SUBJ_OUT_DIR"orig_NS.mgz" --output $I_SUBJ_OUT_DIR"wm_distance_set_NS.nii.gz" --dilation_iters 1

            # Move useful output
            mv $I_OUT_TMP_FOLDER"surf/"*".thickness" $I_SUBJ_OUT_DIR              # Thickness overlay

            # Delete tmp files
            #rm -r $I_OUT_TMP_FOLDER
            echo $[$SECONDS/60/60]"-hours"
            echo $[$SECONDS/60/60]"-hours" > $LOGS_DIR$I_SUBJ_FILENAME"_time.txt"
                        
        fi
    fi
}
export -f gt_estimation_function

grab_and_run_function(){
    # Function to grab the i_subj and run the GT function: to use with $PARALLEL
    #   IN: $i_subj (i.e., $1)
    #   OUT: ** run GT estimation **
    #   $1: $i_subj (aka T1)

    # Decompose 'i_subj' for the folder name and ID
    I_SUBJ_FILENAME=$(echo `basename "$1"`)
    I_SUBJ_FILENAME=${I_SUBJ_FILENAME%".nii.gz"}
    FULL_FILENAME_OUT=$OUT_DIR$I_SUBJ_FILENAME"/"$OUT_IDENTIFIER
    
    # Run threads relative to skull stripping
    gt_estimation_function $1 $FULL_FILENAME_OUT $I_SUBJ_FILENAME

}
export -f grab_and_run_function

$PARALLEL --jobs $N_THREADS "grab_and_run_function {}" ::: $SUBJ_NII_DIR"/"*

echo "ALL DONE"
