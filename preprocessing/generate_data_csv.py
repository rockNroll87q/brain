#!/usr/bin/env python3

"""
Generate a CSV mapping subjects and processing tasks for anatomical .nii.gz files,
with configurable train/validation/test splits. The output CSV is saved into a
'csv' directory at the project root (determined relative to the provided anat_dir).
The output CSV filename is provided by the user via an argument.

This script scans an input directory for all unique .nii.gz files, and for each
subject generates rows for each of the predefined tasks (deface, skull_strip,
lr, parcellation, segmentation, tumour, vessels), pairing them with corresponding
affixes. It outputs a CSV with columns:

subj, anat, task, mask, split

where:
  - mask: ProjectName/anat/{subj}_{affix}.nii.gz
  - task: one of the TASKS
  - anat: ProjectName/GT/{subj}.nii.gz
  - subj: subject base filename
  - split: one of {'train', 'valid', 'test'} according to percentages

The output file will be named as specified by --csv-name and placed in:

    <project_root>/csv/<csv-name>


Usage:
  python generate_csv_split.py /path/to/project/imgs/anat \
      --project-name MyProject \
      --csv-name tasks.csv \
      --train 60 --valid 20 --test 20 \
      [--random-seed 42]


Optionally, a random seed can be provided for reproducible splits.
"""

import os
import csv
import argparse
import random
import sys

# Define the tasks and their corresponding affixes
TASKS = [
    "deface",
    "skull_strip",
    "lr",
    "parcellation",
    "segmentation",
    "tumour",
    "vessels",
]
AFFIXES = [
    "defaced",
    "skull_stripped",
    "lr",
    "parcellation",
    "segmentation",
    "tumour_seg",
    "vessel_seg",
]


def find_nii_gz_files(directory):
    """Recursively find all .nii.gz files under directory."""
    for root, _, files in os.walk(directory):
        for f in files:
            if "/._" in f:
                continue
            if f.endswith(".nii.gz"):
                yield os.path.join(root, f)


def get_subjects(anat_dir):
    """Collect unique subject base names from .nii.gz files in anat_dir."""
    bases = set()
    for fullpath in find_nii_gz_files(anat_dir):
        filename = os.path.basename(fullpath)
        bases.add(filename[:-7])  # strip .nii.gz
    return sorted(bases)


def split_subjects(subjects, train_pct, valid_pct, test_pct, seed=None):
    """Partition subjects into train/valid/test lists based on percentages."""
    if seed is not None:
        random.seed(seed)
    subs = subjects.copy()
    random.shuffle(subs)

    total = len(subs)
    n_train = int(total * train_pct / 100)
    n_valid = int(total * valid_pct / 100)
    n_test = total - (n_train + n_valid)

    return subs[:n_train], subs[n_train:n_train + n_valid], subs[n_train + n_valid:]


def main(args):

    # Validate split sums to 100
    total_pct = args.train + args.valid + args.test
    if total_pct != 100:
        parser.error(f"Train+valid+test percentages must sum to 100 (got {total_pct})")

    # Gather subjects
    subjects = get_subjects(args.anat_dir)
    if not subjects:
        sys.exit("No .nii.gz files found in anat_dir. Exiting.")

    # Split subjects
    train_subjs, valid_subjs, test_subjs = split_subjects(
        subjects, args.train, args.valid, args.test, args.random_seed
    )
    split_map = {s: 'train' for s in train_subjs}
    split_map.update({s: 'valid' for s in valid_subjs})
    split_map.update({s: 'test' for s in test_subjs})

    # Determine output directory
    anat_abs = os.path.abspath(args.anat_dir)
    project_root = os.path.dirname(os.path.dirname(anat_abs))
    csv_dir = os.path.join(project_root, 'csv')
    os.makedirs(csv_dir, exist_ok=True)
    output_path = os.path.join(csv_dir, args.csv_name)

    # Write CSV
    fieldnames = ["anat", "task", "mask", "subj", "split"]
    with open(output_path, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for subj in subjects:
            anat_path = f"{args.project_name}/GT/{subj}.nii.gz"
            for task, affix in zip(TASKS, AFFIXES):
                mask_path = f"{args.project_name}/anat/{subj}_{affix}.nii.gz"
                writer.writerow({
                    "subj": subj,
                    "anat": anat_path,
                    "task": task,
                    "mask": mask_path,
                    "split": split_map[subj],
                })

    total_rows = len(subjects) * len(TASKS)
    print(f"Saved CSV to {output_path} with {total_rows} rows across splits:")
    print(f"  train: {len(train_subjs)} subjects, valid: {len(valid_subjs)} subjects, test: {len(test_subjs)} subjects")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate a tasks CSV from an anat directory with splits."
    )
    parser.add_argument(
        "anat_dir",
        help="Path to the directory containing your .nii.gz anatomical files",
    )
    parser.add_argument(
        "--project-name",
        required=True,
        help="Project name to prefix your paths (e.g. MyProject)",
    )
    parser.add_argument(
        "--csv-name",
        required=True,
        help="Filename for the output CSV (e.g., tasks.csv)",
    )
    parser.add_argument(
        "--train",
        type=int,
        default=60,
        help="Percentage of subjects in the training set (default: 60)",
    )
    parser.add_argument(
        "--valid",
        type=int,
        default=20,
        help="Percentage of subjects in the validation set (default: 20)",
    )
    parser.add_argument(
        "--test",
        type=int,
        default=20,
        help="Percentage of subjects in the test set (default: 20)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Random seed for reproducible splits (default: None)",
    )
    args = parser.parse_args()

    main(args)
