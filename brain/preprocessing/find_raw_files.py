#!/usr/bin/env python3

"""
This script allows you to find all the files in a given src_dir and find their locations in the trg_dir (and recursive dirs) 
storing the src and target file paths in a csv 

As input you provide:
- 'src_dir' : The src directory containing the files you want to find 
- 'trg_dir' : The root dir to start the search from (will search all subsidary dirs) 
- 'output' : the dir to save the mapping csv

Example terminal command:   
./find_raw_files.py --src_dir /analyse/Project0404/brain_age/data/project_name/raw/ --trg_dir /analyse/Project0404/brain_age/data/project_name/ --output /analyse/Project0404/brain_age/data/project_name/raw

"""

import os
import csv
import argparse

def build_filename_index(root_dir):
    """
    Walk root_dir recursively and return a dict mapping
    filename -> list of full paths where that filename occurs.
    """
    index = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            index.setdefault(fname, []).append(os.path.join(dirpath, fname))
    return index

def main(args):
    """Run everything"""
    # 1) Build index of trg_dir
    print(f"Indexing filenames under {args.trg_dir}…")
    filename_index = build_filename_index(args.trg_dir)

    # 2) Open CSV and write header
    with open(f'{args.output}/raw_file_mapping.csv', "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["src_path", "found_path"])

        # 3) Walk src_dir and look up each filename
        for dirpath, _, filenames in os.walk(args.src_dir):
            for fname in filenames:
                if "/._" in fname:
                    continue
                src = os.path.join(dirpath, fname)
                matches = filename_index.get(fname, [])
                for found in matches:
                    writer.writerow([src, found])
                    print(f"Matched: {src} → {found}")

    print(f"\nDone! Results written to {args.output}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Map every file in src_dir to matching names under trg_dir, outputting a CSV."
    )
    parser.add_argument(
        "-d1", "--src_dir", required=True,
        help="Source directory whose files you want to match"
    )
    parser.add_argument(
        "-d2", "--trg_dir", required=True,
        help="Target directory tree to search for matching filenames"
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Path to output CSV (will be overwritten if it exists)"
    )
    args = parser.parse_args()

    main(args)
