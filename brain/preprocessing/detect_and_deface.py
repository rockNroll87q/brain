"""
pydeface_detect.py

Determines whether NIfTI volumes are already defaced by running pydeface and
measuring how much the volume changes before and after. A volume that is already
defaced will show little or no change; an intact volume will show a large change.

For volumes classified as intact (based on the face ROI metric), the pydeface
face mask is saved to the corresponding derivative directory before the temporary
working directory is cleaned up:

    {data_root}/{dataset}/imgs/derivative/{modality}/{stem}/deface_mask.nii.gz

where {stem} is the volume filename with the extension stripped. Each volume gets
its own participant folder; the mask is always named deface_mask.nii.gz so that
future masks (aparc, aseg, etc.) can sit alongside it, differentiated by filename.

Usage (old) (csh shell on HPC):
    python pydeface_detect.py --data_root /path/to/TheOneData --out_dir /path/to/results --n_jobs 64

Requirements:
    pip install pydeface nibabel numpy pandas joblib tqdm

FSL must be on PATH:
    setenv FSLDIR /usr/local/fsl
    setenv PATH /usr/local/fsl/bin:$PATH


to run updated script on a single dataset (e.g. ADNI):
python /analyse/Project0406/TheOneSeg/data/src/detect_and_deface.py \
    --data_root /analyse/Project0406/TheOneSeg/data/TheOneData \
    --out_dir /analyse/Project0406/TheOneSeg/data/TheOneData/ADNI/csv \
    --dataset ADNI \
    --n_jobs 12
"""

import argparse
import json
import logging
import random
import shutil
import subprocess
import tempfile
import warnings
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

CONFIG = {
    "changed_voxel_threshold": 5,    # minimum per-voxel intensity change to count as changed
    "changed_ratio_threshold": 0.01,  # fraction of whole-volume voxels changed; below = already defaced
    "roi_ratio_threshold":     0.02, # same signal but restricted to face ROI; tuned

    # Face ROI definition (LIA orientation, 256^3 at 1mm isotropic)
    # Derived from freeview voxel coordinates on a known-intact ADNI volume (means that its tuned for ADNI)
    # Could add: if dataset = x, then use these coords for each dataset
    #   nose tip:  [127, 159, 231]
    #   chin:      [127, 218, 227]
    #   mouth:     [127, 175, 188]
    # Coordinates are fractional (0-1) along each axis.
    # X: full width — face is symmetric left-right
    # Y: 0.62-0.85 — voxels ~159-218, nose to chin
    # Z: 0.73-0.90 — voxels ~188-231, mouth to nose tip
    "face_roi": {
        "x": (0.0, 1.0),
        "y": (0.62, 0.85),
        "z": (0.73, 0.90),
    },
    "min_voxels_in_roi": 500,  # skip ROI metrics if ROI is smaller than this
}


# ---------------------------------------------------------------------------
# VOLUME DISCOVERY
# ---------------------------------------------------------------------------

def discover_volumes(data_root, sample=None):
    """
    Find all NIfTI volumes under data_root matching:
        {dataset}/imgs/anat/{modality}/{file}.nii.gz
    Excludes non-conformed folders. If sample is set, returns N random
    volumes per dataset.
    """
    records = []
    hits = [
        p for p in data_root.glob("*/imgs/anat/*/*.nii*")
        if not p.parent.name.endswith("_non_conf")
    ]

    for fpath in hits:
        parts = fpath.relative_to(data_root).parts
        records.append({
            "dataset":  parts[0] if len(parts) >= 5 else "unknown",
            "modality": parts[3] if len(parts) >= 5 else "unknown",
            "filepath": str(fpath),
        })

    if sample:
        by_dataset = defaultdict(list)
        for r in records:
            by_dataset[r["dataset"]].append(r)
        records = []
        for vols in by_dataset.values():
            records.extend(random.sample(vols, min(sample, len(vols))))

    return records


# ---------------------------------------------------------------------------
# RESUME
# ---------------------------------------------------------------------------

def load_completed(out_dir):
    """
    Read existing results CSV and return filepaths already successfully
    processed. Allows the run to resume after interruption.
    """
    csv_path = out_dir / "results.csv"
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path)
        completed = set(df.loc[df["error"].isna(), "filepath"].tolist())
        logging.info("Resuming — {} volumes already processed".format(len(completed)))
        return completed
    except Exception as exc:
        logging.warning("Could not read existing CSV, starting fresh: {}".format(exc))
        return set()


# ---------------------------------------------------------------------------
# MASK PATH HELPERS
# ---------------------------------------------------------------------------

def _gt_mask_path(filepath, data_root):
    """
    Given an anat volume path and data_root, return the target path for the
    pydeface mask in the derivative tree.

    Input layout:
        {data_root}/{dataset}/imgs/anat/{modality}/{filename}.nii.gz

    Output layout:
        {data_root}/{dataset}/imgs/derivative/{modality}/{stem}/deface_mask.nii.gz

    where {stem} is the volume filename with the .nii.gz / .nii extension
    stripped. This gives each volume its own participant folder, with a fixed
    mask filename so that future masks (aparc, aseg, etc.) can sit alongside
    it differentiated by filename rather than by path.

    data_root may be a symlink directory (multi-node HPC setup); we resolve
    the real path so the mask lands next to the real data.
    """
    filepath  = Path(filepath)
    data_root = Path(data_root)

    # Resolve symlinks so the output path is absolute and real
    real_filepath  = filepath.resolve()
    real_data_root = data_root.resolve()

    try:
        rel_parts = real_filepath.relative_to(real_data_root).parts
    except ValueError:
        # Fallback: use unresolved path (shouldn't happen in normal operation)
        rel_parts = filepath.relative_to(data_root).parts

    # rel_parts: (dataset, "imgs", "anat", modality, filename)
    if len(rel_parts) < 5:
        return None

    dataset, modality, filename = rel_parts[0], rel_parts[3], rel_parts[4]

    # Participant folder = full filename stem (strip .nii.gz or .nii)
    stem = filename
    for suffix in (".nii.gz", ".nii"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    dest = real_data_root / dataset / "imgs" / "derivative" / modality / stem / "deface_mask.nii.gz"
    return dest


# ---------------------------------------------------------------------------
# PROCESS ONE VOLUME
# ---------------------------------------------------------------------------

def process_volume(filepath, cfg, data_root):
    """
    Run pydeface on one volume and compute whole-volume before/after diff metrics,
    plus two focused metrics within the face ROI specifically.

    The input volume is copied into a temporary directory before running pydeface
    so that all intermediate files (mask, .mat) are contained there and cleaned
    up automatically. Nothing is written to the data directory — except for the
    defacemask when the volume is classified as intact (see below).

    IMPORTANT — ROI metrics are computed on abs_diff (the change volume), NOT on
    the original intensities (data_pre). (The difference from
    detect_defaced.py, which measured signal in the face region of the original
    volume. That approach failed for ADNI/ALLFTD probably because blurred faces have low
    signal in the original volume, indistinguishable from defaced volumes, either way,
    using the whole volume signal as decision metric incorrectly identified intact volumes as defaced)

    By applying the ROI to abs_diff instead, we directly measure how much the
    face region changed after pydeface ran:
        - Already defaced:   face region has nothing left to remove -> low ROI change
        - Blurred but intact: face region has residual signal -> pydeface removes
                              some -> expected to show more ROI change than defaced
        - Fully intact:      large face signal -> pydeface removes it -> high ROI change

    Mask saving
    -----------
    pydeface writes a face mask alongside the defaced volume:
        <tmp>/{stem}_defacemask.nii.gz

    If the volume is classified as intact based on the ROI metric
    (changed_ratio_face_roi >= roi_ratio_threshold), this mask is copied to the
    derivative tree *before* the temp directory is deleted:
        {data_root}/{dataset}/imgs/derivative/{modality}/{stem}/deface_mask.nii.gz

    If the ROI metric is unavailable (volume too small), falls back to the
    whole-volume changed_voxel_ratio for the mask-saving decision.

    The destination directory is created with exist_ok=True. If the mask file
    already exists it is silently overwritten.

    Whole-volume metrics:
        changed_voxel_ratio  — fraction of ALL voxels that changed by more than
            changed_voxel_threshold. Retained for comparison; not used for decisions.
        mean_diff            — mean absolute change across all voxels.
        max_diff             — largest single voxel change anywhere in the volume.

    Face ROI metrics (computed on abs_diff, not original intensities):
        mean_diff_face_roi      — mean absolute change within the face ROI only.
        changed_ratio_face_roi  — fraction of face ROI voxels that changed by more
                                  than changed_voxel_threshold. CURRENT PRIMARY decision signal.
    """
    base = {"filepath": filepath, "error": None, "mask_saved_path": None}

    try:
        # Load original volume
        img_pre  = nib.load(filepath)
        data_pre = np.asarray(img_pre.dataobj, dtype=np.float32)
        data_pre = np.nan_to_num(data_pre, nan=0.0)  # convert nan to 0 in case defacing tool used this method

        shape = data_pre.shape[:3]

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)

            # Copy input into temp dir so pydeface writes all files there rather than data dir
            # (prevents .mat and mask files appearing in the data directory)
            tmp_input = tmp_dir / Path(filepath).name

            stem = Path(filepath).name
            for suf in (".nii.gz", ".nii"):
                if stem.endswith(suf):
                    stem = stem[: -len(suf)]
                    break

            shutil.copy2(filepath, str(tmp_input))

            out_path  = tmp_dir / "{}_defaced.nii.gz".format(stem)
            mask_path = tmp_dir / "{}_pydeface_mask.nii.gz".format(stem) # pydeface's output name

            # Run pydeface (as a subprocess)
            result = subprocess.run(
                ["pydeface", str(tmp_input), "--outfile", str(out_path), "--force", "--nocleanup"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            if result.returncode != 0:
                raise RuntimeError(result.stderr.decode())  # continue to next volume and log the error

            # TEMP DIAGNOSTIC — remove after confirming
            import os
            logging.info("Temp dir contents for {}: {}".format(filepath, os.listdir(str(tmp_dir))))
            
            # Load defaced volume
            data_post = np.asarray(nib.load(str(out_path)).dataobj, dtype=np.float32)
            data_post = np.nan_to_num(data_post, nan=0.0)

            # --- Whole-volume difference ---
            # abs_diff is a 3D array of the same shape as the volume.
            # Each voxel value = how much that voxel changed after pydeface.
            abs_diff            = np.abs(data_pre - data_post)
            changed_voxel_ratio = float(np.mean(abs_diff > cfg["changed_voxel_threshold"]))
            mean_diff           = float(np.mean(abs_diff))
            max_diff            = float(np.max(abs_diff))

            # --- Face ROI difference ---
            # Slice the face ROI out of abs_diff (NOT out of data_pre).
            # Coords are fractional so they scale correctly across resolutions.
            roi_cfg = cfg["face_roi"]
            x0 = int(roi_cfg["x"][0] * shape[0]);  x1 = int(roi_cfg["x"][1] * shape[0])
            y0 = int(roi_cfg["y"][0] * shape[1]);  y1 = int(roi_cfg["y"][1] * shape[1])
            z0 = int(roi_cfg["z"][0] * shape[2]);  z1 = int(roi_cfg["z"][1] * shape[2])

            roi_diff = abs_diff[x0:x1, y0:y1, z0:z1]

            if roi_diff.size < cfg["min_voxels_in_roi"]:
                # ROI too small to be meaningful — record None but keep whole-volume metrics
                mean_diff_face_roi     = None
                changed_ratio_face_roi = None
            else:
                mean_diff_face_roi     = float(np.mean(roi_diff))
                changed_ratio_face_roi = float(np.mean(roi_diff > cfg["changed_voxel_threshold"]))

            # ----------------------------------------------------------------
            # Save mask for intact volumes before temp dir is wiped.
            # Decision is based on the ROI metric (primary signal). Falls back
            # to the whole-volume ratio if ROI was unavailable.
            # ----------------------------------------------------------------
            if changed_ratio_face_roi is not None:
                is_intact = changed_ratio_face_roi >= cfg["roi_ratio_threshold"]
            else:
                is_intact = changed_voxel_ratio >= cfg["changed_ratio_threshold"]

            if is_intact and mask_path.exists():
                dest = _gt_mask_path(filepath, data_root)
                if dest is not None:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(mask_path), str(dest))
                    base["mask_saved_path"] = str(dest)
                else:
                    logging.warning("Could not derive GT mask path for {}".format(filepath))

        # temp directory and all pydeface outputs deleted here automatically

        # package all metrics into a dict
        return {
            **base,
            # Whole-volume metrics
            "mean_diff":              mean_diff,
            "max_diff":               max_diff,
            "changed_voxel_ratio":    changed_voxel_ratio,
            # Face ROI diff metrics (computed on abs_diff, not data_pre)
            "mean_diff_face_roi":     mean_diff_face_roi,
            "changed_ratio_face_roi": changed_ratio_face_roi,
            # Volume metadata
            "volume_shape":           json.dumps(list(shape)),
            "voxel_dims":             json.dumps([float(z) for z in img_pre.header.get_zooms()[:3]]),
        }

    except Exception as exc:
        base["error"] = str(exc)
        return base


# ---------------------------------------------------------------------------
# CLASSIFICATION
# ---------------------------------------------------------------------------

def classify(df, cfg):
    """
    Classify each volume using two independent predictions:

    prediction (whole-volume) — retained for comparison and visualisation only.
        Based on changed_voxel_ratio across the entire volume.

    prediction_roi — PRIMARY label, drives colour coding, needs_review, and
        determines whether a mask was saved.
        Based on changed_ratio_face_roi, which is the same signal restricted to the
        face region only. More discriminating for blurred-face datasets (ADNI/ALLFTD)
        Set to 'roi_unavailable' if the ROI metric could not be computed (volume too
        small); in that case the whole-volume prediction is used as the fallback.

    Both thresholds are independently configurable in CONFIG so they can be tuned
    separately once results are inspected.
    """
    thresh     = cfg["changed_ratio_threshold"]
    roi_thresh = cfg["roi_ratio_threshold"]

    # Whole-volume prediction (retained for comparison — not used for decisions)
    df["prediction"] = np.where(df["changed_voxel_ratio"] < thresh, "defaced", "intact")

    # ROI prediction (PRIMARY — drives needs_review and mask saving)
    roi_available = df["changed_ratio_face_roi"].notna()
    df["prediction_roi"] = "roi_unavailable"
    df.loc[roi_available, "prediction_roi"] = np.where(
        df.loc[roi_available, "changed_ratio_face_roi"] < roi_thresh,
        "defaced",
        "intact",
    )

    # needs_review based on ROI metric proximity to roi_ratio_threshold.
    # Falls back to whole-volume threshold for roi_unavailable rows.
    df["needs_review"] = False
    df.loc[roi_available, "needs_review"] = (
        (df.loc[roi_available, "changed_ratio_face_roi"] - roi_thresh).abs()
        < (roi_thresh * 0.5)
    )
    df.loc[~roi_available, "needs_review"] = (
        (df.loc[~roi_available, "changed_voxel_ratio"] - thresh).abs()
        < (thresh * 0.5)
    )

    # Errors override both predictions
    df.loc[df["error"].notna(), "prediction"]     = "error"
    df.loc[df["error"].notna(), "prediction_roi"] = "error"
    df.loc[df["error"].notna(), "needs_review"]   = False

    return df


# ---------------------------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------------------------

def save_results(df, out_dir):
    """Save results CSV, summary JSON, and HTML report."""
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / "results.csv", index=False)

    masks_saved = int(df["mask_saved_path"].notna().sum()) if "mask_saved_path" in df.columns else 0

    # Summary counts are based on prediction_roi (the primary label)
    summary = {
        "total":        int(len(df)),
        "defaced":      int((df["prediction_roi"] == "defaced").sum()),
        "intact":       int((df["prediction_roi"] == "intact").sum()),
        "errors":       int((df["prediction_roi"] == "error").sum()),
        "needs_review": int(df["needs_review"].sum()),
        "masks_saved":  masks_saved,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    _write_html_report(df, summary, out_dir / "report.html")
    logging.info("Results saved to {}".format(out_dir))
    return summary


def _write_html_report(df, summary, path):
    rows = ""
    for _, r in df.iterrows():
        # Colour coding driven by prediction_roi (primary label)
        pred_roi = r.get("prediction_roi", "")
        review   = r.get("needs_review", False)
        colour = (
            "#fff3cd" if pred_roi == "defaced"
            else "#f8d7da" if pred_roi == "error"
            else "#d1ecf1" if review
            else "#d4edda"
        )
        mask_path = r.get("mask_saved_path") or ""
        rows += """<tr style="background:{c}">
          <td>{dataset}</td><td>{modality}</td>
          <td style="font-size:0.75em;word-break:break-all">{fp}</td>
          <td><b>{pred_roi}</b></td><td>{pred}</td>
          <td>{roi_ratio}</td><td>{ratio}</td>
          <td>{roi_mdiff}</td><td>{mdiff}</td>
          <td>{maxd}</td>
          <td style="font-size:0.75em;word-break:break-all">{mask}</td>
          <td>{err}</td></tr>""".format(
            c=colour,
            dataset=r.get("dataset", ""), modality=r.get("modality", ""),
            fp=r["filepath"],
            pred_roi =pred_roi,
            pred     =r.get("prediction", "-"),
            roi_ratio="{:.4f}".format(r["changed_ratio_face_roi"]) if pd.notna(r.get("changed_ratio_face_roi")) else "-",
            ratio    ="{:.4f}".format(r["changed_voxel_ratio"])    if pd.notna(r.get("changed_voxel_ratio"))    else "-",
            roi_mdiff="{:.2f}".format(r["mean_diff_face_roi"])     if pd.notna(r.get("mean_diff_face_roi"))     else "-",
            mdiff    ="{:.2f}".format(r["mean_diff"])              if pd.notna(r.get("mean_diff"))              else "-",
            maxd     ="{:.1f}".format(r["max_diff"])               if pd.notna(r.get("max_diff"))               else "-",
            mask=mask_path,
            err=r.get("error") or "",
        )

    html = """<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>Pydeface Detection Report</title>
<style>
  body{{font-family:monospace;font-size:13px;margin:20px}}
  .summary{{display:flex;gap:16px;margin-bottom:12px;flex-wrap:wrap}}
  .card{{padding:8px 16px;border-radius:6px;font-weight:bold}}
  table{{border-collapse:collapse;width:100%}}
  th,td{{border:1px solid #ccc;padding:5px 8px;text-align:left}}
  th{{background:#222;color:#fff;cursor:pointer}}
  input{{margin-bottom:10px;padding:6px;width:300px}}
</style></head><body>
<h1>Pydeface Detection Report</h1>
<div class="summary">
  <div class="card" style="background:#d4edda">Intact (ROI): {intact}</div>
  <div class="card" style="background:#fff3cd">Defaced (ROI): {defaced}</div>
  <div class="card" style="background:#d1ecf1">Needs review: {needs_review}</div>
  <div class="card" style="background:#f8d7da">Errors: {errors}</div>
  <div class="card" style="background:#cce5ff">Masks saved: {masks_saved}</div>
  <div class="card" style="background:#eee">Total: {total}</div>
</div>
<input type="text" id="fi" onkeyup="var q=this.value.toLowerCase();document.querySelectorAll('#t tbody tr').forEach(function(r){{r.style.display=r.innerText.toLowerCase().indexOf(q)>-1?'':'none'}})" placeholder="Filter...">
<table id="t"><thead><tr>
  <th>Dataset</th><th>Modality</th><th>Filepath</th>
  <th>ROI Prediction ★</th><th>Whole-Vol Prediction</th>
  <th>ROI Changed Ratio</th><th>Changed Ratio</th>
  <th>ROI Mean Diff</th><th>Mean Diff</th>
  <th>Max Diff</th><th>Mask Saved Path</th><th>Error</th>
</tr></thead><tbody>{rows}</tbody></table>
</body></html>""".format(rows=rows, **summary)

    path.write_text(html)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main(data_root, out_dir, n_jobs, limit=None, sample=None, dataset=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(str(out_dir / "run.log"))],
    )

    records = discover_volumes(data_root, sample=sample)

    if dataset is not None:
        records = [r for r in records if r["dataset"] == dataset]
        logging.info("Filtered to dataset '{}': {} volumes".format(dataset, len(records)))

    if limit:
        records = records[:limit]

    logging.info("Found {} volumes".format(len(records)))

    completed = load_completed(out_dir)  # read results and skip volumes already checked
    records   = [r for r in records if r["filepath"] not in completed]
    logging.info("{} volumes to process".format(len(records)))

    if not records:
        logging.info("Nothing to do — all volumes already processed")
        return

    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(process_volume)(r["filepath"], CONFIG, data_root)
        for r in tqdm(records, desc="Processing", unit="vol")
    )

    for rec, res in zip(records, results):
        res["dataset"]  = rec["dataset"]
        res["modality"] = rec["modality"]

    new_df = pd.DataFrame(results)

    csv_path = out_dir / "results.csv"
    if csv_path.exists() and completed:
        df = pd.concat([pd.read_csv(csv_path), new_df], ignore_index=True)
    else:
        df = new_df

    df      = classify(df, CONFIG)
    summary = save_results(df, out_dir)

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print("  {:<16}: {}".format(k, v))
    print("\nResults saved to: {}".format(out_dir))


# ---------------------------------------------------------------------------
# ENTRYPOINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect defaced NIfTI volumes using pydeface diff")
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--out_dir",   type=Path, default=Path("./pydeface_results"))
    parser.add_argument("--n_jobs",    type=int,  default=4)
    parser.add_argument("--limit",     type=int,  default=None)
    parser.add_argument("--sample",    type=int,  default=None)
    parser.add_argument("--dataset",   type=str,  default=None,
                        help="Process only one dataset folder under data_root")
    args = parser.parse_args()

    main(args.data_root, args.out_dir, args.n_jobs, args.limit, args.sample, args.dataset)