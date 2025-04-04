"""
Created on Wednesday, 2 April 2025.

@authors:
* Austin Dibble, University of Glasgow

----------------------------------------------
ResultManager - Job Artifact Management Engine
----------------------------------------------

An extendable class for emitting and collecting
job results/artifacts.


Example:
    manager = ResultManager("results", "{task_name}/{dataset_name}/{job_id}.yaml", fmt='yaml')
    result = manager.create_and_emit_result(job, results={"accuracy": 0.9}, status="success")
    results = manager.collect_results()

"""

import json
import yaml
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union
import re
from string import Formatter

from loguru import logger

class ResultManager:
    """
    Manages writing and reading experiment results in a configurable and consistent format.

    Responsibilities:
    - Emit result files for individual jobs using a structured schema
    - Resolve output paths using a configurable format string
    - Collect and aggregate result files from a directory (JSON/YAML)
    
    Output Pattern:
        You can specify a pattern using Python format string placeholders:
            - {job_id} (required)
            - {dataset_name} (optional)
            - {task_name} (optional)

        Example:
            out_dir = "results"
            output_pattern = "{task_name}/{dataset_name}/{job_id}.json"
            -> results/finetune/dataset_alpha/job123.json

    Collection:
        collect_results will recursively find all the yielded .json or .yml
        results in the given root_dir. If an output_pattern was given, then 
        that will be used to infer the keys "dataset_name" and "task_name" on 
        the returned objects if desired.

    Args:
        output_pattern (str): Pattern used to generate output file paths.
                              Defaults to "{job_id}.json"
        root_dir (str): Directory to place or search for results. Default is 'results'
        fmt (str): Format to write result files in — either 'json' or 'yaml'.

    Note:
        You may use any placeholder in output_pattern — e.g.:
        output_pattern='by_exp/{task_name}/{dataset_name}/{job_id}.yaml'

    Example:
        manager = ResultManager("results", "{task_name}/{dataset_name}/{job_id}.yaml", fmt='yaml')
        result = manager.create_and_emit_result(job, results={"accuracy": 0.9}, status="success")
        results = manager.collect_results()
    """

    def __init__(self, root_dir:Union[str, Path], output_pattern: str = "{job_id}.json", fmt: str = "json"):
        self.output_pattern = output_pattern
        self.root_dir = Path(root_dir)
        self.fmt = fmt.lower()
        if self.fmt not in {"json", "yaml"}:
            raise ValueError("Format must be 'json' or 'yaml'.")
        self._validate_output_pattern()

    def _validate_output_pattern(self):
        allowed_keys = {"job_id", "dataset_name", "task_name"}
        used_keys = set(re.findall(r"{([^}]+)}", self.output_pattern))

        unknown_keys = used_keys - allowed_keys
        if unknown_keys:
            raise ValueError(
                f"Invalid output_pattern: unknown placeholder(s) {sorted(unknown_keys)}.\n"
                f"Allowed keys are: {sorted(allowed_keys)}"
            )
        
        if "job_id" not in used_keys:
            raise ValueError(
                "Invalid output_pattern: '{job_id}' is required in the pattern "
                "to ensure job outputs are uniquely identifiable."
            )
        
    def create_and_emit_result(
        self,
        job: dict,
        results:dict,
        status:str = "success",
        extra: Optional[Dict] = None,
        output_path: Optional[Union[str, Path]] = None
    ) -> str:
        """
        Create and emit a result object for output according to our recommended schema.

        Args:           
            job (dict): The job spec.
            results (dict): Custom dictionary of results to report. Can be metrics, or something else.
            status (str): End job status, like "success" or "error"
            extra (dict): Optional metadata keys to add onto the result object.
            output_path (str|Path): Optional output path override to emit the result
            
        Returns:
            result object (dict)
        """
        result = self.create_result(job, results, status, extra)
        return self.emit_result(result, job, output_path)
        
    def create_result(self, job: dict, results: dict, status: str = "success", extra: Optional[Dict] = None) -> dict:
        """
        Create a result object for output according to our recommended schema. Can be emitted via `emit_result`

        Args:           
            job (dict): The job spec.
            results (dict): Custom dictionary of results to report. Can be metrics, or something else.
            status (str): End job status, like "success" or "error"
            extra (dict): Optional metadata keys to add onto the result object.
            
        Returns:
            result object (dict)
        """
        result = {
            "job_id": job["job_id"],
            "status": status,
            "results": results,
            "params": job.get("params", {}),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if extra:
            result.update(extra)
        return result
        
    def emit_result(
        self,
        result: dict,
        job: Optional[dict] = None,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """
        Emit a result file for a job.

        Args:           
            result (dict): Dictionary of results. Can be created via `create_result` or according to a custom schema.
            job (dict): The job spec.
            output_path (str): Optional manual override of output path.

        Returns:
            Path: Path to the written result file.
        """

        if output_path is None and job is None:
            raise ValueError("Either output_path or job object must be given to resolve the output location.")

        path = Path(output_path or self._resolve_output_path(job))
        path.parent.mkdir(exist_ok=True, parents=True)

        with open(path, "w") as f:
            if self.fmt == "json":
                json.dump(result, f, indent=2)
            else:
                yaml.safe_dump(result, f)

        return path

    def collect_results(self, root_dir: Optional[Union[str, Path]] = None, infer_metadata:bool=True) -> List[Dict]:
        """
        Recursively collect all result files under the specified output directory.
        Metadata for dataset_name and task_name can be inferred if the `output_pattern`
        on this class instance was set, and will be added to the yielded objects. To 
        do this, use `infer_metadata=True`.

        Args:
            root_dir (str or Path): Directory to start scanning from.
                Defaults to the configured root_dir. If overridden, metadata inference will be skipped.
            infer_metadata (bool): Whether to infer the metadata keys from the path pattern.

        Returns:
            List[dict]: List of parsed result dictionaries, enriched with inferred metadata when possible.
        """
        root = Path(root_dir or self.root_dir)
        results = []

        if root_dir is not None and infer_metadata and root != self.root_dir:
            logger.warning('collect_results root_dir is set and infer_metadata is True. If you desire metadata to be inferred, the root must match the original specified for the class instance!')

        for file in root.rglob("*"):
            if not file.is_file():
                continue
            if file.suffix not in {".json", ".yaml", ".yml"}:
                continue

            try:
                with open(file, "r") as f:
                    if file.suffix == ".json":
                        data = json.load(f)
                    else:
                        data = yaml.safe_load(f)

                if (root_dir is None or (root == self.root_dir)) and infer_metadata:
                    metadata = self._infer_metadata_from_path(file)
                    data.update(metadata)

                results.append(data)
            except Exception as e:
                raise ValueError(
                    f"Failed to parse result file: {file}\n"
                    f"Ensure the file matches your output_pattern structure if using default root_dir.\n"
                    f"Error: {e}"
                )

        return results


    def _resolve_output_path(self, job: dict) -> str:
        """
        Format the output path from the job spec and output pattern.

        Returns:
            str: Resolved file path.
        """
        rel_path = self.output_pattern.format(
            job_id=job["job_id"],
            dataset_name=job.get("dataset_name", "unknown_dataset"),
            task_name=job.get("task_name", "unknown_experiment")
        )
        return Path(self.root_dir) / rel_path
    

    def _infer_metadata_from_path(self, full_path: Path) -> Dict[str, str]:
        """
        Infers job metadata (like experiment_name, dataset_name, job_id)
        by parsing the full file path using the output_pattern.

        Args:
            full_path (Path): The full path to a result file.

        Returns:
            dict: Dictionary of inferred values from path (e.g., {"job_id": "...", "dataset_name": "..."}).

        Raises:
            ValueError: If the file path doesn't match the pattern structure.
        """
        # Strip root_dir from path
        try:
            rel_path = full_path.relative_to(self.root_dir)
        except ValueError:
            raise ValueError(
                f"Result file '{full_path}' is not under the output directory '{self.root_dir}'"
            )

        # Remove file extension for comparison
        rel_parts = rel_path.with_suffix('').parts
        pattern_parts = Path(self.output_pattern).with_suffix('').parts

        if len(rel_parts) != len(pattern_parts):
            raise ValueError(
                f"File path '{rel_path}' does not match the expected output pattern structure: "
                f"{self.output_pattern}"
            )

        # Parse pattern into literal + placeholder structure
        parsed = list(Formatter().parse(str(Path(self.output_pattern))))
        inferred = {}

        for (path_part, pattern_part) in zip(rel_parts, parsed):
            literal, field_name, *_ = pattern_part
            if field_name:  # it's a placeholder
                inferred[field_name] = path_part
            elif literal and path_part != literal:
                raise ValueError(
                    f"Literal mismatch in path '{rel_path}': expected '{literal}', got '{path_part}'"
                )

        return inferred

