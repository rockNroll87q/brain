"""
Created on Wednesday, 2 April 2025.

@authors:
* Austin Dibble, University of Glasgow

---------------------------------
Utility for aggregating job results
---------------------------------

example:
    df = aggregate_results(
        results,
        index_fields=['job_id', 'task_name'],
        auto_detect_metrics=True,
        param_fields=['lr', 'dropout']
    )
"""

import pandas as pd
from typing import List, Dict, Optional


def aggregate_results(
    results: List[Dict],
    index_fields: Optional[List[str]] = None,
    metric_fields: Optional[List[str]] = None,
    param_fields: Optional[List[str]] = None,
    auto_detect_metrics: bool = False,
    auto_detect_params: bool = False,
    strict: bool = True
) -> pd.DataFrame:
    """
    Aggregates a list of result dictionaries into a structured pandas DataFrame.

    This utility flattens each result's top-level fields, along with the nested 'results'
    and 'params' dictionaries (if present), and merges them into a table for comparison
    across jobs or configurations.

    Args:
        results (List[Dict]): List of result dictionaries (as collected from ResultManager).
        index_fields (List[str], optional): Top-level keys to include as identifying columns.
            Example: ['job_id', 'dataset_name', 'task_name']
        metric_fields (List[str], optional): Keys to extract from each result['results'] dictionary.
            If None and auto_detect_metrics=True, keys will be inferred across all results.
        param_fields (List[str], optional): Keys to extract from each result['params'] dictionary.
            If None and auto_detect_params=True, keys will be inferred across all results.
        auto_detect_metrics (bool): If True, extract metric fields by inspecting all results.
        auto_detect_params (bool): If True, extract param fields by inspecting all results.
        strict (bool): If True, raise informative errors on missing fields. If False, fill with NaN.

    Returns:
        pd.DataFrame: A DataFrame where each row corresponds to a result, and columns include
            selected index, metric, and param values.

    Raises:
        ValueError: If a required field is missing in strict mode, or input is malformed.

    Example:
        df = aggregate_results(
            results,
            index_fields=['job_id', 'task_name'],
            auto_detect_metrics=True,
            param_fields=['lr', 'dropout']
        )
    """
    if not results:
        raise ValueError("No results provided for aggregation.")

    # Auto-detect metric fields
    if metric_fields is None and auto_detect_metrics:
        detected = set()
        for i, r in enumerate(results):
            metrics = r.get("results")
            if isinstance(metrics, dict):
                detected.update(metrics.keys())
        if not detected:
            raise ValueError("Could not auto-detect metric fields: no valid 'results' dicts found.")
        metric_fields = sorted(detected)

    # Auto-detect param fields
    if param_fields is None and auto_detect_params:
        detected = set()
        for i, r in enumerate(results):
            params = r.get("params")
            if isinstance(params, dict):
                detected.update(params.keys())
        if not detected:
            raise ValueError("Could not auto-detect param fields: no valid 'params' dicts found.")
        param_fields = sorted(detected)

    rows = []
    for i, result in enumerate(results):
        if not isinstance(result, dict):
            raise ValueError(f"Result #{i} is not a dictionary (got type {type(result)}).")

        row = {}

        # Extract top-level fields (index)
        for field in (index_fields or []):
            if field in result:
                row[field] = result[field]
            elif strict:
                raise ValueError(f"Missing expected index field '{field}' in result #{i}.")
            else:
                row[field] = None

        # Extract from nested 'results'
        if metric_fields:
            metrics = result.get("results")
            if not isinstance(metrics, dict):
                if strict:
                    raise ValueError(f"Missing or invalid 'results' block in result #{i}.")
                else:
                    metrics = {}
            for key in metric_fields:
                row[key] = _get_field_with_check(metrics, key, "results", i, strict)

        # Extract from nested 'params'
        if param_fields:
            params = result.get("params")
            if not isinstance(params, dict):
                if strict:
                    raise ValueError(f"Missing or invalid 'params' block in result #{i}.")
                else:
                    params = {}
            for key in param_fields:
                row[key] = _get_field_with_check(params, key, "results", i, strict)

        rows.append(row)

    return pd.DataFrame(rows)

def _get_field_with_check(source: dict, key: str, section: str, index: int, strict: bool):
    if key in source:
        return source[key]
    elif strict:
        raise ValueError(f"Missing '{key}' in result['{section}'] for result #{index}")
    else:
        return None