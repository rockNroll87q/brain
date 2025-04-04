"""
Created on Wednesday, 2 April 2025.

@authors:
* Austin Dibble, University of Glasgow

------------------------
Full Pipeline Example
------------------------

"""

import sys
import os

# Script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
# Add project root to sys.path
ba_path = os.path.join(script_dir, '../')
sys.path.append(os.path.abspath(ba_path))

import random
import pandas as pd
from multiprocessing import Manager
from autotrainer import ConfigLoader, JobCreator, JobRunner
from autotrainer.aggregation import aggregate_results
from autotrainer.result_manager import ResultManager, create_result

# ------------------------------
# Example config (could be YAML)
# ------------------------------
example_config = {
    "tasks": {
        "finetune": {
            "script": "train.py",
            "epochs": 10,
            "param_set": {
                "lr": [0.001, 0.0005],
                "dropout": [0.1, 0.2]
            }
        }
    },
    "datasets": {
        "dataset_alpha": {
            "root": "/mnt/data/alpha",
            "tasks": [
                {"name": "finetune"}
            ]
        }
    }
}

# ---------------------------------
# Custom JobRunner with result logic
# ---------------------------------
class SimulatedRunner(JobRunner):
    def __init__(self, jobs, result_store, **kwargs):
        super().__init__(jobs, **kwargs)
        self.result_store = result_store

    def run_one(self, job: dict):
        # Simulate evaluation
        simulated_metrics = {
            "accuracy": round(random.uniform(0.85, 0.95), 3),
            "f1": round(random.uniform(0.80, 0.90), 3)
        }

        result_obj = create_result(job, results=simulated_metrics)
        self.result_store.append(result_obj)  # thread-safe shared list


# -----------------------------
# Main pipeline (in-memory only)
# -----------------------------

# Step 1: Load config
loader = ConfigLoader(example_config)
validated_config = loader.load()

# Step 2: Create jobs
creator = JobCreator(validated_config)
jobs = creator.create()

with Manager() as manager:
    result_list = manager.list()

    # Step 4: Run jobs using multiprocessing
    runner = SimulatedRunner(
        jobs,
        result_store=result_list,
        max_workers=4  # Multiprocessing!
    )
    runner.run()

    # Step 5: Aggregate results into DataFrame
    df = aggregate_results(
        list(result_list),
        index_fields=["job_id", "task_name", "dataset_name"],
        auto_detect_metrics=True,
        auto_detect_params=True,
        strict=False
    )

print(df)
