"""
Created on Friday, 28 March 2025.

@authors:
* Austin Dibble, University of Glasgow

---------------------------------
JobCreator - Job Expansion Engine
---------------------------------

Purpose:
    Given a validated experiment config (from ConfigLoader), expand all experiment + dataset combinations
    into fully-resolved, atomic job specifications ready for execution or scheduling.

Job Output Format:
    Each job is represented as a dictionary with the following fields:

    {
        "job_id": str,            # unique string, e.g., dataset_experiment_idx
        "experiment_name": str,  # name of experiment (from global definition)
        "dataset_name": str,     # name of dataset
        "params": dict,          # fully-resolved parameter dictionary
        "data_root": str         # dataset root path
    }

High-Level Algorithm:
    For each dataset:
        For each experiment listed under the dataset:
            Look up the global experiment definition
            Merge static params + dataset-level override + entry-level override + param_set (entry > dataset > global)
            If param_set is defined:
                Expand into all combinations (cartesian product)
            For each param combo:
                Create job dict with resolved fields and metadata
                Generate unique job_id

"""

from typing import List, Dict
import itertools
from .config_loader import ConfigLoader

from loguru import logger

class JobCreator:
    def __init__(self, validated_config: dict):
        """
        Initialize the job creator with a validated config.

        Args:
            validated_config (dict): Output of ConfigLoader.load()
        """
        self.config = validated_config
        self.jobs: List[Dict] = []
        self.job_counter = 0  # used for generating unique job IDs

    def create(self) -> List[dict]:
        """
        Expand the full set of jobs from the validated config.

        Returns:
            List[dict]: List of fully-resolved job dictionaries.
        """
        self.jobs = self._expand_jobs()
        return self.jobs

    def _expand_jobs(self) -> List[dict]:
        """
        Internal function that generates all job dicts by resolving datasets and experiments.
        """
        jobs = []
        datasets = self.config['datasets']
        experiments = self.config['experiments']

        for dataset_name, dataset_info in datasets.items():
            ds_override = dataset_info.get('override', {})
            ds_param_set = dataset_info.get('param_set')
            ds_static = {
                k: v for k, v in dataset_info.items()
                if k not in ConfigLoader._g_inbuilt_keywords
            }
            
            for exp_entry in dataset_info['experiments']:
                exp_name = exp_entry['name']
                base_exp = experiments[exp_name]

                entry_override = exp_entry.get("override", {})
                entry_param_set = exp_entry.get('param_set')
                entry_static = {
                    k: v for k, v in exp_entry.items()
                    if k not in ConfigLoader._g_inbuilt_keywords
                }

                param_combos = self._expand_param_set(
                    base_exp.get('param_set'),
                    ds_param_set,
                    entry_param_set
                )

                # logger.debug(param_combos)

                for combo in param_combos:
                    params = self._resolve_params(
                        base=base_exp,
                        dataset_override=ds_override,
                        entry_override=entry_override,
                        dataset_static=ds_static,
                        entry_static=entry_static,
                        param_combo=combo
                    )

                    job = {
                        "job_id": self._generate_job_id(dataset_name, exp_name),
                        "experiment_name": exp_name,
                        "dataset_name": dataset_name,
                        "params": params,
                        "data_root": dataset_info['root']
                    }
                    jobs.append(job)
        return jobs

    def _expand_param_set(self, global_ps: dict, dataset_ps: dict, entry_ps: dict) -> List[dict]:
        """
        Compute all param combinations from the appropriate param_set.

        Args:
            global_ps (dict): global param_set from experiment definition
            dataset_ps (dict): dataset-level param_set
            entry_ps (dict): param_set from dataset experiment entry

        Returns:
            List[dict]: list of param dictionaries
        """
        merged = {}
        for level in (global_ps, dataset_ps, entry_ps):
            if level:
                merged.update(level)

        if not merged:
            return [{}]  # single variant: no param sweep

        keys, values = zip(*merged.items())
        combos = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
        return combos

    def _resolve_params(self, base: dict, dataset_override: dict, entry_override: dict,
                        dataset_static: dict, entry_static: dict, param_combo: dict) -> dict:
        """
        Merge static parameters with overrides and param_set combo.

        Args:
            base (dict): base experiment definition
            dataset_override (dict): dataset-level overrides
            entry_override (dict): entry-level overrides
            dataset_static (dict): dataset-level static params
            entry_static (dict): entry-level static params
            param_combo (dict): one resolved variant from param_set

        Returns:
            dict: combined param dictionary
        """
        reserved = ConfigLoader._g_inbuilt_keywords
        static = {k: v for k, v in base.items() if k not in reserved}
        return {
            **static,
            **dataset_static,
            **entry_static,
            **dataset_override,
            **entry_override,
            **param_combo
        }

    def _generate_job_id(self, dataset_name: str, experiment_name: str) -> str:
        """
        Generate a unique job identifier.
        """
        job_id = f"{dataset_name}_{experiment_name}_{self.job_counter}"
        self.job_counter += 1
        return job_id
