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
        "script": str,           # training script path
        "params": dict,          # fully-resolved parameter dictionary
        "output_vars": list,     # optional; only if required by experiment
        "data_root": str         # dataset root path
    }

High-Level Algorithm:
    For each dataset:
        For each experiment listed under the dataset:
            Look up the global experiment definition
            Merge static params + override + param_set (local > global)
            If param_set is defined:
                Expand into all combinations (cartesian product)
            For each param combo:
                Create job dict with resolved fields and metadata
                Generate unique job_id

"""

from typing import List, Dict
import itertools

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
            for exp_entry in dataset_info['experiments']:
                exp_name = exp_entry['name']
                base_exp = experiments[exp_name]

                param_combos = self._expand_param_set(
                    base_exp.get('param_set'),
                    exp_entry.get('param_set')
                )

                for combo in param_combos:
                    params = self._resolve_params(
                        base_exp,
                        exp_entry.get('override', {}),
                        combo
                    )
                    job = {
                        "job_id": self._generate_job_id(dataset_name, exp_name),
                        "experiment_name": exp_name,
                        "dataset_name": dataset_name,
                        "script": base_exp['script'],
                        "params": params,
                        "output_vars": exp_entry.get('output_vars'),
                        "data_root": dataset_info['root']
                    }
                    jobs.append(job)
        return jobs

    def _expand_param_set(self, global_ps: dict, local_ps: dict) -> List[dict]:
        """
        Compute all param combinations from the appropriate param_set.

        Args:
            global_ps (dict): global param_set from experiment definition
            local_ps (dict): param_set from dataset experiment entry

        Returns:
            List[dict]: list of param dictionaries
        """
        ps = local_ps if local_ps is not None else global_ps
        if not ps:
            return [{}]  # single variant: no param sweep

        keys, values = zip(*ps.items())
        combos = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
        return combos

    def _resolve_params(self, base: dict, override: dict, param_combo: dict) -> dict:
        """
        Merge static parameters with override and param_set combo.

        Args:
            base (dict): base experiment definition
            override (dict): user-defined overrides from dataset config
            param_combo (dict): one resolved variant from param_set

        Returns:
            dict: combined param dictionary
        """
        reserved_keys = {'script', 'requires_output_var', 'param_set', 'description'}
        static_params = {k: v for k, v in base.items() if k not in reserved_keys}
        merged = {**static_params, **override, **param_combo}
        return merged

    def _generate_job_id(self, dataset_name: str, experiment_name: str) -> str:
        """
        Generate a unique job identifier.
        """
        job_id = f"{dataset_name}_{experiment_name}_{self.job_counter}"
        self.job_counter += 1
        return job_id
