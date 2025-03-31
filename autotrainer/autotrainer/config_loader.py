"""
Created on Friday, 28 March 2025.

@authors:
* Austin Dibble, University of Glasgow


-------------------------------------
YAML Experiment Protocol Specification in Brief
-------------------------------------

Purpose:
    Define a structured YAML format to run deep learning experiments across multiple datasets,
    allowing parameterized variants and per-dataset overrides.

Structure:

1. experiments:
    A dictionary of globally defined experiment templates.
    Each experiment includes:
        - static parameters (epochs, lr, etc.)
        - param_set (optional): keys with lists of values to sweep

    Example:
    experiments:
      finetune_all:
        script: train.py
        epochs: 50
        learning_rate: 1e-5
        finetune_all_layers: true

2. datasets:
    Dataset-specific job instructions. Each dataset includes:
        - root (str): base path to dataset
        - experiments: list of experiments to run
            - name: experiment key from global section
            - override (optional): static parameter overrides
            - param_set (optional): replaces global param_set

    Example:
    datasets:
      dataset_alpha:
        root: /mnt/data/alpha
        experiments:
          - name: finetune_all
            output_vars: [label1]
          - name: finetune_all
            output_vars: [label2]
            override:
              learning_rate: 2e-5
            param_set:
              finetune_depth: [1, 3]

Precedence:
    - override > local param_set > global param_set > static params
    - param_set expands to cartesian product of values
    - Conflicts between static and param_set keys raise validation errors

Validation Rules:
    - All experiment names in datasets must match defined experiments
    - param_set keys must not also appear as static params in the same block

-------------------------------------
Overview of this File (How to Use)
-------------------------------------




"""

import yaml
import itertools
from typing import List, Dict, Union
from pathlib import Path

class ConfigValidationError(Exception):
    """Raised when the configuration is invalid."""
    pass

class ConfigLoader:
    # In our overlap and param checks, these are the built-in keywords which cannot be used as user-defined variables.
    _g_inbuilt_keywords = {"param_set",\
                            "name", "output_vars", "override", "experiments", "root"}


    def __init__(self, config_source: Union[str, Path, Dict]):
        """
        Initializes the configuration loader.

        Args:
            config_source (str or dict): Path to a YAML config file or a pre-parsed dictionary.
        """
        self._source = config_source
        self._raw_config = None
        self._validated_config = None

    def load(self) -> dict:
        """
        Load and validate the configuration.

        Returns:
            dict: A validated and normalized configuration dictionary.

        Raises:
            ConfigValidationError: If validation fails.
        """
        self._raw_config = self._load_source()
        self._validated_config = self._validate(self._raw_config)
        return self._validated_config

    def _load_source(self) -> dict:
        """
        Load the configuration from file or return directly if already a dict.

        Returns:
            dict: Raw configuration dictionary.

        Raises:
            TypeError: If input is not a str or dict.
        """
        if isinstance(self._source, str) or isinstance(self._source, Path):
            with open(self._source, 'r') as f:
                return yaml.safe_load(f)
        elif isinstance(self._source, dict):
            return self._source
        else:
            raise TypeError("Config source must be a file path or a dictionary.")

    def _validate(self, config: dict) -> dict:
        """
        Validate and normalize the configuration.

        Args:
            config (dict): Raw configuration dictionary.

        Returns:
            dict: Normalized configuration dictionary.

        Raises:
            ConfigValidationError: If validation fails.
        """
        if not isinstance(config, dict):
            raise ConfigValidationError("Config must be a dictionary.")

        experiments = config.get("experiments")
        datasets = config.get("datasets")

        if not isinstance(experiments, dict):
            raise ConfigValidationError("Missing or invalid 'experiments' section. Must be a dictionary.")

        if not isinstance(datasets, dict):
            raise ConfigValidationError("Missing or invalid 'datasets' section. Must be a dictionary.")

        self._validate_experiments(experiments)

        for dataset_name, dataset_def in datasets.items():
            if not isinstance(dataset_def, dict):
                raise ConfigValidationError(f"Dataset '{dataset_name}' must be a dictionary.")
            if "root" not in dataset_def:
                raise ConfigValidationError(f"Dataset '{dataset_name}' is missing required field: 'root'.")
            if "experiments" not in dataset_def or not isinstance(dataset_def["experiments"], list):
                raise ConfigValidationError(f"Dataset '{dataset_name}' must have a list under 'experiments'.")

            # Check for param_set conflicts at the dataset level
            if "param_set" in dataset_def:
                self._check_param_conflicts(dataset_name, dataset_def)
        
            # Make sure the overrides, if present, are a dictionary object
            dataset_override = dataset_def.get('override', {})
            if not isinstance(dataset_override, dict):
                raise ConfigValidationError(f"In '{dataset_name}', 'override' must be a dictionary.")

            for i, exp_entry in enumerate(dataset_def["experiments"]):
                if not isinstance(exp_entry, dict):
                    raise ConfigValidationError(f"Experiment entry #{i} in dataset '{dataset_name}' must be a dictionary.")
                name = exp_entry.get("name")
                if name not in experiments:
                    raise ConfigValidationError(f"Dataset '{dataset_name}' references undefined experiment '{name}'.")
                global_exp = experiments[name]

                if "param_set" in exp_entry:
                    self._check_param_conflicts(name, exp_entry)

                exp_override = exp_entry.get('override', {})
                if not isinstance(exp_override, dict):
                    raise ConfigValidationError(f"In '{dataset_name}:{name}', 'override' must be a dictionary.")
                
                # Check user variables of this experiment listing under the dataset against the global experiment def
                conflicts = self._check_user_var_conflicts(global_exp, exp_entry)
                if conflicts:
                    raise ConfigValidationError(f"In {dataset_name}:{name}, user-provided variables [{', '.join(conflicts)}] \
conflict with those given in the experiment definition. Use override if required.")
                
                # Check the user variables of the broader dataset def against the global experiment def
                conflicts = self._check_user_var_conflicts(global_exp, dataset_def)
                if conflicts:
                    raise ConfigValidationError(f"In {dataset_name}, user-provided variables [{', '.join(conflicts)}] \
conflict with those given in the experiment definition '{name}'. Use override if required.")
                
                # Check the user variables of the dataset-experiment def against the dataset def
                conflicts = self._check_user_var_conflicts(dataset_def, exp_entry)
                if conflicts:
                    raise ConfigValidationError(f"In {dataset_name}:{name}, user-provided variables [{', '.join(conflicts)}] \
conflict with those given in the dataset definition '{dataset_name}'. Use override if required.")

        return config
    
    def _validate_experiments(self, experiments:dict):
        for exp_name, exp_def in experiments.items():
            if not isinstance(exp_def, dict):
                raise ConfigValidationError(f"Experiment '{exp_name}' must be a dictionary.")
            if "param_set" in exp_def:
                self._check_param_conflicts(exp_name, exp_def)

    def _check_param_conflicts(self, scope_name: str, exp_def: dict):
        """
        Ensure no overlap between static params and param_set keys.

        Args:
            scope_name (str): Name of the experiment or dataset+experiment scope
            exp_def (dict): The definition containing static keys and param_set

        Raises:
            ConfigValidationError: If conflicts are found.
        """
        static_keys = ConfigLoader._get_object_user_vars(exp_def)
        param_set = exp_def.get("param_set", {})
        if not isinstance(param_set, dict):
            raise ConfigValidationError(f"In '{scope_name}', 'param_set' must be a dictionary.")
        conflicts = static_keys & set(param_set.keys())
        if conflicts:
            raise ConfigValidationError(
                f"In '{scope_name}', the following keys appear in both static fields and param_set: {sorted(conflicts)}"
            )
    
    def _check_user_var_conflicts(self, global_def:dict, local_def:dict):
        """
        Get the overlapping variables between some user definition (like in a dataset definition, or dataset-experiment)
        and the more global-level variables from the global experiment definition.

        Args:
            global_def (dict): Object from the experiment's definition
            local_def (dict): dataset or dataset-experiment object.

        Returns:
            a set of overlapping keys, if any
        """
        global_var_keys = ConfigLoader._get_object_user_vars(global_def)
        exp_keys = ConfigLoader._get_object_user_vars(local_def)
        return global_var_keys & exp_keys

    @staticmethod
    def _get_object_user_vars(obj: dict):
        return set(obj.keys()) - ConfigLoader._g_inbuilt_keywords

    @property
    def raw(self) -> dict:
        """Access the raw configuration before validation."""
        return self._raw_config

    @property
    def validated(self) -> dict:
        """Access the validated and normalized configuration."""
        return self._validated_config
