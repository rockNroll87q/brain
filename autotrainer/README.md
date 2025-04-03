# AutoTrainer

**AutoTrainer** is a lightweight, extensible framework for defining, running, and analyzing deep learning tasks across multiple datasets using simple YAML configuration files. It provides a consistent protocol for specifying tasks, datasets, parameter sweeps, and overrides — and produces fully-specified job definitions ready for execution.

## 📦 Project Structure

```
autotrainer/
├── autotrainer/            # Core logic (config loader, job creator, etc.)
│   ├── config_loader.py
│   ├── ...
│
├── examples/               # Example YAML configs
│   ├── example1-basic.yml
│   ├── example2-sweep.yml
│   └── ...
│
├── tests/                   # Unit tests
│   ├── test_config_loader.py
│   ├── test_job_creator.py
│
├── __init__.py             # Exposes main classes to the package
├── spec.md                 # Markdown version of the YAML protocol spec
└── README.md               # This file
```


## 🚀 Quick Start

1. **Install requirements** (if any):

```bash
pip install -r requirements.txt  # currently none strictly required
```

2. **Create a config file**:

```yaml
tasks:
  finetune:
    script: train.py
    epochs: 50
    learning_rate: 1e-4

datasets:
  my_dataset:
    root: /data/my_dataset
    tasks:
      - name: finetune
        output_vars: [label1, label2]
```

3. **Use the API**:

```python
from autotrainer import ConfigLoader, JobCreator

loader = ConfigLoader("examples/basic.yml")
validated_config = loader.load()

creator = JobCreator(validated_config)
jobs = creator.create()

for job in jobs:
    print(job)
```

Optionally, extend the `JobRunner` as a clean way to run the defined jobs:

```python
from autotrainer import JobRunner

class MyRunner(JobRunner):
    def run_one(self, job:dict):
        # Run your job script here!
        return

# Set max_workers > 1 for multiprocessing
runner = MyRunner(jobs, max_workers=1) 
runner.run()
```

## Features

- Simple, declarative YAML task definitions  
- Dataset-specific overrides and parameter sweeps  
- Automatic expansion into per-job configurations  
- Platform-agnostic — you plug in the execution backend (Slurm, subprocess, etc.)  
- Built-in validation with informative error messages
- Decoupled, logical class interfaces


## Examples

Explore the `examples/` directory for working YAML templates:

- `example1-basic.yml`: One task on one dataset
- `example2-sweep.yml`: Parameterized task with a `param_set`
- `example3-override.yml`: Per-dataset overrides of task parameters
- `example4-multi-dataset.yml`: Multiple datasets sharing and customizing tasks
- `example5-inheritance.yml`: Multi-level inheritance and overrides

Additionally, run any of the above using `examples/proc_example.py <path>` to see the produced job objects.


## 📄 Spec

For a complete outline of the definitions YAML protocol format, see [`spec.md`](spec.md).


## 📊 Result Creation & Aggregation

AutoTrainer also includes a flexible and optional result management system via the `ResultManager` class. This allows you to:

✅ Automatically emit per-job result files in `.json` or `.yaml`  
✅ Structure your output using configurable file path patterns  
✅ Collect and aggregate results across jobs  
✅ Infer job metadata from file paths (e.g., task name, dataset name)  

### 🔧 Emitting Results

```python
from autotrainer import ResultManager

# Define output layout and format
results = ResultManager(
    root_dir="results",
    output_pattern="{task_name}/{dataset_name}/{job_id}.json",
    fmt="json"
)

# Emit result for a single job
results.create_and_emit_result(
    job,
    results={"accuracy": 0.91, "f1": 0.88},
    status="success",
    extra={"notes": "trial_1"}
)
```

This creates a file like:
```
results/finetune/dataset_alpha/job123.json
```

### 📥 Collecting Results

```python
all_results = results.collect_results()
```

By default, any metadata that can be inferred from the file path (e.g., `task_name`, `dataset_name`) will be added back to each result object.

You can also opt-out of metadata inference, or override the root directory:

```python
results.collect_results(root_dir="custom_results", infer_metadata=False)
```

## Running Tests

```bash
python -m unittest discover tests
```

**Author**: Austin Dibble / Brain Imaging and Artificial Intelligence Research Lab
**License**: 
