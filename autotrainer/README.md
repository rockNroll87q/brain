# AutoTrainer

**AutoTrainer** is a lightweight, extensible framework for defining and managing deep learning tasks across multiple datasets using simple YAML configuration files. It provides a consistent protocol for specifying tasks, datasets, parameter sweeps, and overrides — and produces fully-specified job definitions ready for execution.

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
├── __init__.py             # Exposes main classes to the package root
├── spec.md                 # Markdown version of the YAML protocol specification
└── README.md               # This file
```

---

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

---

## 🧠 Features

✅ Simple, declarative YAML task definitions  
✅ Dataset-specific overrides and parameter sweeps  
✅ Automatic expansion into per-job configurations  
✅ Platform-agnostic — you plug in the execution backend (Slurm, subprocess, etc.)  
✅ Built-in validation with informative error messages

---

## 🔍 Examples

Explore the `examples/` directory for working YAML templates:

- `example1-basic.yml`: One task on one dataset
- `example2-sweep.yml`: Parameterized task with a `param_set`
- `example3-override.yml`: Per-dataset overrides of task parameters
- `example4-multi-dataset.yml`: Multiple datasets sharing and customizing tasks
- `example5-inheritance.yml`: Multi-level inheritance and overrides

Additionally, run any of the above using `examples/proc_example.py <path>` to see the produced job objects.

---

## 🧪 Running Tests

```bash
python -m unittest discover tests
```

---

## 📄 Spec

For a complete outline of the YAML protocol format, see [`spec.md`](spec.md).

---

**Author**: Austin Dibble / Brain Imaging and Artificial Intelligence Research Lab
**License**: 
