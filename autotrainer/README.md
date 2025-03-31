# AutoTrainer

**AutoTrainer** is a lightweight, extensible framework for defining and managing deep learning experiments across multiple datasets using simple YAML configuration files. It provides a consistent protocol for specifying experiments, datasets, parameter sweeps, and overrides — and produces fully-specified job definitions ready for execution.

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
experiments:
  finetune:
    script: train.py
    epochs: 50
    learning_rate: 1e-4
    requires_output_var: true

datasets:
  my_dataset:
    root: /data/my_dataset
    experiments:
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

---

## 🧠 Features

✅ Simple, declarative YAML experiment definitions  
✅ Dataset-specific overrides and parameter sweeps  
✅ Automatic expansion into per-job configurations  
✅ Platform-agnostic — you plug in the execution backend (Slurm, subprocess, etc.)  
✅ Built-in validation with informative error messages

---

## 🔍 Examples

Explore the `examples/` directory for working YAML templates:

- `basic.yml`: One experiment on one dataset
- `param_sweep.yml`: Parameterized experiment with a `param_set`
- `overrides.yml`: Per-dataset overrides of experiment parameters
- `multi_dataset.yml`: Multiple datasets sharing and customizing experiments

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
