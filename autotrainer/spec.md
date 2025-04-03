---

# 🧾 YAML Experiment Protocol Specification

---

## 🎯 Purpose

Define a set of deep learning tasks and apply them across multiple datasets, allowing for flexible parameterization, per-dataset overrides, and batch execution (e.g., on HPC/Slurm).

---

## 📐 Structure

### 🧪 `tasks:` (Global Experiment Definitions)

A dictionary where keys are **task names**, and values define:
- A training script
- Fixed parameters
- Optional parameter sweeps

```yaml
tasks:
  finetune_all:
    description: "Finetune all layers"
    script: train.py
    epochs: 50
    learning_rate: 1e-5
    finetune_all_layers: true

  depth_sweep:
    script: train.py
    epochs: 40
    learning_rate: 1e-4
    param_set:
      finetune_depth: [1, 3, 5, 10]
```

#### ✅ Rules:
- Keys must be unique task names
- If a param appears both as a static field and inside `param_set`, raise an error
- `param_set` defines parameters to sweep over (cartesian product)
- Optional fields: `description`, `requires_output_var`

---

### 📚 `datasets:` (Dataset Configurations)

Each dataset defines:
- Root data path
- A list of tasks to run
- Any required overrides or param sweeps specific to this dataset

```yaml
datasets:
  dataset_alpha:
    root: /mnt/data/alpha
    tasks:
      - name: finetune_all
        output_vars: [label1, label2]

      - name: depth_sweep
        output_vars: [label3]

      - name: finetune_all
        output_vars: [label4]
        override:
          learning_rate: 2e-5
        param_set:
          finetune_depth: [2, 4]
```

#### ✅ Rules:
- Each task entry **must** include a `name` matching a global task
- `override` allows static parameter overrides per run
- `param_set` **replaces** global `param_set` if defined locally

---

## 🧠 Resolution and Precedence Logic

| Level               | Behavior                                                              |
|---------------------|-----------------------------------------------------------------------|
| `override`          | Takes absolute precedence over everything                             |
| Local `param_set`   | Replaces global `param_set` entirely                                  |
| Global `param_set`  | Applied only if local `param_set` is not present                      |
| Global Static params| Used unless overridden or replaced by param_set                       |

---

## ❗ Validation Requirements

1. **Experiment names must match between global and dataset sections**
2. **No duplication of parameter keys between static fields and `param_set` in the same scope**
3. **Each param sweep is treated as a cartesian product unless explicitly changed (future support)**

---

## 🧪 Example of Full Config

```yaml
tasks:
  finetune_all:
    script: train.py
    epochs: 50
    learning_rate: 1e-5
    finetune_all_layers: true

  depth_sweep:
    script: train.py
    epochs: 40
    learning_rate: 1e-4
    param_set:
      finetune_depth: [1, 3, 5, 10]

datasets:
  dataset_alpha:
    root: /mnt/data/alpha
    tasks:
      - name: finetune_all
        output_vars: [label1, label2]

      - name: depth_sweep
        output_vars: [label3]

  dataset_beta:
    root: /mnt/data/beta
    tasks:
      - name: finetune_all
        output_vars: [label1]
        override:
          learning_rate: 2e-5
        param_set:
          finetune_depth: [2, 4]
```

---
