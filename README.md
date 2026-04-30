# Brain: Brain Imaging and Artificial Intelligence Research Lab — shared library

This is the central utility library for our lab's codebase. It is not a standalone project -- it exists as a shared dependency across several of our repositories, providing common utilities for neuroimaging and AI research (e.g. data augmentation, preprocessing helpers).

---

## Related Projects

This library is used as a dependency in the following projects:

| Repository | Role |
|---|---|
| [TheOneSeg](https://github.com/rockNroll87q/TheOneSeg) | Segmentation project; includes `brain` as a git submodule |
| [NeuroFM-training](https://github.com/rockNroll87q/NeuroFM-training) | Training code for [NeuroFM](https://rocknroll87q.github.io/NeuroFM/); uses `brain` during development |

---

## Installation

### As a submodule (recommended)

If you are cloning a project that already includes `brain` as a submodule (e.g. TheOneSeg), clone with:

```bash
git clone --recurse-submodules git@github.com:rockNroll87q/TheOneSeg.git
```

### Standalone

```bash
git clone git@github.com:rockNroll87q/brain.git
```

## Quick Usage

```python
from brain.augmentation import Augmenter
```

