import unittest
from autotrainer import ConfigLoader, ConfigValidationError

import unittest
from autotrainer import _resolve_task_inheritance, InheritanceError

class TestExperimentInheritance(unittest.TestCase):

    def test_single_inheritance(self):
        tasks = {
            "base": {"epochs": 10, "learning_rate": 0.1},
            "child": {"extends": "base", "learning_rate": 0.01}
        }
        resolved = _resolve_task_inheritance(tasks)
        self.assertEqual(resolved["child"]["epochs"], 10)
        self.assertEqual(resolved["child"]["learning_rate"], 0.01)

    def test_multi_level_inheritance(self):
        tasks = {
            "base": {"epochs": 10, "lr": 0.1},
            "mid": {"extends": "base", "optimizer": "adam"},
            "final": {"extends": "mid", "lr": 0.001}
        }
        resolved = _resolve_task_inheritance(tasks)
        self.assertEqual(resolved["final"]["epochs"], 10)
        self.assertEqual(resolved["final"]["optimizer"], "adam")
        self.assertEqual(resolved["final"]["lr"], 0.001)

    def test_no_inheritance(self):
        tasks = {
            "plain": {"epochs": 20, "lr": 0.2}
        }
        resolved = _resolve_task_inheritance(tasks)
        self.assertEqual(resolved["plain"]["epochs"], 20)
        self.assertEqual(resolved["plain"]["lr"], 0.2)

    def test_undefined_base(self):
        tasks = {
            "child": {"extends": "nonexistent", "epochs": 5}
        }
        with self.assertRaises(InheritanceError) as ctx:
            _resolve_task_inheritance(tasks)
        self.assertIn("not defined", str(ctx.exception))

    def test_cycle_detection(self):
        tasks = {
            "a": {"extends": "b", "x": 1},
            "b": {"extends": "c", "y": 2},
            "c": {"extends": "a", "z": 3}
        }
        with self.assertRaises(InheritanceError) as ctx:
            _resolve_task_inheritance(tasks)
        self.assertIn("cycle detected", str(ctx.exception))

    def test_deep_override(self):
        tasks = {
            "base": {"x": 1, "y": 2, "z": 3},
            "child": {"extends": "base", "y": 20, "z": 30},
            "grandchild": {"extends": "child", "z": 300}
        }
        resolved = _resolve_task_inheritance(tasks)
        self.assertEqual(resolved["grandchild"]["x"], 1)
        self.assertEqual(resolved["grandchild"]["y"], 20)
        self.assertEqual(resolved["grandchild"]["z"], 300)

class TestConfigLoader(unittest.TestCase):

    def test_valid_minimal_config(self):
        config = {
            "tasks": {
                "exp1": {
                    "script": "train.py"
                }
            },
            "datasets": {
                "ds1": {
                    "root": "/data",
                    "tasks": [
                        {"name": "exp1"}
                    ]
                }
            }
        }
        loader = ConfigLoader(config)
        validated = loader.load()
        self.assertEqual(validated["datasets"]["ds1"]["root"], "/data")

    def test_missing_tasks_key(self):
        config = {"datasets": {}}
        loader = ConfigLoader(config)
        with self.assertRaises(ConfigValidationError) as cm:
            loader.load()
        self.assertIn("'tasks' section", str(cm.exception))

    def test_missing_dataset_root(self):
        config = {
            "tasks": {"exp1": {"script": "train.py"}},
            "datasets": {
                "ds1": {
                    "tasks": [
                        {"name": "exp1"}
                    ]
                }
            }
        }
        loader = ConfigLoader(config)
        with self.assertRaises(ConfigValidationError) as cm:
            loader.load()
        self.assertIn("missing required field: 'root'", str(cm.exception))

    def test_undefined_task_reference(self):
        config = {
            "tasks": {"exp1": {"script": "train.py"}},
            "datasets": {
                "ds1": {
                    "root": "/data",
                    "tasks": [
                        {"name": "exp2"}
                    ]
                }
            }
        }
        loader = ConfigLoader(config)
        with self.assertRaises(ConfigValidationError) as cm:
            loader.load()
        self.assertIn("references undefined task", str(cm.exception))

    def test_param_set_conflict(self):
        config = {
            "tasks": {
                "exp1": {
                    "script": "train.py",
                    "epochs": 10,
                    "param_set": {
                        "epochs": [5, 10]
                    }
                }
            },
            "datasets": {
                "ds1": {
                    "root": "/data",
                    "tasks": [
                        {"name": "exp1"}
                    ]
                }
            }
        }
        loader = ConfigLoader(config)
        with self.assertRaises(ConfigValidationError) as cm:
            loader.load()
        self.assertIn("appear in both static fields and param_set", str(cm.exception))

    def test_no_override_task(self):
        config = {
            "tasks": {
                "exp1": {
                    "script": "train.py",
                    "epochs": 50
                }
            },
            "datasets": {
                "ds1": {
                    "root": "/data",
                    "tasks": [
                        {
                            "name": "exp1",
                            "epochs": 20
                        }
                    ]
                }
            }
        }
        loader = ConfigLoader(config)
        with self.assertRaises(ConfigValidationError) as cm:
            loader.load()
        self.assertIn("conflict with those given in the task definition", str(cm.exception))

    def test_no_override_dataset(self):
        config = {
            "tasks": {
                "exp1": {
                    "script": "train.py",
                    "epochs": 50
                }
            },
            "datasets": {
                "ds1": {
                    "root": "/data",
                    "tasks": [
                        {
                            "name": "exp1",
                        }
                    ],
                    "epochs": 20
                }
            }
        }
        loader = ConfigLoader(config)
        with self.assertRaises(ConfigValidationError) as cm:
            loader.load()
        self.assertIn("conflict with those given in the task definition", str(cm.exception))

    def test_no_override_dataset_task(self):
        config = {
            "tasks": {
                "exp1": {
                    "script": "train.py",
                }
            },
            "datasets": {
                "ds1": {
                    "root": "/data",
                    "tasks": [
                        {
                            "name": "exp1",
                            "epochs": 50
                        }
                    ],
                    "epochs": 20
                }
            }
        }
        loader = ConfigLoader(config)
        with self.assertRaises(ConfigValidationError) as cm:
            loader.load()
        self.assertIn("conflict with those given in the dataset definition", str(cm.exception))

    def test_inheritance_resolves_correctly(self):
        config_dict = {
            "tasks": {
                "base": {
                    "script": "train.py",
                    "epochs": 20,
                    "learning_rate": 0.01
                },
                "child": {
                    "extends": "base",
                    "learning_rate": 0.001
                }
            },
            "datasets": {
                "ds1": {
                    "root": "/data/ds1",
                    "tasks": [
                        {"name": "child"}
                    ]
                }
            }
        }

        loader = ConfigLoader(config_dict)
        validated = loader.load()
        child_exp = validated["tasks"]["child"]

        self.assertEqual(child_exp["script"], "train.py")
        self.assertEqual(child_exp["epochs"], 20)
        self.assertEqual(child_exp["learning_rate"], 0.001)

    def test_inheritance_missing_parent_raises(self):
        config_dict = {
            "tasks": {
                "child": {
                    "extends": "missing_base",
                    "epochs": 10
                }
            },
            "datasets": {
                "ds1": {
                    "root": "/data/ds1",
                    "tasks": [
                        {"name": "child"}
                    ]
                }
            }
        }

        with self.assertRaises(InheritanceError) as ctx:
            ConfigLoader(config_dict).load()

        self.assertIn("missing_base", str(ctx.exception).lower())

if __name__ == "__main__":
    unittest.main()
