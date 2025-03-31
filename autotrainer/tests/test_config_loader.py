import unittest
from autotrainer import ConfigLoader, ConfigValidationError

class TestConfigLoader(unittest.TestCase):

    def test_valid_minimal_config(self):
        config = {
            "experiments": {
                "exp1": {
                    "script": "train.py"
                }
            },
            "datasets": {
                "ds1": {
                    "root": "/data",
                    "experiments": [
                        {"name": "exp1"}
                    ]
                }
            }
        }
        loader = ConfigLoader(config)
        validated = loader.load()
        self.assertEqual(validated["datasets"]["ds1"]["root"], "/data")

    def test_missing_experiments_key(self):
        config = {"datasets": {}}
        loader = ConfigLoader(config)
        with self.assertRaises(ConfigValidationError) as cm:
            loader.load()
        self.assertIn("'experiments' section", str(cm.exception))

    def test_missing_dataset_root(self):
        config = {
            "experiments": {"exp1": {"script": "train.py"}},
            "datasets": {
                "ds1": {
                    "experiments": [
                        {"name": "exp1"}
                    ]
                }
            }
        }
        loader = ConfigLoader(config)
        with self.assertRaises(ConfigValidationError) as cm:
            loader.load()
        self.assertIn("missing required field: 'root'", str(cm.exception))

    def test_undefined_experiment_reference(self):
        config = {
            "experiments": {"exp1": {"script": "train.py"}},
            "datasets": {
                "ds1": {
                    "root": "/data",
                    "experiments": [
                        {"name": "exp2"}
                    ]
                }
            }
        }
        loader = ConfigLoader(config)
        with self.assertRaises(ConfigValidationError) as cm:
            loader.load()
        self.assertIn("references undefined experiment", str(cm.exception))

    def test_param_set_conflict(self):
        config = {
            "experiments": {
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
                    "experiments": [
                        {"name": "exp1"}
                    ]
                }
            }
        }
        loader = ConfigLoader(config)
        with self.assertRaises(ConfigValidationError) as cm:
            loader.load()
        self.assertIn("appear in both static fields and param_set", str(cm.exception))

if __name__ == "__main__":
    unittest.main()
