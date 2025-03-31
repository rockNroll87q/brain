import unittest
from autotrainer import ConfigLoader, JobCreator

class TestJobCreator(unittest.TestCase):

    def load_jobs(self, config):
        loader = ConfigLoader(config)
        validated = loader.load()
        creator = JobCreator(validated)
        return creator.create()

    def test_single_job_no_param_set(self):
        config = {
            "experiments": {
                "exp1": {"script": "train.py", "epochs": 10}
            },
            "datasets": {
                "ds1": {
                    "root": "/data",
                    "experiments": [{"name": "exp1"}]
                }
            }
        }
        jobs = self.load_jobs(config)
        self.assertEqual(len(jobs), 1)
        job = jobs[0]
        self.assertEqual(job["job_id"], "ds1_exp1_0")
        self.assertEqual(job["params"], {"epochs": 10})

    def test_global_param_set_expansion(self):
        config = {
            "experiments": {
                "exp1": {
                    "script": "train.py",
                    "epochs": 20,
                    "param_set": {
                        "layers": [1, 2],
                        "dropout": [0.1, 0.2]
                    }
                }
            },
            "datasets": {
                "ds1": {
                    "root": "/data",
                    "experiments": [{"name": "exp1"}]
                }
            }
        }
        jobs = self.load_jobs(config)
        self.assertEqual(len(jobs), 4)
        expected_combinations = [
            {"epochs": 20, "layers": 1, "dropout": 0.1},
            {"epochs": 20, "layers": 1, "dropout": 0.2},
            {"epochs": 20, "layers": 2, "dropout": 0.1},
            {"epochs": 20, "layers": 2, "dropout": 0.2},
        ]
        for i, job in enumerate(jobs):
            self.assertEqual(job["params"], expected_combinations[i])
            self.assertEqual(job["job_id"], f"ds1_exp1_{i}")

    def test_local_param_set_overrides_global(self):
        config = {
            "experiments": {
                "exp1": {
                    "script": "train.py",
                    "epochs": 30,
                    "param_set": {"layers": [1, 2]}
                }
            },
            "datasets": {
                "ds1": {
                    "root": "/data",
                    "experiments": [{
                        "name": "exp1",
                        "param_set": {"layers": [5]}
                    }]
                }
            }
        }
        jobs = self.load_jobs(config)
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["params"], {"epochs": 30, "layers": 5})

    def test_dataset_override_applies(self):
        config = {
            "experiments": {
                "exp1": {"script": "train.py", "epochs": 50, "lr": 1e-3}
            },
            "datasets": {
                "ds1": {
                    "root": "/data",
                    "experiments": [{
                        "name": "exp1",
                        "override": {"lr": 1e-4}
                    }]
                }
            }
        }
        jobs = self.load_jobs(config)
        self.assertEqual(jobs[0]["params"], {"epochs": 50, "lr": 1e-4})

    def test_multiple_experiments_and_datasets(self):
        config = {
            "experiments": {
                "exp1": {"script": "a.py", "epochs": 10},
                "exp2": {"script": "b.py", "epochs": 20}
            },
            "datasets": {
                "ds1": {
                    "root": "/a",
                    "experiments": [{"name": "exp1"}, {"name": "exp2"}]
                },
                "ds2": {
                    "root": "/b",
                    "experiments": [{"name": "exp2"}]
                }
            }
        }
        jobs = self.load_jobs(config)
        self.assertEqual(len(jobs), 3)
        job_ids = [j["job_id"] for j in jobs]
        self.assertEqual(sorted(job_ids), ["ds1_exp1_0", "ds1_exp2_1", "ds2_exp2_2"])

    def test_job_with_required_output_vars(self):
        config = {
            "experiments": {
                "exp1": {
                    "script": "train.py",
                    "epochs": 10,
                    "requires_output_var": True
                }
            },
            "datasets": {
                "ds1": {
                    "root": "/data",
                    "experiments": [{
                        "name": "exp1",
                        "output_vars": ["label1", "label2"]
                    }]
                }
            }
        }
        jobs = self.load_jobs(config)
        self.assertEqual(jobs[0]["output_vars"], ["label1", "label2"])

if __name__ == "__main__":
    unittest.main()
