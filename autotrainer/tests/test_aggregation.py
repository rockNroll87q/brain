import unittest
import pandas as pd
from autotrainer import aggregate_results

class TestAggregateResults(unittest.TestCase):

    def setUp(self):
        self.results = [
            {
                "job_id": "job1",
                "task_name": "finetune",
                "dataset_name": "ds1",
                "results": {"accuracy": 0.91, "f1": 0.87},
                "params": {"lr": 0.001, "dropout": 0.2}
            },
            {
                "job_id": "job2",
                "task_name": "finetune",
                "dataset_name": "ds2",
                "results": {"accuracy": 0.88, "f1": 0.83},
                "params": {"lr": 0.002, "dropout": 0.1}
            }
        ]

        self.results_dirty = [
            {
                "job_id": "job1",
                "task_name": "finetune",
                "dataset_name": "ds1",
                "results": {"accuracy": 0.91},
                "params": {"lr": 0.001, "dropout": 0.2}
            },
            {
                "job_id": "job2",
                "task_name": "finetune",
                "dataset_name": "ds2",
                "results": {"accuracy": 0.88, "f1": 0.83},
                "params": {"dropout": 0.1}
            }
        ]

    def test_basic_index_extraction(self):
        df = aggregate_results(self.results, index_fields=["job_id", "task_name"])
        self.assertIn("job_id", df.columns)
        self.assertIn("task_name", df.columns)
        self.assertEqual(len(df), 2)
        self.assertEqual(df["job_id"].iloc[0], "job1")

    def test_auto_detect_metrics(self):
        df = aggregate_results(self.results, index_fields=["job_id"], auto_detect_metrics=True)
        self.assertIn("accuracy", df.columns)
        self.assertIn("f1", df.columns)
        self.assertEqual(df.loc[0, "accuracy"], 0.91)

    def test_auto_detect_params(self):
        df = aggregate_results(self.results, index_fields=["job_id"], auto_detect_params=True)
        self.assertIn("lr", df.columns)
        self.assertIn("dropout", df.columns)
        self.assertAlmostEqual(df.loc[1, "lr"], 0.002)

    def test_auto_detect_both(self):
        df = aggregate_results(
            self.results,
            index_fields=["dataset_name"],
            auto_detect_metrics=True,
            auto_detect_params=True
        )
        self.assertIn("f1", df.columns)
        self.assertIn("lr", df.columns)
        self.assertEqual(df.shape[0], 2)

    def test_strict_mode_missing_results_raises(self):
        bad_data = [{"job_id": "missing_results"}]
        with self.assertRaises(ValueError) as ctx:
            aggregate_results(bad_data, metric_fields=["accuracy"], strict=True)
        self.assertIn("Missing or invalid 'results'", str(ctx.exception))

    def test_non_strict_mode_missing_results_fills_nan(self):
        bad_data = [{"job_id": "missing_results"}]
        df = aggregate_results(bad_data, metric_fields=["accuracy"], strict=False)
        self.assertTrue(pd.isna(df.loc[0, "accuracy"]))

    def test_strict_mode_missing_metric_key_raises(self):
        partial = [{
            "job_id": "partial",
            "results": {"loss": 0.9}
        }]
        with self.assertRaises(ValueError) as ctx:
            aggregate_results(partial, metric_fields=["accuracy"], strict=True)
        self.assertIn("Missing 'accuracy'", str(ctx.exception))

    def test_strict_mode_missing_param_key_raises(self):
        partial = [{
            "job_id": "partial",
            "params": {"dropout": 0.1}
        }]
        with self.assertRaises(ValueError):
            aggregate_results(partial, param_fields=["lr"], strict=True)

    def test_non_strict_mode_fills_missing_fields(self):
        partial = [{
            "job_id": "partial",
            "params": {"dropout": 0.1}
        }]
        df = aggregate_results(partial, param_fields=["lr", "dropout"], strict=False)
        self.assertTrue(pd.isna(df.loc[0, "lr"]))
        self.assertEqual(df.loc[0, "dropout"], 0.1)

    def test_empty_results_raises(self):
        with self.assertRaises(ValueError):
            aggregate_results([])

    def test_dirty_results_strict_raises_on_missing_fields(self):
        with self.assertRaises(ValueError) as ctx:
            aggregate_results(
                self.results_dirty,
                index_fields=["job_id"],
                metric_fields=["accuracy", "f1"],
                param_fields=["lr", "dropout"],
                strict=True
            )
        self.assertIn("Missing 'f1'", str(ctx.exception))  # fails on metric first

    def test_dirty_results_non_strict_fills_missing_fields(self):
        df = aggregate_results(
            self.results_dirty,
            index_fields=["job_id"],
            metric_fields=["accuracy", "f1"],
            param_fields=["lr", "dropout"],
            strict=False
        )

        self.assertEqual(df.shape[0], 2)
        self.assertTrue(pd.isna(df.loc[0, "f1"]))  # job1 missing f1
        self.assertTrue(pd.isna(df.loc[1, "lr"]))  # job2 missing lr
        self.assertAlmostEqual(df.loc[0, "dropout"], 0.2)
        self.assertAlmostEqual(df.loc[1, "accuracy"], 0.88)

if __name__ == "__main__":
    unittest.main()
