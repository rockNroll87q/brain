import unittest
import sys
import tempfile
import shutil
import json
from pathlib import Path

from loguru import logger

from autotrainer.result_manager import ResultManager, create_result

class TestResultManager(unittest.TestCase):
    def setUp(self):
        logger.remove()
        logger.add(sys.stderr, level="ERROR")

        self.tmpdir = tempfile.mkdtemp()
        self.job = {
            "job_id": "job123",
            "dataset_name": "ds1",
            "task_name": "taskA",
            "params": {"lr": 0.01}
        }

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_invalid_output_pattern_missing_job_id(self):
        with self.assertRaises(ValueError):
            ResultManager(self.tmpdir, output_pattern="{dataset_name}.json")

    def test_invalid_output_pattern_with_unknown_key(self):
        with self.assertRaises(ValueError):
            ResultManager(self.tmpdir, output_pattern="{unknown_key}/{job_id}.json")

    def test_emit_and_collect_json_result(self):
        mgr = ResultManager(self.tmpdir, output_pattern="{task_name}/{dataset_name}/{job_id}.json", fmt="json")
        result_data = {"acc": 0.9}
        mgr.create_and_emit_result(self.job, results=result_data)

        results = mgr.collect_results()
        self.assertEqual(len(results), 1)
        res = results[0]
        self.assertEqual(res["job_id"], "job123")
        self.assertEqual(res["results"], result_data)
        self.assertEqual(res["task_name"], "taskA")
        self.assertEqual(res["dataset_name"], "ds1")

    def test_emit_and_collect_yaml_result(self):
        mgr = ResultManager(self.tmpdir, output_pattern="{job_id}.yaml", fmt="yaml")
        result_data = {"loss": 0.3}
        mgr.create_and_emit_result(self.job, results=result_data)
        results = mgr.collect_results()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["results"]["loss"], 0.3)

    def test_collect_without_inference(self):
        mgr = ResultManager(self.tmpdir, output_pattern="{task_name}/{dataset_name}/{job_id}.json")
        mgr.create_and_emit_result(self.job, results={"val": 1.0})

        collected = mgr.collect_results(infer_metadata=False)
        self.assertEqual(len(collected), 1)
        self.assertIn("results", collected[0])        # Still has result content

    def test_collect_with_custom_root_dir_skips_inference(self):
        mgr = ResultManager(self.tmpdir, output_pattern="{task_name}/{dataset_name}/{job_id}.json")
        mgr.create_and_emit_result(self.job, results={"acc": 0.8})

        alt_root = Path(self.tmpdir) / "new_tmp"
        alt_root.mkdir(exist_ok=True)
        collected = mgr.collect_results(root_dir=alt_root, infer_metadata=True)
        self.assertEqual(len(collected), 0) # No results found because root was changed to an empty one
        
        # Same here, no results
        collected = mgr.collect_results(root_dir=alt_root, infer_metadata=False)
        self.assertEqual(len(collected), 0) # No results found because root was changed to an empty one

        mgr = ResultManager(alt_root, output_pattern="{task_name}/{dataset_name}/{job_id}.json")
        mgr.create_and_emit_result(self.job, results={"acc": 0.8})

        collected = mgr.collect_results(root_dir=self.tmpdir, infer_metadata=True) # Should be able to find since they're from the previous
        # self.assertNotIn("task_name", collected[0])  # Inference skipped due to root mismatch

    def test_infer_metadata_failure_due_to_path_mismatch(self):
        # Emit manually to cause mismatch with pattern
        mgr = ResultManager(self.tmpdir, output_pattern="{task_name}/{dataset_name}/{job_id}.json")
        bad_path = Path(self.tmpdir) / "unexpected" / "wrong.json"
        bad_path.parent.mkdir(parents=True)
        with open(bad_path, "w") as f:
            json.dump({"job_id": "bad"}, f)

        with self.assertRaises(ValueError):
            mgr.collect_results()

    def test_emit_and_load_multiple_results(self):
        mgr = ResultManager(self.tmpdir, output_pattern="{task_name}/{dataset_name}/{job_id}.json")
        for i in range(3):
            job = {**self.job, "job_id": f"job_{i}"}
            mgr.create_and_emit_result(job, results={"i": i})
        
        collected = mgr.collect_results()
        self.assertEqual(len(collected), 3)
        ids = {r["job_id"] for r in collected}
        self.assertEqual(ids, {"job_0", "job_1", "job_2"})

    def test_create_result_default_schema(self):
        mgr = ResultManager(self.tmpdir)
        result = create_result(
            job=self.job,
            results={"loss": 0.42},
            status="done",
            extra={"note": "test_run"}
        )

        self.assertEqual(result["job_id"], "job123")
        self.assertEqual(result["status"], "done")
        self.assertEqual(result["results"]["loss"], 0.42)
        self.assertEqual(result["params"]["lr"], 0.01)
        self.assertEqual(result["note"], "test_run")
        self.assertIn("timestamp", result)

    def test_emit_result_with_custom_object(self):
        mgr = ResultManager(self.tmpdir, output_pattern="{job_id}.json")
        custom_result = {
            "id": "abc999",
            "metrics": {"precision": 0.77},
            "note": "manually constructed"
        }

        # Emit without using create_result
        mgr.emit_result(custom_result, job=self.job)

        collected = mgr.collect_results()
        self.assertEqual(len(collected), 1)
        self.assertEqual(collected[0]["id"], "abc999")
        self.assertEqual(collected[0]["metrics"]["precision"], 0.77)
        self.assertNotIn("task_name", collected[0]) # Not inferred nor added


class TestResultManagerPaths(unittest.TestCase):
    def setUp(self):
        logger.remove()
        logger.add(sys.stderr, level="ERROR")

        self.job = {
            "job_id": "abc123",
            "task_name": "finetune",
            "dataset_name": "dataset_alpha"
        }
        self.root = Path("/tmp/autotrainer_results")

    def test_basic_output_path_resolution(self):
        mgr = ResultManager(root_dir=self.root, output_pattern="{job_id}.json")
        path = mgr._resolve_output_path(self.job)
        expected = self.root / "abc123.json"
        self.assertEqual(path, expected)

    def test_nested_output_path_resolution(self):
        mgr = ResultManager(
            root_dir=self.root,
            output_pattern="{task_name}/{dataset_name}/{job_id}.json"
        )
        path = mgr._resolve_output_path(self.job)
        expected = self.root / "finetune/dataset_alpha/abc123.json"
        self.assertEqual(path, expected)

    def test_missing_fields_use_fallbacks(self):
        job_incomplete = {"job_id": "xyz789"}
        mgr = ResultManager(root_dir=self.root, output_pattern="{task_name}/{dataset_name}/{job_id}.json")
        path = mgr._resolve_output_path(job_incomplete)
        expected = self.root / "unknown_task/unknown_dataset/xyz789.json"
        self.assertEqual(path, expected)

    def test_invalid_output_pattern_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            ResultManager(root_dir=self.root, output_pattern="{job_id}/{bad_field}.json")
        self.assertIn("unknown placeholder", str(ctx.exception))

    def test_job_id_required_in_output_pattern(self):
        with self.assertRaises(ValueError) as ctx:
            ResultManager(root_dir=self.root, output_pattern="{task_name}.json")
        self.assertIn("{job_id}", str(ctx.exception))

if __name__ == '__main__':
    unittest.main()
