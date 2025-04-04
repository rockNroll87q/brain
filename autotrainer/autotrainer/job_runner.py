"""
Created on Wednesday, 2 April 2025.

@authors:
* Austin Dibble, University of Glasgow

---------------------------------
JobRunner - Job Execution Engine
---------------------------------

An extendable class framework for execution of provided
job lists from the JobCreator. Provides utilties for 
filtering and sorting jobs into new objects.

Can be paired with ResultManager for managing job 
artifacts/results.

example:
    class MyRunner(JobRunner):
        def run_one(self, job:dict):
            # Run your job here
            return

    runner = MyRunner(jobs, dry_run=True)
    runner.filter_jobs(lambda job: job['dataset_name'] == 'alpha', inplace=True)
    runner.sort_jobs(lambda job: job['params']['learning_rate'], inplace=True)
    runner.run()

"""

import multiprocessing
from typing import List, Dict, Optional, Callable

from loguru import logger

class JobRunner:
    """
    A base class for running job specifications in a flexible and extensible way.

    Supports sequential and parallel execution of jobs defined as dictionaries.
    Jobs must be executed via the user-implemented `run_one()` method.

    Features:
    - Sequential and multiprocessing-based execution
    - Dry-run support for previewing jobs
    - Optional lifecycle hooks (start, success, failure)
    - In-place filtering and sorting of jobs before execution

    Args:
        jobs (List[Dict]): List of job dictionaries to execute (from JobCreator).
        dry_run (bool): If True, job commands are printed but not executed.
        max_workers (Optional[int]): If set > 1, enables parallel execution via multiprocessing.
    
    Example:
        runner = MyRunner(jobs, dry_run=True)
        runner.filter_jobs(lambda job: job['dataset_name'] == 'alpha', inplace=True)
        runner.sort_jobs(lambda job: job['params']['learning_rate'], inplace=True)
        runner.run()

    The utiltiy functions filter_jobs and sort_jobs behave like the pandas interface so 
    that they can be chained.

    Chaining example:
        runner = MyRunner(jobs, dry_run=True)
        runner = runner.filter_jobs(lambda j: j["dataset_name"] == "ds1")
                        .sort_jobs(lambda j: j["params"]["learning_rate"])

        runner.run()

    Such chaining can be used to create sub-runners, for example:
        base_runner = BaseRunner(jobs, dry_run=True)
        ds1_runner = base_runner.filter_jobs(lambda j: j["dataset_name"] == "ds1")
        ds2_runner = base_runner.filter_jobs(lambda j: j["dataset_name"] == "ds2")

        ds1_runner.run()
        ds2_runner.run()

    """


    def __init__(self, jobs: List[Dict], dry_run: bool = False, max_workers: Optional[int] = None):
        """
        Initialize the runner.

        Args:
            jobs (List[Dict]): List of job specifications to run.
            dry_run (bool): If True, jobs are not actually executed.
            max_workers (Optional[int]): If set, enables multiprocessing with this many workers.
        """
        self.jobs = jobs
        self.dry_run = dry_run
        self.max_workers = max_workers

    def run(self):
        """
        Run all jobs. Supports sequential and parallel execution.
        """
        if self.max_workers and self.max_workers > 1:
            self._run_parallel()
        else:
            self._run_sequential()

    def _run_sequential(self):
        for job in self.jobs:
            self._run_wrapper(job)

    def _run_parallel(self):
        with multiprocessing.Pool(processes=self.max_workers) as pool:
            pool.map(self._run_wrapper, self.jobs)

    def _run_wrapper(self, job: Dict):
        """
        Internal wrapper that handles dry run and error catching.

        Args:
            job (Dict): The job spec to run.
        """
        if self.dry_run:
            logger.info(f"[DRY RUN] Would run job: {job['job_id']}")
            return

        try:
            self.on_job_start(job)
            self.run_one(job)
            self.on_job_success(job)
        except Exception as e:
            self.on_job_failure(job, e)

    def run_one(self, job: Dict):
        """
        Override this method to define how a single job should be executed.

        Args:
            job (Dict): The job to run.
        """
        raise NotImplementedError("You must implement run_one() in a subclass.")

    # Expressive utility tools, pandas-style

    def _copy_with_new_jobs(self, new_jobs: List[Dict]) -> "JobRunner":
        """
        Internal method to clone the runner with a new job list.
        """
        return self.__class__(jobs=new_jobs, dry_run=self.dry_run, max_workers=self.max_workers)
    
    def _return_or_inplace(self, jobs: List[Dict], inplace: bool):
        if inplace:
            self.jobs = jobs
        else:
            return self._copy_with_new_jobs(jobs)

    def filter_jobs(self, predicate: Callable[[Dict], bool], inplace: bool = False):
        """
        Filters the job list based on a predicate function.

        Args:
            predicate (Callable): A function that takes a job dict and returns True if it should be kept.
            inplace (bool): If True, modifies this JobRunner instance. Otherwise returns a new one.

        Returns:
            JobRunner: This instance or a new one depending on inplace.
        """
        filtered = [job for job in self.jobs if predicate(job)]
        return self._return_or_inplace(filtered, inplace)

    def sort_jobs(self, key_fn: Callable[[Dict], any], reverse: bool = False, inplace: bool = False):
        """
        Sorts the job list based on a key function.

        Args:
            key_fn (Callable): Function to extract a sort key from a job dict.
            reverse (bool): If True, sorts in descending order.
            inplace (bool): If True, modifies this JobRunner instance. Otherwise returns a new one.

        Returns:
            JobRunner: This instance or a new one depending on inplace.
        """
        sorted_jobs = sorted(self.jobs, key=key_fn, reverse=reverse)
        return self._return_or_inplace(sorted_jobs, inplace)

    # --- Optional lifecycle hooks ---
    def on_job_start(self, job: Dict):
        print(f"Starting job: {job['job_id']}")

    def on_job_success(self, job: Dict):
        print(f"Completed job: {job['job_id']}")

    def on_job_failure(self, job: Dict, error: Exception):
        print(f"Job failed: {job['job_id']} with error: {error}")
