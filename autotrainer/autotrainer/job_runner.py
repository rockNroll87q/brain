from typing import List, Dict

class JobRunner:
    def __init__(self, jobs: List[dict]):
        self.jobs = jobs

    def run(self):
        for job in self.jobs:
            self.exec_job(job)

    def exec_job(self, job: dict):
        raise NotImplementedError("Override this in a subclass.")


"""
filter_jobs(predicate_fn)

sort_jobs(key_fn)

limit(n)

dry_run=True

class MyLocalRunner(JobRunner):
    def run_one(self, job):
        cmd = self._build_cmd(job)
        print(f"Running: {cmd}")
        subprocess.run(cmd, shell=True)

    def _build_cmd(self, job):
        params = ' '.join(f"--{k} {v}" for k, v in job['params'].items())
        return f"python {job['script']} {params}"


"""