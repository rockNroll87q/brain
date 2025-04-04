import sys
import os
import argparse
from pathlib import Path

from time import sleep

# Script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
# Add brain age to sys path
ba_path = os.path.join(script_dir, '../')
sys.path.append(os.path.abspath(ba_path))

from autotrainer import ConfigLoader, JobCreator, JobRunner, ResultManager, aggregate_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to quickly see what examples are parsed as.')
    parser.add_argument('example_file', help='File path to .yml')
    args = parser.parse_args()

    runner_config = ConfigLoader(args.example_file).load()
    job_list = JobCreator(runner_config).create()

    class MyRunner(JobRunner):
        def run_one(self, job:dict):
            print(job)
    
    runner = MyRunner(job_list, max_workers=1)
    runner.run()