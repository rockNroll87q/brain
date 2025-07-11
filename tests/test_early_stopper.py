"""
Code by Austin Dibble 2024, University of Glasgow
Copyright (c) 2024 Austin Dibble

This code file is for unit testing the early stopper functions from utilities/early_stopper.py
"""

import sys
import unittest
import datetime
import time
import os

os.environ['KERAS_BACKEND'] = 'torch'

import numpy as np
from loguru import logger

from utilities import EarlyStoppingWithTimer

# test with python -m pytest from brain root directory
class TestEarlyStopperFunctions(unittest.TestCase):
    """Test functions for the EarlyStoppingWithTimer utility."""
    def setUp(self):
        """Setup variables/randomness/etc."""
        logger.remove()
        logger.add(sys.stderr, level="ERROR")

        # Sample 100 values from a normal distribution for testing
        np.random.seed(184)  # For reproducibility

    def test_string_parse(self):
        """Test for accurate string parsing."""
        # Example using string
        for day in range(7):
            for hour in range(24):
                for minute in [5, 10, 25, 45, 0]:
                    earlystopper = EarlyStoppingWithTimer(monitor='val_loss', patience=15, verbose=1,
                                                        restore_best_weights=True, timelimit=f"{day}d {hour}h {minute}m")
                    # Calculate the equivalent seconds as an integer, by hand
                    seconds = day*86400 + hour*3600 + minute*60
                    self.assertAlmostEqual(seconds, earlystopper.timelimit)


    def test_timedelta(self):
        """Test that it works with timedelta objects."""
        # Example using a timedelta object
        for day in range(7):
            for hour in range(24):
                for minute in [5, 10, 25, 45, 0]:
                    earlystopper = EarlyStoppingWithTimer(monitor='val_loss', patience=15, verbose=1,
                                                        restore_best_weights=True, timelimit=datetime.timedelta(days=day, hours=hour, minutes=minute))
                    # Calculate the equivalent seconds as an integer, by hand
                    seconds = day*86400 + hour*3600 + minute*60
                    self.assertAlmostEqual(seconds, earlystopper.timelimit)

    def test_integer_seconds(self):
        """Test that it works with seconds."""
        # Example using float/int seconds
        for day in range(7):
            for hour in range(24):
                for minute in [5, 10, 25, 45, 0]:
                    seconds = day*86400 + hour*3600 + minute*60
                    earlystopper = EarlyStoppingWithTimer(monitor='val_loss', patience=15, verbose=1, timelimit=seconds)
                    self.assertAlmostEqual(seconds, earlystopper.timelimit)

    def test_delay(self):
        """Test a short time delay has the right behaviour."""
        earlystopper = EarlyStoppingWithTimer(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True, timelimit=1)
        earlystopper.on_epoch_begin(epoch=0)
        time.sleep(2)
        earlystopper.on_epoch_end(epoch=1)
        # Calculate the equivalent seconds as an integer, by hand
        self.assertAlmostEqual(earlystopper.total_time, 2, delta=0.5)

if __name__ == "__main__":
    unittest.main()
