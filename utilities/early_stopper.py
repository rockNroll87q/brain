"""
Created on 01-03-2024

@authors:
* Austin Dibble, University of Glasgow

Custom early stopping callback with a timer component.
"""

import tensorflow as tf
from datetime import timedelta
from typing import Union
import re
from loguru import logger


class EarlyStoppingWithTimer(tf.keras.callbacks.EarlyStopping):
    """
    Custom early stopping callback with a timer component.

    This callback combines the functionality of TensorFlow's built-in `EarlyStopping` 
    with an additional timer-based mechanism. It allows you to stop training based on 
    the monitored metric (e.g., validation loss) as well as after a specified time limit.

    If the estimated total runtime after the next epoch will be greater than the set limit 
    then training will stop after the current epoch.
    
    Parameters:
        - monitor: The metric to monitor for early stopping. Defaults to 'val_loss'.
        - min_delta: The minimum change in the monitored metric required to qualify as an improvement.
        - patience: The number of epochs with no improvement after which training will be stopped. 
            If `patience` is set to zero, training will stop when a new best is found.
        - verbose: The verbosity level for the callback. Defaults to 0.
        - mode: One of {'auto', 'min', 'max'}. In 'auto' mode, the direction of change
            is automatically inferred from the sign of the derivative (i.e., an improvement 
            is defined as a decrease if the metric is being minimized, or an increase if it's 
            being maximized). When `mode` is either 'min' or 'max', direction of change is 
            determined by whether the metric is being minimized or maximized.
        - baseline: The baseline value for the monitored metric. Training will stop when
            the difference between the current and best values reaches `min_delta`. If 
            `baseline` is None, training will not stop based on a baseline value.
        - restore_best_weights: Whether to restore the best weights during early stopping.
        - timelimit: The time limit for training in seconds. It can be an integer or float,
            a string in 'Xd Yh Zm' format (e.g., '1d 2h 30m'), or a `timedelta` object.
            A value of 'inf' means there is no timelimit.

    Attributes:
        - self.timelimit: The time limit for training in seconds.

    Examples::

        # Example using string
        earlystopper = EarlyStoppingWithTimer(monitor='val_loss', patience=15, verbose=1, 
                                            restore_best_weights=True, timelimit="1d 2h 30m")

        # Example using a timedelta object
        earlystopper = EarlyStoppingWithTimer(monitor='val_loss', patience=15, verbose=1, timelimit=datetime.timedelta(days=1))

        # Example using float/int seconds
        earlystopper = EarlyStoppingWithTimer(monitor='val_loss', patience=15, verbose=1, timelimit=3600)

        # Then, use just like you would with the tf.keras.callbacks.EarlyStopping callback.
        

    """
    def __init__(self,
                monitor='val_loss',
                min_delta=0,
                patience=0,
                verbose=0,
                mode='auto',
                baseline=None,
                restore_best_weights=False,
                timelimit:Union[int, float, str, timedelta]='inf'): # Time limit in seconds, as a string like 'Xd Yh Zm', or as a datetime.timedelta object. 'inf' means no timelimit
        
        super(EarlyStoppingWithTimer, self).__init__(monitor, min_delta, patience, verbose, mode, baseline, restore_best_weights)
        
        # Determine the type of timelimit and set it appropriately
        if isinstance(timelimit, (int, float)):
            self.timelimit = timelimit
        elif isinstance(timelimit, str):
            if timelimit != 'inf':
                self.timelimit = self.parse_duration(timelimit).total_seconds()
            else:
                self.timelimit = timelimit
        elif isinstance(timelimit, timedelta):
            self.timelimit = timelimit.total_seconds()
        else:
            raise ValueError("timelimit must be a number, a string in 'Xd Yh Zm' format, or a timedelta object.")

        logger.debug(f'EarlyStoppingWithTimer initialized with a timelimit of {self.timelimit}.')

        self.total_time = 0  # Total time taken so far
        self.start_time = tf.timestamp()  # Start time of the current epoch

    @staticmethod
    def parse_duration(duration_str):
        """Parse duration string formatted as 'Xd Yh Zm' into a timedelta object."""
        pattern = r'((?P<days>\d+?)d)?\s*((?P<hours>\d+?)h)?\s*((?P<minutes>\d+?)m)?'
        match = re.match(pattern, duration_str)
        if not match:
            raise ValueError("Invalid format for timelimit. Please use 'Xd Yh Zm' format.")
        
        parts = match.groupdict()
        days = int(parts['days']) if parts['days'] else 0
        hours = int(parts['hours']) if parts['hours'] else 0
        minutes = int(parts['minutes']) if parts['minutes'] else 0

        return timedelta(days=days, hours=hours, minutes=minutes)

    def on_epoch_begin(self, epoch, logs=None):
        super().on_epoch_begin(epoch, logs)
        self.start_time = tf.timestamp() # Mark the beginning time of the current epoch

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)

        if self.model.stop_training: # The early stopping callback may have already triggered a stop
            return
        
        if isinstance(self.timelimit, str) and self.timelimit == 'inf':
            return
        
        # Calculate the time taken for this epoch and update the total time
        epoch_time = tf.timestamp() - self.start_time
        self.total_time += epoch_time
        # Calculate the average time per epoch based on the epochs completed so far
        average_epoch_time = self.total_time / (epoch + 1)
        # Project the end time for the next epoch
        projected_time = self.total_time + average_epoch_time
        
        # Check if projected end of next epoch exceeds the timelimit. timelimit == 'inf' will never stop using time.
        if projected_time > self.timelimit:
            self.model.stop_training = True
            if self.verbose > 0:
                logger.warning(f"==Stopping training==: projected end time for next epoch ({projected_time:.2f} seconds) exceeds time limit of {self.timelimit} seconds.")
            
            # This is repeated from the stop clause of the EarlyStopping class. Restore best weights, if needed, before stopping
            self.stopped_epoch = epoch
            if self.restore_best_weights and self.best_weights is not None:
                if self.verbose > 0:
                    logger.info('Restoring model weights from the end of the best epoch.')
                self.model.set_weights(self.best_weights)