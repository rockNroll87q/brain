"""
Created on 01-03-2024

@authors:
* Austin Dibble, University of Glasgow

Custom early stopping callback with a timer component.
"""

import copy
import re
import time
from datetime import timedelta

import keras
from loguru import logger


class EarlyStoppingWithTimer(keras.callbacks.EarlyStopping):
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
        earlystopper = EarlyStoppingWithTimer(monitor='val_loss', patience=15, \
            verbose=1, timelimit=datetime.timedelta(days=1))

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
                timelimit:int | float | str | timedelta='inf'):
        """Initialise class; parse time information."""

        super().__init__(monitor, min_delta, patience, verbose, mode, baseline, restore_best_weights)
        
        # Determine the type of timelimit and set it appropriately
        if isinstance(timelimit, int | float):
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
        self.start_time = time.time()  # Start time of the current epoch

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
        """Callback called at the beginning of each epoch."""
        super().on_epoch_begin(epoch, logs)
        self.start_time = time.time() # Mark the beginning time of the current epoch

    def on_epoch_end(self, epoch, logs=None):
        """Callback called at the end of each epoch. Calculates time and updates projections."""
        super().on_epoch_end(epoch, logs)

        if self.model and self.model.stop_training: # The early stopping callback may have already triggered a stop
            return
        
        if isinstance(self.timelimit, str) and self.timelimit == 'inf':
            return
        
        # Calculate the time taken for this epoch and update the total time
        epoch_time = time.time() - self.start_time
        self.total_time += epoch_time
        # Calculate the average time per epoch based on the epochs completed so far
        average_epoch_time = self.total_time / (epoch + 1)
        # Project the end time for the next epoch
        projected_time = self.total_time + average_epoch_time
        
        # Check if projected end of next epoch exceeds the timelimit. timelimit == 'inf' will never stop using time.
        if projected_time > self.timelimit:
            if self.model: 
                self.model.stop_training = True
            if self.verbose > 0:
                logger.warning(f"==Stopping training==: projected end time for next epoch \
                                ({projected_time:.2f} seconds) exceeds time limit of {self.timelimit} seconds.")
            
            # This is repeated from the stop clause of the EarlyStopping class. Restore best weights, if needed.
            self.stopped_epoch = epoch
            if self.restore_best_weights and self.best_weights is not None:
                if self.verbose > 0:
                    logger.info('Restoring model weights from the end of the best epoch.')
                if self.model: 
                    self.model.set_weights(self.best_weights)


class TorchEarlyStoppingWithTimer:
    """
    PyTorch-compatible early stopping callback with time-based stopping and best weight restoration.

    This class mimics the functionality of Keras' EarlyStopping, including:
    - Monitoring a validation metric (e.g. `val_loss`)
    - Stopping training if no improvement is seen after `patience` epochs
    - Stopping training if the estimated time to finish another epoch would exceed a `timelimit`
    - Optionally restoring the best model weights seen so far

    Parameters:
    -----------
    monitor : str
        The name of the metric to monitor (default is 'val_loss'; purely for logging).
    min_delta : float
        Minimum change in the monitored metric to qualify as an improvement (default: 0.0).
    patience : int
        Number of epochs with no improvement after which training is stopped (default: 0).
    verbose : int
        Logging verbosity. 0 = silent, 1 = minimal info, 2 = detailed (default: 1).
    mode : str
        One of {"min", "max"}. Whether lower or higher is better for the monitored metric (default: "min").
    restore_best_weights : bool
        If True, restores model weights from epoch with best metric value (default: False).
    timelimit : float | str | timedelta
        Total time allowed for training. Can be seconds, "Xd Yh Zm" format, or timedelta.
        If "inf", disables time-based stopping (default: "inf").

    Attributes:
    -----------
    best : float
        Best value observed for the monitored metric.
    best_weights : dict
        Copy of the model weights from the best epoch (only if `restore_best_weights` is True).
    total_time : float
        Total training time accumulated across epochs.

    Example:
    --------
    >>> early_stopper = TorchEarlyStoppingWithTimer(
    ...     monitor="val_loss",
    ...     patience=3,
    ...     mode="min",
    ...     restore_best_weights=True,
    ...     timelimit="1h 30m"
    ... )

    >>> for epoch in range(100):
    ...     early_stopper.on_epoch_begin()
    ...     train(...)  # your training logic
    ...     val_loss = evaluate(...)  # calculate validation loss
    ...     if early_stopper.on_epoch_end(epoch, val_loss, model):
    ...         early_stopper.restore_weights_if_needed(model)
    ...         break

    Notes:
    ------
    - `restore_weights_if_needed()` must be called manually after early stopping is triggered.
    - Time limit is enforced based on projected average time per epoch.
    """

    def __init__(
        self,
        monitor="val_loss",
        min_delta=0.0,
        patience=0,
        verbose=1,
        mode="min",
        restore_best_weights=False,
        timelimit="inf",
    ):
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.restore_best_weights = restore_best_weights

        # Comparison function
        if self.mode == "min":
            self.monitor_op = lambda current, best: current < best - self.min_delta
            self.best = float("inf")
        elif self.mode == "max":
            self.monitor_op = lambda current, best: current > best + self.min_delta
            self.best = -float("inf")
        else:
            raise ValueError("mode must be 'min' or 'max'")

        # Parse time limit
        if isinstance(timelimit, int | float):
            self.timelimit = float(timelimit)
        elif isinstance(timelimit, timedelta):
            self.timelimit = timelimit.total_seconds()
        elif isinstance(timelimit, str):
            if timelimit.strip().lower() == "inf":
                self.timelimit = float("inf")
            else:
                self.timelimit = self.parse_duration(timelimit).total_seconds()
        else:
            raise ValueError("Invalid timelimit format.")

        self.start_time = None
        self.total_time = 0.0
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None

        logger.debug(f"TorchEarlyStoppingWithTimer initialized with timelimit = {self.timelimit:.1f} sec.")

    @staticmethod
    def parse_duration(duration_str):
        """Parses a string like '1d 2h 30m' into a timedelta."""
        pattern = r'((?P<days>\d+?)d)?\s*((?P<hours>\d+?)h)?\s*((?P<minutes>\d+?)m)?'
        match = re.match(pattern, duration_str.strip())
        if not match:
            raise ValueError("Invalid format. Use 'Xd Yh Zm'")
        parts = match.groupdict()
        return timedelta(
            days=int(parts["days"] or 0),
            hours=int(parts["hours"] or 0),
            minutes=int(parts["minutes"] or 0)
        )

    def on_epoch_begin(self):
        """Mark the start time of a new epoch."""
        self.start_time = time.time()

    def on_epoch_end(self, epoch, current_value, model):
        """
        Called at end of each epoch. Check for stopping condition.

        Args:
            epoch (int): Current epoch number.
            current_value (float): Current value of the monitored metric (e.g., val_loss).
            model (torch.nn.Module): The model being trained.

        Returns:
            stop (bool): Whether to stop training.
        """
        elapsed = time.time() - self.start_time
        self.total_time += elapsed

        # Metric-based early stopping
        if self.monitor_op(current_value, self.best):
            if self.verbose > 1:
                logger.info(f"Epoch {epoch+1}: {self.monitor} improved from {self.best:.5f} to {current_value:.5f}")
            self.best = current_value
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(model.state_dict())
        else:
            self.wait += 1
            if self.verbose > 0:
                logger.info(f"Epoch {epoch+1}: {self.monitor} did not improve. Patience {self.wait}/{self.patience}")
            if self.wait >= self.patience:
                logger.warning(f"Early stopping triggered (no improvement after {self.patience} epochs).")
                return True

        # Time-based early stopping
        avg_time = self.total_time / (epoch + 1)
        projected_total = self.total_time + avg_time
        if projected_total > self.timelimit:
            logger.warning(f"Time limit exceeded: stopping at epoch {epoch + 1}")
            return True

        return False

    def restore_weights_if_needed(self, model):
        """Restore best weights to model if enabled."""
        if self.restore_best_weights and self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            logger.info("Restored best model weights from early stopping.")
