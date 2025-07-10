#!/usr/bin/env python3
"""
This script allows you to track the movements of all files within a 'watch-dir' (including subsidaries)
logging each file's source path and destination path and the time of movement within a csv file.
Once you run this script you can make the file movements in another terminal or via files app.
You can press enter to stop watching. The script will automatically terminate after 30 mins of dir inactivity/ 
no file movements.

As input you provide:
- 'watch-dir' : The root dir to watch
- 'logdir' : The dir to save the 'file_movement_tracking.csv' in

Example terminal command:   
./track_and_map_files.py --watch-dir /analyse/Project0404/brain_age/data/project_name/ --logdir /analyse/Project0404/brain_age/data/project_name/raw/

In a new terminal:
mv /analyse/Project0404/brain_age/data/project_name/raw/T1w_images/*.nii.gz /analyse/Project0404/brain_age/data/project_name/imgs/anat/

/analyse/Project0404/brain_age/data/project_name/raw/file_movement_tracking.csv 
will contain the tracked information

"""


import csv
import sys
import time
import threading
import argparse
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileMovedEvent

INACTIVITY_TIMEOUT = 30 * 60  # 30 minutes

class MoveLogger(FileSystemEventHandler):
    """
    Utility class for tracking file movements.
    For each movement triggered via on_moved, logs into a CSV what the move was.
    """
    def __init__(self, logdir, reset_timer):
        """Initialize the MoveLogger."""
        self.logdir = f'{logdir}/file_movement_tracking.csv'
        self.reset_timer = reset_timer
        # write header if file doesn't exist
        try:
            with open(self.logdir, 'x', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['event_time', 'src_path', 'dest_path'])
        except FileExistsError:
            pass

    def on_moved(self, event):
        """Track each movement event."""
        if isinstance(event, FileMovedEvent):
            # human‐readable timestamp
            event_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.logdir, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([event_time, event.src_path, event.dest_path])
            print(f"[{event_time}] {event.src_path} → {event.dest_path}")
            # reset inactivity timer
            self.reset_timer()

def main(args):
    """Run and watch for file moves."""
    observer = Observer()
    timer = None

    def stop_watching():
        observer.stop()
        print(f"\nNo file‐move events for {INACTIVITY_TIMEOUT/60:.0f} minutes. Watcher stopped.")
        sys.exit(0)

    def reset_timer():
        nonlocal timer
        if timer:
            timer.cancel()
        timer = threading.Timer(INACTIVITY_TIMEOUT, stop_watching)
        timer.daemon = True
        timer.start()

    # Start manual‐stop helper
    def wait_for_manual_stop():
        input(">>> Press ENTER to stop watching and exit.\n")
        stop_watching()
    threading.Thread(target=wait_for_manual_stop, daemon=True).start()

    handler = MoveLogger(args.logdir, reset_timer)
    observer.schedule(handler, path=args.watch_dir, recursive=True)
    observer.start()

    print(f"Watching for file moves under {args.watch_dir}…")
    reset_timer()

    try:
        observer.join()
    except KeyboardInterrupt:
        print("\nInterrupted by user; shutting down.")
    finally:
        if timer:
            timer.cancel()
        observer.stop()
        observer.join()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Watch for file moves under a directory and log them to CSV."
    )
    parser.add_argument('-w', '--watch-dir', required=True,
                        help="Directory tree to monitor (recursive)")
    parser.add_argument('-l', '--logdir',   required=True,
                        help="Path to CSV file where moves will be logged")
    args = parser.parse_args()

    main(args)
