# This script uses APScheduler to run the Champion/Challenger pipeline on a schedule.

import time
import subprocess
import os
from apscheduler.schedulers.blocking import BlockingScheduler

def run_pipeline_job():
    """
    The job function that will be executed by the scheduler.
    It runs the champion_challenger.py script as a subprocess.
    """
    print(f"[{time.ctime()}] --- Scheduler Triggered: Starting Champion/Challenger Pipeline ---")

    # Define the path to the script to be executed
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    pipeline_script = os.path.join(project_root, 'src/tier4/champion_challenger.py')

    try:
        # Run the script as a subprocess
        subprocess.run(['python', pipeline_script], check=True)
        print(f"[{time.ctime()}] --- Pipeline Run Completed Successfully ---")
    except subprocess.CalledProcessError as e:
        print(f"[{time.ctime()}] --- ERROR: Pipeline Run Failed with exit code {e.returncode} ---")
    except Exception as e:
        print(f"[{time.ctime()}] --- An unexpected error occurred: {e} ---")

if __name__ == '__main__':
    # Create a scheduler that will block the main thread
    scheduler = BlockingScheduler()

    # Schedule the job to run once a week on Sunday at midnight
    scheduler.add_job(run_pipeline_job, 'cron', day_of_week='sun', hour=0)

    print("--- Automated Training Scheduler ---")
    print("The training pipeline is scheduled to run every Sunday at midnight.")
    print("Press Ctrl+C to exit.")

    try:
        # Start the scheduler
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("Scheduler stopped.")
