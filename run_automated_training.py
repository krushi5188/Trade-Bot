import time
import subprocess
import os
from apscheduler.schedulers.blocking import BlockingScheduler

# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PIPELINE_SCRIPT_PATH = os.path.join(PROJECT_ROOT, 'src/tier4/champion_challenger.py')
LOG_FILE_PATH = os.path.join(PROJECT_ROOT, 'automated_training.log')

def run_training_pipeline():
    """
    Executes the main champion/challenger training pipeline and logs its output.
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"--- [{timestamp}] Kicking off weekly automated training pipeline ---")

    try:
        # We use subprocess.run to execute the pipeline.
        # stdout and stderr are redirected to our log file.
        with open(LOG_FILE_PATH, 'a') as log_file:
            log_file.write(f"\n\n--- PIPELINE START: {timestamp} ---\n")

            # The `capture_output=True` and `text=True` arguments would be better,
            # but for simplicity and to ensure logs are written in real-time,
            # we'll pipe the output directly.
            process = subprocess.Popen(
                ['python', PIPELINE_SCRIPT_PATH],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True
            )
            process.wait() # Wait for the subprocess to complete

            log_file.write(f"--- PIPELINE END: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")

        print(f"--- [{time.strftime('%Y-%m-%d %H:%M:%S')}] Pipeline run finished successfully. See {LOG_FILE_PATH} for details. ---")

    except Exception as e:
        error_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        error_message = f"--- [{error_timestamp}] CRITICAL ERROR: The training pipeline failed. Exception: {e} ---"
        print(error_message)
        with open(LOG_FILE_PATH, 'a') as log_file:
            log_file.write(error_message)

def main():
    """
    Sets up and starts the scheduler.
    """
    print("--- Initializing Automated Training Scheduler ---")

    scheduler = BlockingScheduler(timezone="UTC")

    # Schedule the training pipeline to run every Sunday at 00:00 UTC
    scheduler.add_job(run_training_pipeline, 'cron', day_of_week='sun', hour=0, minute=0)

    print("Scheduler started. First training run is scheduled.")
    print(f"Next run: {scheduler.get_jobs()[0].next_run_time}")
    print("The scheduler is a blocking process. Press Ctrl+C to exit.")

    # To test the pipeline immediately without waiting for the schedule,
    # uncomment the following line:
    # run_training_pipeline()

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("Scheduler stopped.")

if __name__ == '__main__':
    main()
