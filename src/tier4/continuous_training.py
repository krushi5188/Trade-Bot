# This script creates a continuous, infinite training loop for the AI model.
# It repeatedly calls the Champion/Challenger pipeline, allowing the AI to
# continuously train, evaluate, and improve itself in a single, long-running session.

import time
import subprocess
import os
import re

def parse_metrics(output):
    """
    Parses the stdout of the backtester to extract key performance metrics.
    """
    metrics = {}
    patterns = {
        'Total Return (%)': r"Total Return \(%\): ([\-0-9\.]+)",
        'Sharpe Ratio': r"Sharpe Ratio: ([\-0-9\.]+)",
        'Max Drawdown (%)': r"Max Drawdown \(%\): ([\-0-9\.]+)",
        'Promotion Status': r"(Challenger is better\. Promoting to champion\.|Challenger is not better\. Keeping the current champion\.)"
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            metrics[key] = match.group(1)

    return metrics

def print_summary(cycle_number, metrics):
    """
    Prints a clear, human-readable summary of the training cycle.
    """
    print(f"\\n--- Training Cycle #{cycle_number} Summary ---")
    if not metrics:
        print("Could not parse performance metrics.")
        return

    print(f"  - Challenger Performance:")
    print(f"    - Total Return: {metrics.get('Total Return (%)', 'N/A')}%")
    print(f"    - Sharpe Ratio: {metrics.get('Sharpe Ratio', 'N/A')}")
    print(f"    - Max Drawdown: {metrics.get('Max Drawdown (%)', 'N/A')}%")

    promotion_status = metrics.get('Promotion Status', 'Unknown')
    if "Promoting" in promotion_status:
        print("\\n  - Outcome: \\033[92mSUCCESS! New Champion Promoted!\\033[0m")
    else:
        print("\\n  - Outcome: \\033[93mChallenger did not outperform. Champion remains.\\033[0m")
    print("------------------------------------\\n")


def continuous_training_loop():
    """
    An infinite loop that continuously runs the Champion/Challenger pipeline.
    """
    cycle_number = 1
    while True:
        print(f"\\n{'='*60}")
        print(f"[{time.ctime()}] Starting Training Cycle #{cycle_number}")
        print(f"{'='*60}\\n")

        # Define the path to the main pipeline script
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        pipeline_script = os.path.join(project_root, 'src/tier4/champion_challenger.py')

        try:
            # Execute the full training and evaluation pipeline and capture the output
            result = subprocess.run(
                ['python', pipeline_script],
                check=True,
                capture_output=True,
                text=True
            )

            print(result.stdout) # Print the full log
            metrics = parse_metrics(result.stdout)
            print_summary(cycle_number, metrics)

        except subprocess.CalledProcessError as e:
            print(f"\\n[{time.ctime()}] ERROR: Training Cycle #{cycle_number} Failed.")
            print(e.stdout)
            print(e.stderr)
            print("Restarting the loop in 60 seconds...")
            time.sleep(60)

        except Exception as e:
            print(f"\\n[{time.ctime()}] An unexpected error occurred in cycle #{cycle_number}: {e}")
            print("Restarting the loop in 60 seconds...")
            time.sleep(60)

        cycle_number += 1

if __name__ == '__main__':
    print("--- Starting Continuous Training Loop ---")
    print("This script will run indefinitely, constantly retraining the AI.")
    print("Press Ctrl+C to exit.")

    try:
        continuous_training_loop()
    except KeyboardInterrupt:
        print("\\nContinuous training loop stopped by user.")
