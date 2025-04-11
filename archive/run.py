# run.py
import sys
import os
import logging

# Dynamically set PYTHONPATH to include the project root
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Import the main AI Core class AFTER setting the path
from core.AI_Core import AI_Core

if __name__ == "__main__":
    # Setup logging
    log_dir = os.path.join(project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'ai_core_run.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )

    try:
        print("Starting Unified AI Core via run.py...")
        # Adjust initial nodes or cycles as needed
        ai_system = AI_Core(initial_nodes=5)
        # Run for a limited number of cycles for testing, or run indefinitely
        ai_system.start(execution_cycles=15, interval=1)
        print("Unified AI Core run finished.")
    except Exception as e:
        logging.error(f"Failed to run AI Core: {e}", exc_info=True)
        print(f"An error occurred. Check logs/ai_core_run.log for details.")
