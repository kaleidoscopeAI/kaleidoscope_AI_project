# tests/simulation_script.py
import sys
import os
import logging
import time
import random

# Adjust path to import from the root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import core components
try:
    from core.AI_Core import AI_Core
    from utils.config import Config
    from utils.logging_setup import setup_logging
except ImportError as e:
    print(f"Error importing core modules for simulation: {e}")
    print("Ensure you are running this script relative to the project root or have set PYTHONPATH.")
    sys.exit(1)

# Setup logging specifically for the simulation
# setup_logging(log_file="simulation_run.log") # Use a different log file
# Or configure manually:
log_sim_dir = os.path.join(project_root, "logs")
os.makedirs(log_sim_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - SIM - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_sim_dir, 'simulation_run.log')),
        logging.StreamHandler(sys.stdout) # Also print to console
    ]
)
logger = logging.getLogger("Simulation")


def run_basic_simulation(num_cycles=50, initial_nodes=10, max_nodes=75, interval=0.2):
    """Runs a basic simulation of the AI Core."""
    logger.info("--- Starting Basic Simulation ---")
    logger.info(f"Parameters: Cycles={num_cycles}, InitialNodes={initial_nodes}, MaxNodes={max_nodes}, Interval={interval}s")

    start_time = time.time()
    try:
        # Initialize AI Core with simulation parameters
        core = AI_Core(initial_nodes=initial_nodes, max_nodes=max_nodes)

        # Run the core for the specified number of cycles
        core.start(execution_cycles=num_cycles, interval=interval)

    except Exception as e:
        logger.error(f"Simulation failed: {e}", exc_info=True)
    finally:
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"--- Basic Simulation Finished ---")
        logger.info(f"Total duration: {duration:.2f} seconds")
        # Could add final state logging here (e.g., final node count, avg energy)
        if 'core' in locals() and core and core.node_manager:
             logger.info(f"Final Node Count: {core.node_manager.get_node_count()}")
             final_statuses = core.node_manager.get_all_statuses()
             avg_energy = np.mean([s.get('energy', 0) for s in final_statuses.values()]) if final_statuses else 0
             logger.info(f"Final Average Energy: {avg_energy:.2f}")


def run_stress_test(num_cycles=100, initial_nodes=20, max_nodes=150, interval=0.05):
    """Runs a more demanding simulation."""
    logger.info("--- Starting Stress Test Simulation ---")
    logger.info(f"Parameters: Cycles={num_cycles}, InitialNodes={initial_nodes}, MaxNodes={max_nodes}, Interval={interval}s")

    start_time = time.time()
    try:
        core = AI_Core(initial_nodes=initial_nodes, max_nodes=max_nodes)
        core.start(execution_cycles=num_cycles, interval=interval)
    except Exception as e:
        logger.error(f"Stress test failed: {e}", exc_info=True)
    finally:
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"--- Stress Test Finished ---")
        logger.info(f"Total duration: {duration:.2f} seconds")
        if 'core' in locals() and core and core.node_manager:
             logger.info(f"Final Node Count: {core.node_manager.get_node_count()}")


# Add more specific test functions, e.g., test_replication, test_gpt_queries, test_memory_growth etc.
# def test_replication_dynamics(...): ...
# def test_gpt_node_performance(...): ...


if __name__ == "__main__":
    import argparse
    import numpy as np # Needed for final status logging

    parser = argparse.ArgumentParser(description="Run simulations for the AI Core.")
    parser.add_argument('--type', type=str, default='basic', choices=['basic', 'stress'],
                        help='Type of simulation to run (basic or stress).')
    parser.add_argument('--cycles', type=int, default=50, help='Number of execution cycles.')
    parser.add_argument('--nodes', type=int, default=10, help='Initial number of nodes.')
    parser.add_argument('--max_nodes', type=int, default=75, help='Maximum number of nodes.')
    parser.add_argument('--interval', type=float, default=0.2, help='Interval between cycles in seconds.')

    args = parser.parse_args()

    if args.type == 'basic':
        run_basic_simulation(num_cycles=args.cycles, initial_nodes=args.nodes, max_nodes=args.max_nodes, interval=args.interval)
    elif args.type == 'stress':
        # Use potentially different defaults for stress test if not overridden
        cycles = args.cycles if args.cycles != 50 else 100
        nodes = args.nodes if args.nodes != 10 else 20
        max_nodes = args.max_nodes if args.max_nodes != 75 else 150
        interval = args.interval if args.interval != 0.2 else 0.05
        run_stress_test(num_cycles=cycles, initial_nodes=nodes, max_nodes=max_nodes, interval=interval)

    logger.info("Simulation script finished.")
