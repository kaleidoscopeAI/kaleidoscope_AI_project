from collections import deque
from typing import Optional, Any
import logging
# --- Example Subclass (Continued from Part 7)---

import numpy as np # Needed for DataSimulationNode example

class DataSimulationNode(CapabilityNode):
    """Example capability: Simulates data based on input parameters."""
    def __init__(self, core_laws: CoreLaws, node_id: Optional[str] = None):
        super().__init__(capability_name="DataSimulation", core_laws=core_laws, node_id=node_id)
        # Add simulation-specific parameters if needed
        self.simulation_complexity = 0.5 # Example: affects cost or quality

    def _estimate_cost_factor(self, data: Any) -> float:
        """Override cost estimation for simulation parameters."""
        # Cost could depend on requested simulation size or complexity in 'data'
        if isinstance(data, dict):
             count = data.get('count', 10)
             complexity_mod = data.get('complexity', self.simulation_complexity)
             return count * complexity_mod * 2 # Make simulation cost higher
        return 20 # Default cost if parameters unclear

    def execute_capability(self, data: Any, **kwargs) -> Any:
        """
        Performs the data simulation.

        Args:
            data (Any): Expected to be a dictionary with simulation parameters, e.g.,
                        {'type': 'gaussian', 'count': 100, 'mean': 0, 'std_dev': 1}
            **kwargs: Additional parameters (not used here).

        Returns:
            Any: The generated data (e.g., a list of numbers).
        """
        if not isinstance(data, dict):
            logger.warning(f"{self.id}: Invalid data format for DataSimulation. Expected dict.")
            raise ValueError("Invalid input data format for simulation.")

        sim_type = data.get('type', 'uniform')
        count = data.get('count', 10)
        logger.info(f"{self.id}: Executing DataSimulation - Type: {sim_type}, Count: {count}")

        if sim_type == 'gaussian':
            mean = data.get('mean', 0)
            std_dev = data.get('std_dev', 1)
            simulated_data = np.random.normal(mean, std_dev, count).tolist()
        elif sim_type == 'uniform':
            low = data.get('low', 0)
            high = data.get('high', 1)
            simulated_data = np.random.uniform(low, high, count).tolist()
        else:
            logger.warning(f"{self.id}: Unsupported simulation type '{sim_type}'.")
            raise ValueError(f"Unsupported simulation type: {sim_type}")

        logger.info(f"{self.id}: Simulation generated {len(simulated_data)} data points.")
        return {"simulation_type": sim_type, "data": simulated_data}

# --- Example Usage for CapabilityNode ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    laws = CoreLaws(learning_rate=0.1)

    # Test the DataSimulationNode subclass
    sim_node = DataSimulationNode(core_laws=laws)
    print(f"\n--- Testing {sim_node.capability_name} Node (ID: {sim_node.id}) ---")
    print(f"Initial Energy: {sim_node.state.energy}, Initial Capability: {sim_node.capabilities[sim_node.capability_name]:.3f}")

    sim_params = {'type': 'gaussian', 'count': 50, 'mean': 5, 'std_dev': 2}
    result = sim_node.process(sim_params)

    print("\nSimulation Result:")
    if result:
         print(f"  Status: {result.get('status')}")
         if result.get('status') == 'success':
              print(f"  Capability: {result.get('capability')}")
              sim_result = result.get('result', {})
              print(f"  Sim Type: {sim_result.get('simulation_type')}")
              print(f"  Generated Points: {len(sim_result.get('data', []))}")
              # print(f"  Sample Data: {sim_result.get('data', [])[:5]}...") # Uncomment to see sample data
              print(f"  Processing Time (ms): {result.get('processing_time_ms')}")
         else:
              print(f"  Message: {result.get('message')}")
    else:
         print("  Processing returned None.")

    print(f"\nFinal Energy: {sim_node.state.energy}, Final Capability: {sim_node.capabilities[sim_node.capability_name]:.3f}")
