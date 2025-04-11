# core/laws.py
import logging
import random
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class GrowthLaws:
    """
    Implements the fundamental growth laws that govern the Kaleidoscope AI system.
    These laws determine energy dynamics, replication conditions, and stress responses.
    """
    
    def __init__(self):
        # Constants for energy dynamics
        self.ENERGY_THRESHOLD_REPRODUCE = 85.0  # Energy needed to trigger replication
        self.MIN_STABILITY_REPRODUCE = 0.6  # Minimum stability needed for replication
        self.ENERGY_DECAY_BASE = 0.02  # Base energy decay rate per cycle
        self.STRESS_RECOVERY_RATE = 0.1  # Base stress recovery per cycle
        
        # Dynamic parameters (can evolve over time)
        self.system_parameters = {
            "mutation_rate": 0.01,  # Base mutation rate
            "energy_transfer_ratio": 0.4,  # Portion of energy transferred to child
            "stress_spread_factor": 0.2,  # How much stress spreads through connections
            "novelty_reward": 0.1,  # Energy bonus for processing new patterns
            "complexity_cost": 0.05  # Additional energy cost for complex nodes
        }
        
        logger.info(f"Growth Laws initialized with reproduction threshold: {self.ENERGY_THRESHOLD_REPRODUCE}")
    
    def apply(self, nodes, external_stress=0.0):
        """
        Apply growth laws to all nodes.
        
        Args:
            nodes: Iterable of BaseNode objects
            external_stress: Global stress factor (0.0-1.0)
        """
        if not nodes:
            return
            
        # Apply natural energy dynamics to each node
        for node in nodes:
            self._apply_to_node(node, external_stress)
            
        # Track which nodes have set reproduction flags
        ready_nodes = [n for n in nodes if hasattr(n, 'state') and n.state.ready_to_replicate]
        if ready_nodes:
            logger.info(f"{len(ready_nodes)} nodes ready to replicate")
    
    def _apply_to_node(self, node, external_stress):
        """Apply growth laws to a single node."""
        if not hasattr(node, 'state'):
            logger.warning(f"Node {getattr(node, 'id', 'unknown')} has no state attribute")
            return
            
        # Apply external stress
        if external_stress > 0:
            node.apply_stress(external_stress)
            
        # Check replication conditions
        if (node.state.energy >= self.ENERGY_THRESHOLD_REPRODUCE and 
                node.state.stability >= self.MIN_STABILITY_REPRODUCE and
                not node.state.ready_to_replicate):
            node.state.ready_to_replicate = True
            logger.debug(f"Node {node.id} now ready to replicate. Energy: {node.state.energy:.1f}")
            
        # Apply natural stress recovery
        if node.state.stress_level > 0:
            recovery = node.state.stress_level * self.STRESS_RECOVERY_RATE
            node.state.stress_level = max(0, node.state.stress_level - recovery)
