# core/BaseNode.py
import logging
import random
import uuid
import json
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class CoreState:
    """Encapsulates the dynamic state of a node."""
    energy: float = random.uniform(50.0, 100.0)
    ready_to_replicate: bool = False
    dna: Dict[str, Any] = field(default_factory=lambda: {
        "mutation_rate": 0.01,
        "energy_efficiency": random.uniform(0.8, 1.2),
        "stress_resistance": random.uniform(0.8, 1.2)
    })
    metadata: Dict[str, Any] = field(default_factory=lambda: {
        "created_at": time.time(),
        "last_processed": 0.0,
        "processed_count": 0,
        "lineage": []
    })
    stress_level: float = 0.0

class BaseNode:
    """
    Foundation class for all nodes in the Kaleidoscope system.
    Includes state, basic energy dynamics, connections,
    and methods for stress response and replication preparation.
    """

    CoreState = CoreState

    def __init__(self, node_id: Optional[str] = None, core_laws: Optional[Any] = None, 
                 node_type: str = "base", parent_id: Optional[str] = None, 
                 initial_dna: Optional[Dict] = None):
        """
        Initializes a BaseNode.
        Args:
            node_id: Unique identifier. Auto-generated if None.
            core_laws: Instance of GrowthLaws governing behavior.
            node_type: String identifier for the node's type/function.
            parent_id: ID of the node from which this one replicated (if any).
            initial_dna: Specific DNA to initialize with, otherwise defaults are used/mutated.
        """
        self.id = node_id if node_id else str(uuid.uuid4())
        self.core_laws = core_laws
        self.node_type = node_type
        self.state = self.CoreState()

        if initial_dna:
            self.state.dna.update(initial_dna)

        if parent_id:
            self.state.metadata["lineage"] = [parent_id]

        self.connections: Dict[str, Dict[str, Any]] = {}
        logger.debug(f"Initialized Node: {self.id} (Type: {self.node_type}, Parent: {parent_id})")

    def process_data(self, data: Any, **kwargs) -> Dict[str, Any]:
        """
        Core data processing method (intended to be overridden by subclasses).
        Base implementation performs minimal actions and updates state.

        Args:
            data: The input data to process.
            **kwargs: Additional context or parameters.

        Returns:
            A dictionary containing processing results.
        """
        start_time = time.time()
        logger.debug(f"Node {self.id} processing data...")

        try:
            data_size = len(str(data)) if data else 1
        except:
            data_size = 100
        energy_cost = data_size * 0.01 / self.state.dna.get("energy_efficiency", 1.0)
        self.state.energy -= energy_cost
        self.state.energy = max(0, self.state.energy)

        processed_result = {"info": "Base processing complete", "data_echo": str(data)[:100]}

        self.state.metadata["last_processed"] = time.time()
        self.state.metadata["processed_count"] += 1

        processing_time = time.time() - start_time
        logger.debug(f"Node {self.id} finished processing in {processing_time:.4f}s. Energy: {self.state.energy:.2f}")

        return {
            "status": "success",
            "node_id": self.id,
            "node_type": self.node_type,
            "result": processed_result,
            "energy_consumed": energy_cost,
            "energy_remaining": self.state.energy,
            "processing_time": processing_time,
        }

    def apply_stress(self, stress_amount: float):
        """
        Applies external stress to the node, affecting its internal state.
        Stress resistance from DNA mitigates the effect.
        """
        if stress_amount <= 0:
            return
        resistance = self.state.dna.get("stress_resistance", 1.0)
        resistance = max(0.1, resistance)
        effective_stress = stress_amount / resistance
        self.state.stress_level += effective_stress
        energy_drain = effective_stress * 0.5
        self.state.energy -= energy_drain
        self.state.energy = max(0, self.state.energy)
        logger.debug(f"Node {self.id} received stress {stress_amount:.2f}, effective: {effective_stress:.2f}. Energy: {self.state.energy:.2f}, Internal Stress: {self.state.stress_level:.2f}")

    def add_connection(self, target_node_id: str, relationship: str = "related", 
                      strength: float = 1.0, initial_stress: float = 0.0):
        """Adds or updates a connection to another node."""
        if target_node_id == self.id:
            logger.warning(f"Node {self.id} cannot connect to itself.")
            return
        strength = max(0.0, min(1.0, strength))
        initial_stress = max(0.0, initial_stress)

        self.connections[target_node_id] = {
            "relationship": relationship,
            "strength": strength,
            "stress": initial_stress
        }
        logger.debug(f"Node {self.id} connected to {target_node_id} (Rel: {relationship}, Str: {strength:.2f})")

    def remove_connection(self, target_node_id: str):
        """Removes a connection to another node."""
        if target_node_id in self.connections:
            del self.connections[target_node_id]
            logger.debug(f"Node {self.id} disconnected from {target_node_id}")

    def update_connection_stress(self, target_node_id: str, stress_delta: float):
        """Updates the stress level on a specific connection."""
        if target_node_id in self.connections:
            conn = self.connections[target_node_id]
            conn["stress"] = max(0.0, conn["stress"] + stress_delta)
            logger.debug(f"Node {self.id} connection to {target_node_id} stress updated by {stress_delta:.2f}. New stress: {conn['stress']:.2f}")

            if conn["stress"] > 5.0:
                conn["strength"] = max(0.0, conn["strength"] * 0.98)
                self.apply_stress(conn["stress"] * 0.1)

    def get_connection_ids(self) -> List[str]:
        """Returns a list of IDs of connected nodes."""
        return list(self.connections.keys())

    def get_state_summary(self) -> Dict[str, Any]:
        """Returns a concise summary of the node's state."""
        return {
            "id": self.id,
            "type": self.node_type,
            "energy": round(self.state.energy, 2),
            "stress": round(self.state.stress_level, 2),
            "ready_to_replicate": self.state.ready_to_replicate,
            "connections": len(self.connections),
            "processed_count": self.state.metadata["processed_count"],
            "created_at": self.state.metadata.get("created_at"),
            "lineage": self.state.metadata.get("lineage", []),
        }

    def can_replicate(self) -> bool:
        """Checks if the node meets conditions to trigger replication signal."""
        return self.state.ready_to_replicate

    def replicate_dna(self) -> Dict[str, Any]:
        """
        Returns a copy of the node's DNA, potentially with mutations.
        The mutation logic itself is simple here; could be expanded.
        """
        new_dna = self.state.dna.copy()
        mutation_rate = new_dna.get("mutation_rate", 0.01)

        for gene, value in new_dna.items():
            if isinstance(value, (int, float)) and random.random() < mutation_rate:
                if gene == "energy_efficiency":
                    change_factor = random.uniform(0.95, 1.05)
                    new_dna[gene] = max(0.1, value * change_factor)
                elif gene == "stress_resistance":
                    change_factor = random.uniform(0.95, 1.05)
                    new_dna[gene] = max(0.1, value * change_factor)
                elif gene == "mutation_rate":
                    change_factor = random.uniform(0.8, 1.2)
                    new_dna[gene] = max(0.001, min(0.1, value * change_factor))
                else:
                    change_factor = random.uniform(0.9, 1.1)
                    new_dna[gene] = value * change_factor

                if new_dna[gene] != value:
                    logger.debug(f"Node {self.id} DNA mutation in gene '{gene}': {value:.3f} -> {new_dna[gene]:.3f}")

        return new_dna

    def decay(self, base_decay_rate: float = 0.05):
        """Applies natural energy decay and stress reduction over time."""
        efficiency = max(0.1, self.state.dna.get("energy_efficiency", 1.0))
        decay_amount = base_decay_rate * (1 / efficiency)
        self.state.energy -= decay_amount
        self.state.energy = max(0, self.state.energy)

        stress_recovery_rate = 0.05
        stress_recovery = self.state.stress_level * stress_recovery_rate
        self.state.stress_level = max(0.0, self.state.stress_level - stress_recovery)

        if self.state.energy <= 0:
            logger.debug(f"Node {self.id} has no energy remaining.")
