# core/NodeManager.py
import logging
import heapq
from typing import Dict, Any, Optional, List

from .BaseNode import BaseNode

logger = logging.getLogger(__name__)

class NodeManager:
    """
    Enhanced NodeManager that manages nodes and their connections,
    including methods to apply stress and simulate network-wide effects.
    """

    def __init__(self):
        self.nodes: Dict[str, BaseNode] = {}

    def register_node(self, node: BaseNode):
        if node.id in self.nodes:
            logger.warning(f"Node with ID {node.id} already registered. Replacing.")
        self.nodes[node.id] = node
        logger.info(f"Registered Node: {node.id} (Type: {node.node_type})")

    def unregister_node(self, node_id: str):
        if node_id in self.nodes:
            del self.nodes[node_id]
            logger.info(f"Unregistered Node: {node_id}")
            # Handle removing connections from other nodes to this one
            for node in self.nodes.values():
                node.remove_connection(node_id)
        else:
            logger.warning(f"Attempted to unregister non-existent node: {node_id}")

    def get_node(self, node_id: str) -> Optional[BaseNode]:
        return self.nodes.get(node_id)

    def get_node_count(self) -> int:
        return len(self.nodes)

    def get_all_statuses(self) -> Dict[str, Dict[str, Any]]:
        return {node_id: node.get_state_summary() for node_id, node in self.nodes.items()}

    def find_nodes_by_type(self, node_type: str) -> List[BaseNode]:
        return [node for node in self.nodes.values() if node.node_type == node_type]

    def connect_nodes(self, node_id1: str, node_id2: str, relationship: str = "related", strength: float = 1.0):
        """
        Connects two nodes, creating a relationship.
        """
        node1 = self.get_node(node_id1)
        node2 = self.get_node(node_id2)
        if node1 and node2:
            node1.add_connection(node_id2, relationship, strength)
            node2.add_connection(node_id1, relationship, strength)
            logger.info(f"Nodes {node_id1} and {node_id2} connected with '{relationship}' (strength: {strength:.2f}).")
        else:
            logger.warning(f"One or both node IDs not found for connection: {node_id1}, {node_id2}")

    def apply_network_stress(self, stress_pattern: Dict[str, float]):
        """
        Applies a "stress pattern" to the network, affecting node energy
        and connection stress.
        """
        for node_id, node in self.nodes.items():
            # Apply stress directly to the node
            node.apply_stress(stress_pattern.get(node_id, 0.0))

            # Update stress on connections
            for connected_node_id in node.get_connection_ids():
                connected_node_stress = stress_pattern.get(connected_node_id, 0.0)
                stress_delta = connected_node_stress * 0.1
                node.update_connection_stress(connected_node_id, stress_delta)

    def get_highest_priority_nodes(self, count: int = 5) -> List[BaseNode]:
        sorted_nodes = sorted(self.nodes.values(), key=lambda n: n.state.energy, reverse=True)
        return sorted_nodes[:count]
