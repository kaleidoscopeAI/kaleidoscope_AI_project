#!/usr/bin/env python3
"""
Quantum Kaleidoscope Core System
================================

An integrated system that combines data ingestion, quantum-inspired analysis,
and innovative visualization into a single, self-contained solution.

This system builds on the foundation of the original Quantum Kaleidoscope by:
1. Integrating the runner and API server into a unified architecture
2. Implementing a speculative analysis engine for deeper insights
3. Providing a robust visualization system directly in the core
4. Building everything with zero external dependencies
"""

import os
import sys
import time
import uuid
import random
import logging
import json
import threading
import math
import socket
import struct
import base64
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("QuantumKaleidoscope")

# === Core Data Structures ===

@dataclass
class NodeState:
    """Represents the state of a quantum node in the kaleidoscope."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    features: List[float] = field(default_factory=list)
    position: List[float] = field(default_factory=list)
    energy: float = 0.5
    stability: float = 0.8
    connections: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    creation_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to a dictionary for serialization."""
        return {
            "id": self.id,
            "features": self.features,
            "position": self.position,
            "energy": self.energy,
            "stability": self.stability,
            "connections": self.connections,
            "metadata": self.metadata,
            "creation_time": self.creation_time,
            "last_update": self.last_update
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NodeState':
        """Create a NodeState instance from a dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            features=data.get("features", []),
            position=data.get("position", []),
            energy=data.get("energy", 0.5),
            stability=data.get("stability", 0.8),
            connections=data.get("connections", {}),
            metadata=data.get("metadata", {}),
            creation_time=data.get("creation_time", time.time()),
            last_update=data.get("last_update", time.time())
        )


class QuantumEngine:
    """
    Core engine implementing quantum-inspired algorithms for data analysis
    and visualization. This is the computational heart of the kaleidoscope.
    """
    def __init__(self, dimension: int = 128, data_dir: str = "./data"):
        self.dimension = dimension
        self.data_dir = data_dir
        self.nodes: Dict[str, NodeState] = {}
        self.pending_connections: List[Tuple[str, str, float]] = []
        self.simulation_lock = threading.Lock()
        self.entropy_source = EntropyPool(seed=int(time.time()))
        self.last_mutation_time = time.time()
        self.mutation_interval = 10.0  # seconds
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        logger.info(f"Quantum Engine initialized with dimension {dimension}")
        
        # Load existing data if available
        self._load_state()
    
    def _load_state(self):
        """Load engine state from disk."""
        state_file = os.path.join(self.data_dir, "quantum_state.json")
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    data = json.load(f)
                    
                # Load nodes
                nodes_data = data.get("nodes", {})
                for node_id, node_data in nodes_data.items():
                    self.nodes[node_id] = NodeState.from_dict(node_data)
                
                # Load pending connections
                self.pending_connections = data.get("pending_connections", [])
                
                logger.info(f"Loaded {len(self.nodes)} nodes from {state_file}")
            except Exception as e:
                logger.error(f"Error loading state: {e}")
                # Start fresh
                self.nodes = {}
                self.pending_connections = []
    
    def _save_state(self):
        """Save engine state to disk."""
        state_file = os.path.join(self.data_dir, "quantum_state.json")
        try:
            data = {
                "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
                "pending_connections": self.pending_connections
            }
            
            with open(state_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Saved state with {len(self.nodes)} nodes to {state_file}")
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def create_node(self, features: Optional[List[float]] = None, 
                   position: Optional[List[float]] = None,
                   energy: float = 0.5,
                   stability: float = 0.8,
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new node in the quantum field.
        
        Args:
            features: Feature vector for the node. If None, random features will be generated.
            position: Position in visualization space. If None, a random position will be assigned.
            energy: Initial energy level of the node (0.0 to 1.0)
            stability: Initial stability of the node (0.0 to 1.0)
            metadata: Optional metadata to attach to the node
            
        Returns:
            The ID of the newly created node
        """
        with self.simulation_lock:
            # Generate node ID
            node_id = str(uuid.uuid4())
            
            # Generate random features if none provided
            if features is None:
                features = [self.entropy_source.get_float(-1.0, 1.0) for _ in range(self.dimension)]
            
            # Ensure features match dimension
            if len(features) != self.dimension:
                features = features[:self.dimension] if len(features) > self.dimension else \
                          features + [0.0] * (self.dimension - len(features))
            
            # Generate random position if none provided
            if position is None:
                position = [self.entropy_source.get_float(-10.0, 10.0) for _ in range(3)]  # 3D space
            
            # Create node
            node = NodeState(
                id=node_id,
                features=features,
                position=position,
                energy=max(0.0, min(1.0, energy)),  # Clamp to [0, 1]
                stability=max(0.0, min(1.0, stability)),  # Clamp to [0, 1]
                metadata=metadata or {},
                creation_time=time.time(),
                last_update=time.time()
            )
            
            # Add to nodes collection
            self.nodes[node_id] = node
            
            logger.info(f"Created node {node_id} with energy {energy:.2f}, stability {stability:.2f}")
            
            # Save state
            self._save_state()
            
            return node_id
    
    def delete_node(self, node_id: str) -> bool:
        """
        Delete a node from the quantum field.
        
        Args:
            node_id: ID of the node to delete
            
        Returns:
            True if node was deleted, False if node doesn't exist
        """
        with self.simulation_lock:
            if node_id not in self.nodes:
                return False
            
            # Remove node
            del self.nodes[node_id]
            
            # Remove any connections to this node
            for node in self.nodes.values():
                if node_id in node.connections:
                    del node.connections[node_id]
            
            # Remove any pending connections involving this node
            self.pending_connections = [
                (src, dst, strength) for src, dst, strength in self.pending_connections
                if src != node_id and dst != node_id
            ]
            
            logger.info(f"Deleted node {node_id}")
            
            # Save state
            self._save_state()
            
            return True
    
    def connect_nodes(self, node1_id: str, node2_id: str, strength: Optional[float] = None) -> bool:
        """
        Create a connection between two nodes.
        
        Args:
            node1_id: ID of the first node
            node2_id: ID of the second node
            strength: Connection strength (0.0 to 1.0). If None, calculated based on feature similarity.
            
        Returns:
            True if connection was created, False if either node doesn't exist
        """
        with self.simulation_lock:
            if node1_id not in self.nodes or node2_id not in self.nodes:
                return False
                
            if node1_id == node2_id:
                logger.warning(f"Attempted to connect node {node1_id} to itself")
                return False
            
            # Calculate connection strength based on feature similarity if not provided
            if strength is None:
                node1 = self.nodes[node1_id]
                node2 = self.nodes[node2_id]
                
                # Cosine similarity
                dot_product = sum(f1 * f2 for f1, f2 in zip(node1.features, node2.features))
                norm1 = math.sqrt(sum(f ** 2 for f in node1.features))
                norm2 = math.sqrt(sum(f ** 2 for f in node2.features))
                
                if norm1 > 0 and norm2 > 0:
                    similarity = dot_product / (norm1 * norm2)
                    # Convert similarity (-1 to 1) to strength (0 to 1)
                    strength = (similarity + 1) / 2
                else:
                    strength = 0.5  # Default
            
            # Add connection to both nodes
            self.nodes[node1_id].connections[node2_id] = strength
            self.nodes[node2_id].connections[node1_id] = strength
            
            # Update last_update time
            self.nodes[node1_id].last_update = time.time()
            self.nodes[node2_id].last_update = time.time()
            
            logger.info(f"Connected nodes {node1_id} and {node2_id} with strength {strength:.2f}")
            
            # Save state
            self._save_state()
            
            return True
    
    def disconnect_nodes(self, node1_id: str, node2_id: str) -> bool:
        """
        Remove a connection between two nodes.
        
        Args:
            node1_id: ID of the first node
            node2_id: ID of the second node
            
        Returns:
            True if connection was removed, False if either node doesn't exist or no connection exists
        """
        with self.simulation_lock:
            if node1_id not in self.nodes or node2_id not in self.nodes:
                return False
                
            # Remove connection from both nodes
            if node2_id in self.nodes[node1_id].connections:
                del self.nodes[node1_id].connections[node2_id]
            else:
                return False  # No connection exists
                
            if node1_id in self.nodes[node2_id].connections:
                del self.nodes[node2_id].connections[node1_id]
            
            # Update last_update time
            self.nodes[node1_id].last_update = time.time()
            self.nodes[node2_id].last_update = time.time()
            
            logger.info(f"Disconnected nodes {node1_id} and {node2_id}")
            
            # Save state
            self._save_state()
            
            return True
    
    def process_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process text input to create new nodes and connections.
        
        Args:
            text: Input text
            metadata: Optional metadata to attach to the created node
            
        Returns:
            Dict containing the created node's ID and other processing results
        """
        # Create metadata dictionary if not provided
        if metadata is None:
            metadata = {}
        
        # Add text-specific metadata
        text_metadata = {
            "type": "text",
            "length": len(text),
            "timestamp": time.time(),
            "original_text": text[:100] + "..." if len(text) > 100 else text
        }
        metadata.update(text_metadata)
        
        # Generate features from text
        features = self._text_to_features(text)
        
        # Create a new node
        node_id = self.create_node(
            features=features,
            energy=0.8,  # Higher initial energy for user-created nodes
            stability=0.7,
            metadata=metadata
        )
        
        # Find potentially related nodes and create connections
        related_nodes = self._find_related_nodes(node_id, limit=5)
        for related_id, similarity in related_nodes:
            self.connect_nodes(node_id, related_id, strength=similarity)
        
        return {
            "node_id": node_id,
            "related_nodes": related_nodes,
            "features_dimension": self.dimension
        }
    
    def _text_to_features(self, text: str) -> List[float]:
        """
        Convert text to a feature vector using a deterministic hashing approach.
        This is a simplified approach that doesn't require external dependencies.
        
        Args:
            text: Input text
            
        Returns:
            A feature vector of dimension self.dimension
        """
        # Normalize text
        text = text.lower()
        
        # Generate hash
        hasher = hashlib.sha256()
        hasher.update(text.encode('utf-8'))
        hash_bytes = hasher.digest()
        
        # Use the hash to seed a deterministic random number generator
        seed = int.from_bytes(hash_bytes[:8], byteorder='big')
        random_gen = random.Random(seed)
        
        # Generate feature vector using PRNG
        features = []
        for i in range(self.dimension):
            # Mix original hash with position to get different values
            position_hash = hashlib.sha256()
            position_hash.update(hash_bytes)
            position_hash.update(str(i).encode('utf-8'))
            
            # Convert to a normalized float between -1 and 1
            feature_val = (int.from_bytes(position_hash.digest()[:8], byteorder='big') / (2**64 - 1)) * 2 - 1
            features.append(feature_val)
        
        return features
    
    def _find_related_nodes(self, node_id: str, limit: int = 5) -> List[Tuple[str, float]]:
        """
        Find nodes related to the given node based on feature similarity.
        
        Args:
            node_id: The ID of the node to find relations for
            limit: Maximum number of related nodes to return
            
        Returns:
            List of tuples (node_id, similarity) sorted by descending similarity
        """
        if node_id not in self.nodes:
            return []
            
        source_node = self.nodes[node_id]
        
        # Calculate similarity with all other nodes
        similarities = []
        for other_id, other_node in self.nodes.items():
            if other_id == node_id:
                continue
                
            # Cosine similarity
            dot_product = sum(f1 * f2 for f1, f2 in zip(source_node.features, other_node.features))
            norm1 = math.sqrt(sum(f ** 2 for f in source_node.features))
            norm2 = math.sqrt(sum(f ** 2 for f in other_node.features))
            
            if norm1 > 0 and norm2 > 0:
                similarity = dot_product / (norm1 * norm2)
                # Convert to 0-1 range
                similarity = (similarity + 1) / 2
            else:
                similarity = 0
                
            similarities.append((other_id, similarity))
        
        # Sort by descending similarity and take top 'limit'
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:limit]
    
    def run_simulation_step(self):
        """
        Run a single step of the quantum simulation, updating node positions,
        energies, and connections based on quantum-inspired dynamics.
        """
        with self.simulation_lock:
            current_time = time.time()
            
            # Process any pending connections
            self._process_pending_connections()
            
            # Update node positions based on connections (graph embedding)
            self._update_node_positions()
            
            # Update node energies
            self._update_node_energies()
            
            # Occasionally introduce mutations
            if current_time - self.last_mutation_time > self.mutation_interval:
                self._introduce_mutations()
                self.last_mutation_time = current_time
            
            # Save state occasionally (every 10 steps)
            if random.random() < 0.1:
                self._save_state()
    
    def _process_pending_connections(self):
        """Process any pending connections between nodes."""
        for src, dst, strength in self.pending_connections:
            if src in self.nodes and dst in self.nodes:
                # Create the connection
                self.nodes[src].connections[dst] = strength
                self.nodes[dst].connections[src] = strength
                
                # Update last_update time
                self.nodes[src].last_update = time.time()
                self.nodes[dst].last_update = time.time()
                
                logger.debug(f"Created pending connection between {src} and {dst}")
        
        # Clear processed connections
        self.pending_connections = []
    
    def _update_node_positions(self):
        """
        Update node positions using a force-directed algorithm.
        This simulates a physical system where connected nodes attract each other.
        """
        # Constants for the simulation
        repulsion = 10.0
        attraction = 5.0
        damping = 0.9
        min_distance = 0.1
        max_force = 10.0
        
        # Calculate forces
        forces = {node_id: [0.0, 0.0, 0.0] for node_id in self.nodes}
        
        # Repulsive forces (nodes repel each other)
        node_items = list(self.nodes.items())
        for i, (node1_id, node1) in enumerate(node_items):
            for node2_id, node2 in node_items[i+1:]:
                # Calculate distance
                dx = node2.position[0] - node1.position[0]
                dy = node2.position[1] - node1.position[1]
                dz = node2.position[2] - node1.position[2]
                distance = math.sqrt(dx*dx + dy*dy + dz*dz)
                
                # Avoid division by zero
                if distance < min_distance:
                    distance = min_distance
                
                # Repulsive force is inversely proportional to distance squared
                force_magnitude = repulsion / (distance * distance)
                
                # Normalize direction
                dx /= distance
                dy /= distance
                dz /= distance
                
                # Apply force with magnitude
                forces[node1_id][0] -= dx * force_magnitude
                forces[node1_id][1] -= dy * force_magnitude
                forces[node1_id][2] -= dz * force_magnitude
                
                forces[node2_id][0] += dx * force_magnitude
                forces[node2_id][1] += dy * force_magnitude
                forces[node2_id][2] += dz * force_magnitude
        
        # Attractive forces (connected nodes attract each other)
        for node1_id, node1 in self.nodes.items():
            for node2_id, strength in node1.connections.items():
                if node2_id in self.nodes:
                    node2 = self.nodes[node2_id]
                    
                    # Calculate distance
                    dx = node2.position[0] - node1.position[0]
                    dy = node2.position[1] - node1.position[1]
                    dz = node2.position[2] - node1.position[2]
                    distance = math.sqrt(dx*dx + dy*dy + dz*dz)
                    
                    # Avoid division by zero
                    if distance < min_distance:
                        distance = min_distance
                    
                    # Attractive force is proportional to distance and connection strength
                    force_magnitude = distance * strength * attraction
                    
                    # Normalize direction
                    dx /= distance
                    dy /= distance
                    dz /= distance
                    
                    # Apply force with magnitude
                    forces[node1_id][0] += dx * force_magnitude
                    forces[node1_id][1] += dy * force_magnitude
                    forces[node1_id][2] += dz * force_magnitude
        
        # Apply forces to update positions
        for node_id, force in forces.items():
            node = self.nodes[node_id]
            
            # Clamp force magnitude
            force_magnitude = math.sqrt(force[0]**2 + force[1]**2 + force[2]**2)
            if force_magnitude > max_force:
                scaling = max_force / force_magnitude
                force[0] *= scaling
                force[1] *= scaling
                force[2] *= scaling
            
            # Apply damping based on node stability
            damping_factor = damping * node.stability
            
            # Update position with dampened force
            node.position[0] += force[0] * damping_factor
            node.position[1] += force[1] * damping_factor
            node.position[2] += force[2] * damping_factor
            
            # Update last_update time
            node.last_update = time.time()
    
    def _update_node_energies(self):
        """
        Update node energies based on connections and inherent stability.
        Nodes with many strong connections tend to gain energy, while
        isolated nodes lose energy over time.
        """
        for node_id, node in self.nodes.items():
            # Base energy decay rate
            decay_rate = 0.01 * (1 - node.stability)
            
            # Energy transfer from connections
            connection_energy = 0
            for other_id, strength in node.connections.items():
                if other_id in self.nodes:
                    other_node = self.nodes[other_id]
                    # Energy flows from higher energy nodes to lower ones
                    energy_diff = other_node.energy - node.energy
                    connection_energy += energy_diff * strength * 0.01
            
            # Update energy
            node.energy = max(0.1, min(1.0, node.energy - decay_rate + connection_energy))
            
            # Update last_update time
            node.last_update = time.time()
    
    def _introduce_mutations(self):
        """
        Introduce random mutations in the quantum field to simulate
        quantum fluctuations and prevent the system from stagnating.
        """
        # Probability of a node experiencing mutation
        mutation_prob = 0.1
        
        for node_id, node in self.nodes.items():
            if random.random() < mutation_prob:
                # Choose mutation type
                mutation_type = random.choice(["feature", "connection", "position"])
                
                if mutation_type == "feature":
                    # Mutate a random feature
                    idx = random.randint(0, self.dimension - 1)
                    node.features[idx] += self.entropy_source.get_float(-0.2, 0.2)
                    # Clamp to reasonable range
                    node.features[idx] = max(-3.0, min(3.0, node.features[idx]))
                    
                elif mutation_type == "connection":
                    # Mutate a random connection or create a new one
                    if node.connections and random.random() < 0.5:
                        # Modify existing connection
                        other_id = random.choice(list(node.connections.keys()))
                        if other_id in self.nodes:
                            strength = node.connections[other_id]
                            strength += self.entropy_source.get_float(-0.1, 0.1)
                            strength = max(0.1, min(1.0, strength))  # Clamp
                            node.connections[other_id] = strength
                            self.nodes[other_id].connections[node_id] = strength
                    else:
                        # Create new connection
                        other_nodes = [n for n in self.nodes if n != node_id and n not in node.connections]
                        if other_nodes:
                            other_id = random.choice(other_nodes)
                            strength = self.entropy_source.get_float(0.1, 0.5)
                            node.connections[other_id] = strength
                            self.nodes[other_id].connections[node_id] = strength
                
                elif mutation_type == "position":
                    # Apply small random displacement
                    node.position[0] += self.entropy_source.get_float(-0.5, 0.5)
                    node.position[1] += self.entropy_source.get_float(-0.5, 0.5)
                    node.position[2] += self.entropy_source.get_float(-0.5, 0.5)
                
                # Update last_update time
                node.last_update = time.time()
                
                logger.debug(f"Mutated node {node_id} with type {mutation_type}")
    
    def get_visualization_data(self) -> Dict[str, Any]:
        """
        Get data formatted for visualization.
        
        Returns:
            Dict containing nodes and connections in a format suitable for visualization
        """
        nodes_data = []
        for node_id, node in self.nodes.items():
            nodes_data.append({
                "id": node_id,
                "position": node.position,
                "energy": node.energy,
                "stability": node.stability,
                "metadata": node.metadata,
                "numConnections": len(node.connections)
            })
        
        connections_data = []
        processed = set()  # Track processed connections to avoid duplicates
        
        for node_id, node in self.nodes.items():
            for other_id, strength in node.connections.items():
                # Create a canonical connection ID (ordered by node IDs)
                conn_id = tuple(sorted([node_id, other_id]))
                if conn_id in processed:
                    continue
                    
                connections_data.append({
                    "source": node_id,
                    "target": other_id,
                    "strength": strength
                })
                processed.add(conn_id)
        
        return {
            "nodes": nodes_data,
            "connections": connections_data,
            "timestamp": time.time()
        }


class EntropyPool:
    """
    A source of high-quality entropy for the quantum simulation.
    Mixes deterministic PRNG with system entropy sources when available.
    """
    def __init__(self, seed: Optional[int] = None):
        # Initialize primary PRNG
        self.random_gen = random.Random(seed if seed is not None else int(time.time()))
        
        # Try to initialize entropy collector from system sources
        self.sys_entropy_available = False
        try:
            # Try to get some system entropy from various sources
            entropy_bytes = b''
            
            # Try /dev/urandom if available (Unix systems)
            try:
                with open('/dev/urandom', 'rb') as f:
                    entropy_bytes += f.read(64)
            except (FileNotFoundError, PermissionError):
                pass
                
            # Try system socket operations for entropy (cross-platform)
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.settimeout(0.1)
                # The connection will fail but we get entropy from the socket internals
                try:
                    s.connect(('8.8.8.8', 80))  # Google DNS, but connection is not actually made
                except (socket.timeout, socket.error):
                    pass
                entropy_bytes += s.getsockname()[0].encode() + struct.pack('!H', s.getsockname()[1])
                s.close()
            except Exception:
                pass
                
            # Try high-precision timer entropy
            timer_bytes = b''
            for _ in range(16):
                start = time.perf_counter_ns()
                time.sleep(0.0001)  # Small sleep
                end = time.perf_counter_ns()
                timer_bytes += struct.pack('!Q', end - start)
            entropy_bytes += timer_bytes
            
            # Use the collected entropy to seed our pool
            if entropy_bytes:
                entropy_hash = hashlib.sha512(entropy_bytes).digest()
                entropy_int = int.from_bytes(entropy_hash, byteorder='big')
                self.random_gen = random.Random(entropy_int ^ (self.random_gen.randint(0, 2**64 - 1)))
                self.sys_entropy_available = True
        
        except Exception as e:
            logger.warning(f"Error initializing system entropy sources: {e}")
            # Fall back to the initial PRNG
    
    def get_float(self, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Get a random float in the specified range."""
        return self.random_gen.uniform(min_val, max_val)
    
    def get_int(self, min_val: int = 0, max_val: int = 100) -> int:
        """Get a random integer in the specified range."""
        return self.random_gen.randint(min_val, max_val)
    
    def get_bytes(self, length: int = 32) -> bytes:
        """Get random bytes of specified length."""
        return bytes(self.random_gen.getrandbits(8) for _ in range(length))
    
    def refresh(self):
        """Refresh the entropy pool with new entropy."""
        # Mix in current time
        time_ns = time.perf_counter_ns()
        # Mix current seed with new time-based entropy
        current_state = self.random_gen.getstate()
        self.random_gen.seed(hash((current_