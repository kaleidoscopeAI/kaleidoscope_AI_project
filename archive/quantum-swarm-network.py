#!/usr/bin/env python3
"""
Quantum Swarm Intelligence Network (QSIN)

A groundbreaking distributed computing framework that combines:
1. Quantum-inspired state synchronization
2. Swarm intelligence for self-optimization
3. Topological analysis for emergent behavior detection
4. Autonomous node replication through network discovery

This system creates a self-organizing network that can adapt to computing challenges
through emergent intelligence arising from simple node interactions.
"""

import os
import sys
import time
import json
import uuid
import logging
import asyncio
import threading
import numpy as np
import networkx as nx
import websockets
import paramiko
import hashlib
import random
import subprocess
import socket
from enum import Enum, auto
from typing import Dict, List, Tuple, Set, Any, Optional, Union, Callable
from dataclasses import dataclass, field, asdict
from collections import deque
import concurrent.futures
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logger = logging.getLogger("qsin")

# ===========================
# === Constants & Configs ===
# ===========================

HILBERT_SPACE_DIM = 64        # Quantum state dimensionality
NODE_SYNC_INTERVAL = 3.0      # Seconds between node sync operations
QUANTUM_DECOHERENCE_RATE = 0.03  # Rate at which quantum states decohere
ENTANGLEMENT_THRESHOLD = 0.75  # Threshold for considering nodes entangled
COHERENCE_THRESHOLD = 0.6     # Minimum coherence for node synchronization
REPLICATION_ENERGY_THRESHOLD = 100.0  # Energy needed for node replication
COMPLEXITY_EMERGENCE_THRESHOLD = 0.8  # Threshold for detecting emergent complexity
DEFAULT_PORT = 8765           # Default WebSocket port
SSH_TIMEOUT = 10              # SSH connection timeout in seconds
MAX_DISCOVERY_ATTEMPTS = 5    # Maximum number of network discovery attempts
LOCAL_BUFFER_SIZE = 100       # Size of local operation buffer

# =====================================
# === Quantum State Representations ===
# =====================================

class WavefunctionCollapse(Enum):
    """Possible collapse states for quantum wave functions"""
    COHERENT = auto()       # Maintains quantum coherence
    CLASSICAL = auto()      # Collapsed to classical state
    ENTANGLED = auto()      # Entangled with other nodes
    SUPERPOSED = auto()     # In superposition of multiple states
    TELEPORTED = auto()     # State has been teleported

@dataclass
class QuantumState:
    """Represents a quantum state in the Hilbert space"""
    
    dimension: int = HILBERT_SPACE_DIM
    state: Optional[np.ndarray] = None
    fidelity: float = 1.0
    creation_time: float = field(default_factory=time.time)
    collapse_status: WavefunctionCollapse = WavefunctionCollapse.COHERENT
    entangled_with: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        """Initialize state if not provided"""
        if self.state is None:
            # Create random pure state
            state_vector = np.random.normal(0, 1, self.dimension) + 1j * np.random.normal(0, 1, self.dimension)
            self.state = state_vector / np.linalg.norm(state_vector)
    
    def evolve(self, hamiltonian: np.ndarray, dt: float) -> None:
        """Evolve quantum state according to Schrödinger equation"""
        # U = exp(-i*H*dt)
        evolution_operator = np.exp(-1j * hamiltonian * dt)
        self.state = evolution_operator @ self.state
        # Renormalize to handle numerical errors
        self.state = self.state / np.linalg.norm(self.state)
    
    def apply_noise(self, dt: float) -> None:
        """Apply decoherence to the quantum state"""
        elapsed_time = time.time() - self.creation_time
        decoherence = np.exp(-QUANTUM_DECOHERENCE_RATE * elapsed_time)
        
        # Mix with random state based on decoherence
        if decoherence < 1.0:
            random_state = np.random.normal(0, 1, self.dimension) + 1j * np.random.normal(0, 1, self.dimension)
            random_state = random_state / np.linalg.norm(random_state)
            
            # Apply partial decoherence
            self.state = decoherence * self.state + np.sqrt(1 - decoherence**2) * random_state
            self.state = self.state / np.linalg.norm(self.state)
            
            # Update fidelity
            self.fidelity = decoherence
    
    def measure(self, observable: np.ndarray) -> float:
        """Measure an observable on the quantum state"""
        # <ψ|O|ψ>
        expectation = np.real(np.conjugate(self.state) @ observable @ self.state)
        return expectation
    
    def measure_with_collapse(self, observable: np.ndarray) -> Tuple[float, 'QuantumState']:
        """Measure with wavefunction collapse, returning result and new state"""
        # Calculate expectation value
        expectation = self.measure(observable)
        
        # Perform eigendecomposition of the observable
        eigenvalues, eigenvectors = np.linalg.eigh(observable)
        
        # Calculate probabilities based on Born rule
        probabilities = np.abs(np.conjugate(eigenvectors.T) @ self.state)**2
        
        # Choose an eigenvalue based on probabilities
        result_idx = np.random.choice(len(eigenvalues), p=probabilities.real)
        result = eigenvalues[result_idx]
        
        # State collapses to corresponding eigenvector
        new_state = QuantumState(dimension=self.dimension)
        new_state.state = eigenvectors[:, result_idx]
        new_state.collapse_status = WavefunctionCollapse.CLASSICAL
        
        return result, new_state
    
    def entangle_with(self, other_id: str) -> None:
        """Mark this state as entangled with another node"""
        self.entangled_with.add(other_id)
        if len(self.entangled_with) > 0:
            self.collapse_status = WavefunctionCollapse.ENTANGLED
    
    def compute_entropy(self) -> float:
        """Calculate the von Neumann entropy of the state"""
        # S = -Tr(ρ ln ρ) where ρ is the density matrix |ψ⟩⟨ψ|
        # For pure states, entropy is 0. For mixed states, it's positive.
        if self.fidelity > 0.99:  # Pure state
            return 0.0
        
        # Create density matrix ρ = |ψ⟩⟨ψ|
        density_matrix = np.outer(self.state, np.conjugate(self.state))
        
        # Calculate eigenvalues of density matrix
        eigenvalues = np.linalg.eigvalsh(density_matrix)
        
        # Keep only positive eigenvalues (numerical issues may give tiny negative values)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        
        # Calculate entropy: S = -Σ λ_i ln λ_i
        entropy = -np.sum(eigenvalues * np.log(eigenvalues))
        return float(entropy)
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize the quantum state for transmission"""
        return {
            "real": self.state.real.tolist(),
            "imag": self.state.imag.tolist(),
            "fidelity": self.fidelity,
            "creation_time": self.creation_time,
            "collapse_status": self.collapse_status.name,
            "entangled_with": list(self.entangled_with)
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'QuantumState':
        """Deserialize a quantum state from received data"""
        state = cls(dimension=len(data["real"]))
        state.state = np.array(data["real"]) + 1j * np.array(data["imag"])
        state.fidelity = data["fidelity"]
        state.creation_time = data["creation_time"]
        state.collapse_status = WavefunctionCollapse[data["collapse_status"]]
        state.entangled_with = set(data["entangled_with"])
        return state

# =======================================
# === Quantum Observables & Operators ===
# =======================================

class ObservableGenerator:
    """Generates quantum observables for different measurements"""
    
    def __init__(self, dimension: int = HILBERT_SPACE_DIM):
        self.dimension = dimension
        self.cache = {}  # Cache generated observables
    
    def get_random_observable(self) -> np.ndarray:
        """Generate a random Hermitian observable"""
        # Create random Hermitian (H = H†)
        H = np.random.normal(0, 1, (self.dimension, self.dimension)) + \
            1j * np.random.normal(0, 1, (self.dimension, self.dimension))
        H = 0.5 * (H + H.conj().T)  # Ensure Hermitian
        return H
    
    def get_energy_observable(self) -> np.ndarray:
        """Get observable corresponding to energy measurement"""
        if "energy" in self.cache:
            return self.cache["energy"]
        
        # Create an energy observable with increasing eigenvalues
        diagonal = np.arange(self.dimension) / self.dimension
        H = np.diag(diagonal)
        
        # Add small off-diagonal elements for "interactions"
        for i in range(self.dimension-1):
            H[i, i+1] = H[i+1, i] = 0.1 / self.dimension
        
        self.cache["energy"] = H
        return H
    
    def get_coherence_observable(self) -> np.ndarray:
        """Get observable to measure quantum coherence"""
        if "coherence" in self.cache:
            return self.cache["coherence"]
        
        # Create a sparse matrix with high eigenvalues for coherent states
        H = np.zeros((self.dimension, self.dimension), dtype=complex)
        
        # Add elements that favor superposition states
        for i in range(self.dimension):
            for j in range(i+1, self.dimension):
                phase = np.exp(1j * 2 * np.pi * (i+j) / self.dimension)
                H[i, j] = 0.1 * phase
                H[j, i] = 0.1 * np.conjugate(phase)
        
        # Add diagonal
        for i in range(self.dimension):
            H[i, i] = 1.0 - 0.5 * (i / self.dimension)
        
        self.cache["coherence"] = H
        return H
    
    def get_entanglement_observable(self, node_id: str, target_id: str) -> np.ndarray:
        """Get observable to measure entanglement between two nodes"""
        cache_key = f"entanglement_{hash(node_id)}_{hash(target_id)}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Create entanglement observable based on node IDs
        H = np.zeros((self.dimension, self.dimension), dtype=complex)
        
        # Use hash of node IDs to create a unique observable
        seed = int(hashlib.md5((node_id + target_id).encode()).hexdigest(), 16) % 10000
        np.random.seed(seed)
        
        # Create Bell-like measurement projectors
        # Simplified - in a real system, would depend on actual entangled states
        for i in range(0, self.dimension // 2):
            j = self.dimension - i - 1
            # Create projector onto maximally entangled state |i,j⟩ + |j,i⟩
            proj = np.zeros((self.dimension, self.dimension), dtype=complex)
            
            # Set elements for |i⟩⟨j| and |j⟩⟨i|
            proj[i, j] = proj[j, i] = 1.0 / np.sqrt(2)
            
            # Add to Hamiltonian with random strength
            strength = 0.5 + 0.5 * np.random.random()
            H += strength * proj
        
        # Ensure Hermitian
        H = 0.5 * (H + H.conj().T)
        
        self.cache[cache_key] = H
        return H
    
    def get_pauli_operators(self) -> Dict[str, np.ndarray]:
        """Get generalized Pauli operators for the Hilbert space"""
        if "pauli" in self.cache:
            return self.cache["pauli"]
        
        # For qubits, we need log2(dimension) qubits
        n_qubits = int(np.ceil(np.log2(self.dimension)))
        actual_dim = 2**n_qubits
        
        # Pauli matrices
        sigma_x = np.array([[0, 1], [1, 0]])
        sigma_y = np.array([[0, -1j], [1j, 0]])
        sigma_z = np.array([[1, 0], [0, -1]])
        identity = np.eye(2)
        
        operators = {}
        
        # Create Pauli operators for each qubit
        for i in range(n_qubits):
            # Create X operator for qubit i
            X_i = np.array(1.0)
            for j in range(n_qubits):
                if j == i:
                    X_i = np.kron(X_i, sigma_x)
                else:
                    X_i = np.kron(X_i, identity)
            operators[f"X_{i}"] = X_i[:self.dimension, :self.dimension]  # Truncate if needed
            
            # Create Z operator for qubit i
            Z_i = np.array(1.0)
            for j in range(n_qubits):
                if j == i:
                    Z_i = np.kron(Z_i, sigma_z)
                else:
                    Z_i = np.kron(Z_i, identity)
            operators[f"Z_{i}"] = Z_i[:self.dimension, :self.dimension]  # Truncate if needed
            
            # Create Y operator for qubit i
            Y_i = np.array(1.0)
            for j in range(n_qubits):
                if j == i:
                    Y_i = np.kron(Y_i, sigma_y)
                else:
                    Y_i = np.kron(Y_i, identity)
            operators[f"Y_{i}"] = Y_i[:self.dimension, :self.dimension]  # Truncate if needed
        
        self.cache["pauli"] = operators
        return operators

class HamiltonianGenerator:
    """Generates Hamiltonians for quantum evolution"""
    
    def __init__(self, dimension: int = HILBERT_SPACE_DIM):
        self.dimension = dimension
        self.observable_gen = ObservableGenerator(dimension)
    
    def get_base_hamiltonian(self) -> np.ndarray:
        """Create a basic Hamiltonian for time evolution"""
        # Start with energy observable as the base
        H_base = self.observable_gen.get_energy_observable()
        return H_base
    
    def get_network_hamiltonian(self, network_graph: nx.Graph, node_mapping: Dict[str, int]) -> np.ndarray:
        """Create a Hamiltonian based on network topology"""
        H = np.zeros((self.dimension, self.dimension), dtype=complex)
        
        # Add interaction terms for connected nodes
        for node1, node2, attrs in network_graph.edges(data=True):
            if node1 in node_mapping and node2 in node_mapping:
                # Get indices in the Hamiltonian
                i = node_mapping[node1] % self.dimension
                j = node_mapping[node2] % self.dimension
                
                # Get interaction strength (default to 0.5)
                strength = attrs.get('weight', 0.5)
                
                # Add interaction term
                phase = np.exp(1j * np.pi * (i*j) / self.dimension)
                interaction = strength * phase
                H[i, j] += interaction
                H[j, i] += np.conjugate(interaction)
        
        # Add diagonal terms
        for node, idx in node_mapping.items():
            i = idx % self.dimension
            # Node degree affects diagonal energy
            degree = network_graph.degree(node)
            H[i, i] = 1.0 + 0.1 * degree
        
        # Ensure Hermitian (H = H†)
        H = 0.5 * (H + H.conj().T)
        
        return H
    
    def get_evolution_hamiltonian(self, network_graph: nx.Graph, node_mapping: Dict[str, int]) -> np.ndarray:
        """Create a complete Hamiltonian for time evolution"""
        # Combine base and network Hamiltonians
        H_base = self.get_base_hamiltonian()
        H_network = self.get_network_hamiltonian(network_graph, node_mapping)
        
        # Weight between isolation and network effects
        alpha = 0.7  # Favor network effects
        H = (1.0 - alpha) * H_base + alpha * H_network
        
        # Ensure Hermitian
        H = 0.5 * (H + H.conj().T)
        
        return H

# =============================
# === Swarm Intelligence ====
# =============================

@dataclass
class TaskResult:
    """Result from a quantum swarm computation task"""
    task_id: str
    node_id: str
    result_value: float
    confidence: float
    timestamp: float = field(default_factory=time.time)
    computation_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class SwarmOperator:
    """Implements swarm optimization algorithms using quantum states"""
    
    def __init__(self, dimension: int = HILBERT_SPACE_DIM):
        self.dimension = dimension
        self.observables = ObservableGenerator(dimension)
        self.particle_positions = []  # For particle swarm optimization
        self.particle_velocities = []
        self.particle_best_positions = []
        self.particle_best_values = []
        self.global_best_position = None
        self.global_best_value = -float('inf')
        self.iteration = 0
    
    def initialize_swarm(self, swarm_size: int) -> None:
        """Initialize particle swarm for optimization"""
        self.particle_positions = [np.random.rand(self.dimension) for _ in range(swarm_size)]
        self.particle_velocities = [np.random.rand(self.dimension) * 0.1 - 0.05 for _ in range(swarm_size)]
        self.particle_best_positions = self.particle_positions.copy()
        self.particle_best_values = [-float('inf')] * swarm_size
        self.global_best_position = None
        self.global_best_value = -float('inf')
        self.iteration = 0
    
    def objective_function(self, position: np.ndarray, quantum_state: QuantumState) -> float:
        """Evaluate objective function using quantum measurement"""
        # Convert position to an observable
        observable = np.diag(position)  # Simplistic mapping
        
        # Measure the observable on the quantum state
        result = quantum_state.measure(observable)
        
        # Objective is to maximize this measurement
        return result
    
    def update_swarm(self, quantum_state: QuantumState, inertia: float = 0.7, 
                      personal_coef: float = 1.5, global_coef: float = 1.5) -> Dict[str, Any]:
        """Update swarm particles using quantum-influenced PSO algorithm"""
        if not self.particle_positions:
            self.initialize_swarm(10)  # Default swarm size
        
        for i in range(len(self.particle_positions)):
            # Evaluate current position
            value = self.objective_function(self.particle_positions[i], quantum_state)
            
            # Update personal best
            if value > self.particle_best_values[i]:
                self.particle_best_values[i] = value
                self.particle_best_positions[i] = self.particle_positions[i].copy()
            
            # Update global best
            if value > self.global_best_value:
                self.global_best_value = value
                self.global_best_position = self.particle_positions[i].copy()
        
        # Update positions and velocities
        for i in range(len(self.particle_positions)):
            # Generate random coefficients
            r1 = np.random.rand(self.dimension)
            r2 = np.random.rand(self.dimension)
            
            # Calculate "quantum" influence
            if quantum_state.fidelity > 0.7:
                # Highly coherent state - introduce quantum tunneling
                tunnel_prob = 0.1 * quantum_state.fidelity
                if np.random.rand() < tunnel_prob:
                    # Quantum tunneling: jump to random position near global best
                    self.particle_positions[i] = self.global_best_position + np.random.normal(0, 0.2, self.dimension)
                    self.particle_velocities[i] = np.zeros(self.dimension)
                    continue
            
            # Standard PSO update
            # v = w*v + c1*r1*(p_best - x) + c2*r2*(g_best - x)
            self.particle_velocities[i] = (
                inertia * self.particle_velocities[i] +
                personal_coef * r1 * (self.particle_best_positions[i] - self.particle_positions[i]) +
                global_coef * r2 * (self.global_best_position - self.particle_positions[i])
            )
            
            # Apply quantum effects to velocity
            if quantum_state.collapse_status == WavefunctionCollapse.ENTANGLED:
                # Entangled states introduce correlations in velocities
                quantum_factor = 0.3 * quantum_state.fidelity
                avg_velocity = np.mean([v for v in self.particle_velocities], axis=0)
                self.particle_velocities[i] = (1 - quantum_factor) * self.particle_velocities[i] + quantum_factor * avg_velocity
            
            # Update position: x = x + v
            self.particle_positions[i] += self.particle_velocities[i]
            
            # Ensure positions stay within bounds [0, 1]
            self.particle_positions[i] = np.clip(self.particle_positions[i], 0, 1)
        
        self.iteration += 1
        
        return {
            "best_value": self.global_best_value,
            "best_position": self.global_best_position.tolist() if self.global_best_position is not None else None,
            "iteration": self.iteration,
            "average_value": np.mean([self.objective_function(p, quantum_state) for p in self.particle_positions])
        }
    
    def solve_optimization_task(self, task_id: str, node_id: str, quantum_state: QuantumState, 
                                iterations: int = 20) -> TaskResult:
        """Solve an optimization task using quantum-enhanced swarm intelligence"""
        # Initialize swarm
        self.initialize_swarm(15)  # Use 15 particles
        
        start_time = time.time()
        best_result = -float('inf')
        best_position = None
        
        for _ in range(iterations):
            # Update swarm
            result = self.update_swarm(quantum_state)
            
            if result["best_value"] > best_result:
                best_result = result["best_value"]
                best_position = result["best_position"]
        
        computation_time = time.time() - start_time
        
        # Calculate confidence based on convergence and quantum state fidelity
        convergence = 1.0 - np.std([self.particle_best_values[i] for i in range(len(self.particle_positions))]) / abs(best_result)
        confidence = min(0.9, 0.5 * convergence + 0.5 * quantum_state.fidelity)
        
        return TaskResult(
            task_id=task_id,
            node_id=node_id,
            result_value=best_result,
            confidence=confidence,
            computation_time=computation_time,
            metadata={
                "iterations": iterations,
                "final_position": best_position,
                "swarm_size": len(self.particle_positions)
            }
        )

# ===============================
# === Network Node Structure ====
# ===============================

@dataclass
class QSINConfig:
    """Configuration for a QSIN Node"""
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    node_name: str = field(default_factory=lambda: f"QSIN-Node-{uuid.uuid4().hex[:8]}")
    server_url: str = f"ws://localhost:{DEFAULT_PORT}"
    server_mode: bool = False
    port: int = DEFAULT_PORT
    host: str = "0.0.0.0"
    dimension: int = HILBERT_SPACE_DIM
    initial_energy: float = 20.0
    replication_threshold: float = REPLICATION_ENERGY_THRESHOLD
    discovery_enabled: bool = True
    max_discovery_attempts: int = MAX_DISCOVERY_ATTEMPTS
    persistence_path: str = "./qsin_data"
    ssh_username: str = ""
    ssh_password: str = ""
    ssh_key_path: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QSINConfig':
        """Create from dictionary"""
        return cls(**data)

@dataclass
class NodeState:
    """Current state of a QSIN node"""
    node_id: str
    status: str = "initializing"
    energy: float = 0.0
    last_active: float = field(default_factory=time.time)
    quantum_state: Optional[QuantumState] = None
    connected_nodes: Set[str] = field(default_factory=set)
    entangled_nodes: Set[str] = field(default_factory=set)
    processed_tasks: int = 0
    successful_replications: int = 0
    total_uptime: float = 0.0
    
    def __post_init__(self):
        """Initialize with defaults if needed"""
        if self.quantum_state is None:
            self.quantum_state = QuantumState()

@dataclass
class NetworkMessage:
    """Message for inter-node communication"""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    message_type: str = "ping"
    sender_id: str = ""
    receiver_id: str = ""  # Empty for broadcast
    timestamp: float = field(default_factory=time.time)
    content: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for transmission"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NetworkMessage':
        """Create from dictionary"""
        return cls(**data)

class NetworkDiscovery:
    """Network discovery and node detection"""
    
    def __init__(self, config: QSINConfig):
        self.config = config
        self.discovered_nodes = {}  # ip -> node_info
        self.successful_scans = 0
    
    def scan_subnet(self, subnet: str) -> List[str]:
        """Scan subnet for potential nodes"""
        if not subnet:
            # Try to determine local subnet
            subnet = self._get_local_subnet()
            
        logger.info(f"Scanning subnet: {subnet}")
        
        try:
            # Simple ping scan implementation
            if "/" in subnet:  # CIDR notation
                base, prefix = subnet.split("/")
                octets = base.split(".")
                base_ip = ".".join(octets[:3])
                
                live_hosts = []
                for i in range(1, 255):
                    ip = f"{base_ip}.{i}"
                    if self._ping_host(ip):
                        live_hosts.append(ip)
                        
                self.successful_scans += 1
                return live_hosts
            else:
                return [subnet] if self._ping_host(subnet) else []
        except Exception as e:
            logger.error(f"Subnet scan error: {e}")
            return []
    
    def _get_local_subnet(self) -> str:
        """Get local subnet for scanning"""
        try:
            # Get local IP address
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # Doesn't need to be reachable, just to determine interface
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            
            # Extract subnet
            octets = local_ip.split(".")
            subnet = f"{octets[0]}.{octets[1]}.{octets[2]}.0/24"
            return subnet
        except:
            # Fallback to common private subnet
            return "192.168.1.0/24"
    
    def _ping_host(self, ip: str) -> bool:
        """Check if host is reachable via ping"""
        try:
            # Use subprocess to ping with timeout
            param = "-n" if os.name == "nt" else "-c"
            cmd = ["ping", param, "1", "-W", "1", ip]
            result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return result.returncode == 0
        except:
            return False
    
    def check_for_node(self, ip: str, port: int = DEFAULT_PORT) -> Optional[Dict[str, Any]]:
        """Check if IP has a QSIN node running"""
        try:
            # Try to connect to WebSocket
            ws_url = f"ws://{ip}:{port}/qsin"
            
            # Create connection with short timeout
            loop = asyncio.get_event_loop()
            try:
                # Connect with 2 second timeout
                ws = await asyncio.wait_for(
                    websockets.connect(ws_url),
                    timeout=2.0
                )
                
                # Send discovery ping
                discovery_msg = {
                    "type": "discovery_ping",
                    "source_id": self.config.node_id,
                    "source_name": self.config.node_name,
                    "timestamp": time.time()
                }
                
                await ws.send(json.dumps(discovery_msg))
                
                # Wait for response
                response = await asyncio.wait_for(
                    ws.recv(),
                    timeout=2.0
                )
                
                # Parse response
                data = json.loads(response)
                if data.get("type") == "discovery_pong":
                    node_info = {
                        "node_id": data.get("node_id", "unknown"),
                        "node_name": data.get("node_name", "unknown"),
                        "ip_address": ip,
                        "port": port,
                        "last_seen": time.time(),
                        "capabilities": data.get("capabilities", [])
                    }
                    
                    # Store discovered node
                    self.discovered_nodes[ip] = node_info
                    
                    # Close connection
                    await ws.close()
                    
                    return node_info
            except asyncio.TimeoutError:
                # Connection timeout
                return None
            except Exception as e:
                # Other connection error
                return None
                
        except Exception as e:
            logger.debug(f"Error checking for node at {ip}:{port}: {e}")
            return None
    
    async def discover_nodes(self, subnet: Optional[str] = None) -> List[Dict[str, Any]]:
        """Discover QSIN nodes on the network"""
        # Scan subnet for live hosts
        live_hosts = self.scan_subnet(subnet or "")
        
        # Check each host for QSIN node
        discovered = []
        
        # Use gather with limit to avoid too many concurrent connections
        tasks = []
        for ip in live_hosts:
            task = asyncio.create_task(self.check_for_node(ip))
            tasks.append(task)
            
            # Limit concurrent tasks
            if len(tasks) >= 10:
                # Wait for some tasks to complete
                done, tasks = await asyncio.wait(
                    tasks, 
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Process completed tasks
                for task in done:
                    result = task.result()
                    if result:
                        discovered.append(result)
        
        # Wait for remaining tasks
        if tasks:
            done, _ = await asyncio.wait(tasks)
            for task in done:
                result = task.result()
                if result:
                    discovered.append(result)
        
        logger.info(f"Discovered {len(discovered)} QSIN nodes on network")
        return discovered

class SSHDeployer:
    """Handles SSH-based deployment of QSIN nodes"""
    
    def __init__(self, config: QSINConfig):
        self.config = config
        self.ssh_connections = {}  # hostname -> ssh_client
    
    def _create_ssh_client(self, hostname: str, port: int = 22) -> Optional[paramiko.SSHClient]:
        """Create SSH client for remote host"""
        if hostname in self.ssh_connections:
            # Check if connection is still active
            ssh = self.ssh_connections[hostname]
            transport = ssh.get_transport()
            if transport and transport.is_active():
                return ssh
        
        try:
            # Create new SSH connection
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # Connect to the node using available credentials
            connect_kwargs = {
                "hostname": hostname,
                "port": port,
                "timeout": SSH_TIMEOUT
            }
            
            # Use password if provided
            if self.config.ssh_username and self.config.ssh_password:
                connect_kwargs["username"] = self.config.ssh_username
                connect_kwargs["password"] = self.config.ssh_password
            # Use key if provided
            elif self.config.ssh_username and self.config.ssh_key_path:
                connect_kwargs["username"] = self.config.ssh_username
                connect_kwargs["key_filename"] = self.config.ssh_key_path
            else:
                # Try current user without password (key-based auth)
                connect_kwargs["username"] = os.environ.get("USER", "root")
            
            ssh.connect(**connect_kwargs)
            
            # Store connection
            self.ssh_connections[hostname] = ssh
            return ssh
            
        except Exception as e:
            logger.error(f"SSH connection error for {hostname}: {e}")
            return None
    
    def _execute_command(self, ssh_client: paramiko.SSHClient, command: str) -> Tuple[bool, str]:
        """Execute command on remote host"""
        try:
            stdin, stdout, stderr = ssh_client.exec_command(command, timeout=30)
            stdout_str = stdout.read().decode('utf-8').strip()
            stderr_str = stderr.read().decode('utf-8').strip()
            
            exit_code = stdout.channel.recv_exit_status()
            success = exit_code == 0
            
            if stderr_str and not success:
                logger.warning(f"Command stderr: {stderr_str}")
                return False, stderr_str
            
            return success, stdout_str
            
        except Exception as e:
            logger.error(f"Command execution error: {e}")
            return False, str(e)
    
    async def deploy_node(self, hostname: str) -> bool:
        """Deploy QSIN node to remote host"""
        logger.info(f"Deploying QSIN node to {hostname}")
        
        # Create SSH connection
        ssh = self._create_ssh_client(hostname)
        if not ssh:
            return False
        
        try:
            # Check if Python is installed
            success, _ = self._execute_command(ssh, "which python3")
            if not success:
                logger.error(f"Python3 not found on {hostname}")
                return False
            
            # Install dependencies
            install_cmd = (
                "pip3 install --user websockets numpy networkx"
            )
            success, output = self._execute_command(ssh, install_cmd)
            if not success:
                logger.error(f"Failed to install dependencies on {hostname}: {output}")
                return False
            
            # Create deployment directory
            deploy_dir = f"~/qsin_node_{int(time.time())}"
            mkdir_cmd = f"mkdir -p {deploy_dir}"
            success, _ = self._execute_command(ssh, mkdir_cmd)
            if not success:
                logger.error(f"Failed to create deployment directory on {hostname}")
                return False
            
            # Generate node configuration
            child_config = QSINConfig(
                node_name=f"QSIN-Child-{uuid.uuid4().hex[:8]}",
                server_url=f"ws://{self._get_local_ip()}:{self.config.port}",
                server_mode=False,
                initial_energy=self.config.initial_energy * 0.5,  # Child gets half the energy
                replication_threshold=self.config.replication_threshold
            )
            
            config_json = json.dumps(child_config.to_dict(), indent=2)
            config_cmd = f"echo '{config_json}' > {deploy_dir}/config.json"
            success, _ = self._execute_command(ssh, config_cmd)
            if not success:
                logger.error(f"Failed to create config on {hostname}")
                return False
            
            # Copy node script (in a real implementation, we would use SFTP)
            # For this example, we'll simulate with a simplified node script
            node_script = self._generate_node_script()
            script_cmd = f"cat > {deploy_dir}/qsin_node.py << 'EOL'\n{node_script}\nEOL"
            success, _ = self._execute_command(ssh, script_cmd)
            if not success:
                logger.error(f"Failed to create node script on {hostname}")
                return False
            
            # Make script executable
            chmod_cmd = f"chmod +x {deploy_dir}/qsin_node.py"
            success, _ = self._execute_command(ssh, chmod_cmd)
            if not success:
                logger.error(f"Failed to make script executable on {hostname}")
                return False
            
            # Start node in background
            start_cmd = (
                f"cd {deploy_dir} && "
                f"nohup python3 qsin_node.py "
                f"--config config.json "
                f"> qsin_node.log 2>&1 &"
            )
            
            success, output = self._execute_command(ssh, start_cmd)
            if not success:
                logger.error(f"Failed to start node on {hostname}: {output}")
                return False
            
            # Get node process ID
            success, pid = self._execute_command(ssh, "pgrep -f qsin_node.py")
            if success and pid:
                logger.info(f"QSIN node started on {hostname} with PID {pid}")
                return True
            else:
                logger.warning(f"QSIN node started on {hostname} but couldn't get PID")
                return True
                
        except Exception as e:
            logger.error(f"Error deploying QSIN node on {hostname}: {e}")
            return False
    
    def _get_local_ip(self) -> str:
        """Get local IP address"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            return local_ip
        except:
            return "127.0.0.1"
    
    def _generate_node_script(self) -> str:
        """Generate minimal node script for deployment"""
        # In a real implementation, the full node code would be packaged and deployed
        # Here we create a simplified version for demonstration
        return """#!/usr/bin/env python3
import asyncio
import websockets
import json
import time
import uuid
import random
import sys
import os
import argparse
import logging
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("qsin-node")

@dataclass
class QSINConfig:
    node_id: str = str(uuid.uuid4())
    node_name: str = f"QSIN-Node-{uuid.uuid4().hex[:8]}"
    server_url: str = "ws://localhost:8765"
    server_mode: bool = False
    port: int = 8765
    host: str = "0.0.0.0"
    dimension: int = 64
    initial_energy: float = 20.0
    replication_threshold: float = 100.0
    discovery_enabled: bool = True
    
    @classmethod
    def from_dict(cls, data):
        return cls(**data)

class QSINNode:
    def __init__(self, config: QSINConfig):
        self.config = config
        self.energy = config.initial_energy
        self.server_connection = None
        self.is_running = False
        self.connected = False
    
    async def connect_to_server(self):
        try:
            self.server_connection = await websockets.connect(self.config.server_url)
            
            # Register with server
            register_msg = {
                "type": "register",
                "node_id": self.config.node_id,
                "node_name": self.config.node_name,
                "energy": self.energy,
                "timestamp": time.time()
            }
            await self.server_connection.send(json.dumps(register_msg))
            
            # Wait for response
            response = await self.server_connection.recv()
            data = json.loads(response)
            
            if data.get("type") == "register_ack":
                self.connected = True
                logger.info(f"Connected to QSIN server as {self.config.node_name}")
                return True
            else:
                logger.error(f"Registration failed: {data}")
                return False
        except Exception as e:
            logger.error(f"Connection error: {e}")
            self.connected = False
            return False
    
    async def process_energy(self):
        """Simulate energy processing and growth"""
        while self.is_running:
            # Increase energy over time
            self.energy += random.uniform(1.0, 3.0)
            
            # Report energy to server
            if self.connected:
                try:
                    update_msg = {
                        "type": "energy_update",
                        "node_id": self.config.node_id,
                        "energy": self.energy,
                        "timestamp": time.time()
                    }
                    await self.server_connection.send(json.dumps(update_msg))
                except:
                    self.connected = False
            
            # Sleep for a bit
            await asyncio.sleep(5)
    
    async def process_messages(self):
        """Process messages from the server"""
        while self.is_running and self.connected:
            try:
                message = await self.server_connection.recv()
                data = json.loads(message)
                
                # Handle message based on type
                msg_type = data.get("type", "")
                
                if msg_type == "ping":
                    # Respond to ping
                    pong_msg = {
                        "type": "pong",
                        "node_id": self.config.node_id,
                        "timestamp": time.time()
                    }
                    await self.server_connection.send(json.dumps(pong_msg))
                
                elif msg_type == "task_request":
                    # Simulate task processing
                    task_id = data.get("task_id", "")
                    
                    # Process task (simulate computation)
                    await asyncio.sleep(random.uniform(0.5, 2.0))
                    
                    # Send result
                    result_msg = {
                        "type": "task_result",
                        "node_id": self.config.node_id,
                        "task_id": task_id,
                        "result": random.random(),
                        "timestamp": time.time()
                    }
                    await self.server_connection.send(json.dumps(result_msg))
                
                elif msg_type == "discovery_ping":
                    # Respond to discovery ping
                    pong_msg = {
                        "type": "discovery_pong",
                        "node_id": self.config.node_id,
                        "node_name": self.config.node_name,
                        "timestamp": time.time(),
                        "capabilities": ["compute", "storage"]
                    }
                    await self.server_connection.send(json.dumps(pong_msg))
            
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                self.connected = False
                break
    
    async def reconnect_loop(self):
        """Try to reconnect if connection is lost"""
        while self.is_running:
            if not self.connected:
                logger.info("Attempting to reconnect...")
                await self.connect_to_server()
            
            # Wait before checking again
            await asyncio.sleep(5)
    
    async def run(self):
        """Run the node"""
        self.is_running = True
        
        # Connect to server
        connected = await self.connect_to_server()
        if not connected:
            logger.error("Failed to connect to server")
            return
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self.process_energy()),
            asyncio.create_task(self.process_messages()),
            asyncio.create_task(self.reconnect_loop())
        ]
        
        # Wait for tasks to complete (or error)
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def stop(self):
        """Stop the node"""
        self.is_running = False

async def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="QSIN Node")
    parser.add_argument("--config", help="Path to config JSON file")
    args = parser.parse_args()
    
    # Load config
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_data = json.load(f)
        config = QSINConfig.from_dict(config_data)
    else:
        # Use default config
        config = QSINConfig()
    
    # Create and run node
    node = QSINNode(config)
    
    try:
        await node.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        node.stop()

if __name__ == "__main__":
    asyncio.run(main())
"""

# ============================
# === Core Node Implementation
# ============================

class QSINNode:
    """Implementation of a QSIN node"""
    
    def __init__(self, config: QSINConfig):
        self.config = config
        self.node_id = config.node_id
        self.node_name = config.node_name
        
        # Initialize state
        self.state = NodeState(node_id=self.node_id, energy=config.initial_energy)
        
        # Quantum components
        self.quantum_state = QuantumState(dimension=config.dimension)
        self.hamiltonian_gen = HamiltonianGenerator(dimension=config.dimension)
        self.observable_gen = ObservableGenerator(dimension=config.dimension)
        self.swarm_operator = SwarmOperator(dimension=config.dimension)
        
        # Network components
        self.server_mode = config.server_mode
        self.server = None
        self.client_connection = None
        self.connected = False
        self.connections = {}  # node_id -> websocket
        
        # Networking data
        self.network_graph = nx.Graph()
        self.node_mapping = {}  # node_id -> index
        self.discovered_nodes = {}  # node_id -> node_info
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Shutdown signal
        self.shutdown_event = asyncio.Event()
        
        # Tasks
        self.tasks = set()
        
        # Operation buffer
        self.operation_buffer = []
        
        # For node replication
        self.network_discovery = NetworkDiscovery(config)
        self.ssh_deployer = SSHDeployer(config)
        
        # Initialize logger
        self.logger = logging.getLogger(f"qsin-node-{self.node_id[:8]}")
    
    async def start(self) -> None:
        """Start the node"""
        self.logger.info(f"Starting QSIN node: {self.node_name} ({self.node_id})")
        
        # Initialize network graph
        self.network_graph.add_node(
            self.node_id, 
            name=self.node_name, 
            type="self",
            energy=self.state.energy
        )
        
        # Start server or client based on mode
        if self.server_mode:
            await self._start_server()
        else:
            await self._start_client()
        
        # Start background tasks
        self.tasks.add(asyncio.create_task(self._quantum_evolution_loop()))
        self.tasks.add(asyncio.create_task(self._energy_processing_loop()))
        
        if self.config.discovery_enabled:
            self.tasks.add(asyncio.create_task(self._discovery_loop()))
        
        self.state.status = "active"
        self.logger.info(f"Node {self.node_name} started successfully")
    
    async def stop(self) -> None:
        """Stop the node"""
        self.logger.info(f"Stopping node {self.node_name}")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        # Close connections
        if self.server_mode and self.server:
            self.server.close()
            await self.server.wait_closed()
        elif self.client_connection:
            await self.client_connection.close()
        
        # Close all peer connections
        for conn in self.connections.values():
            await conn.close()
        
        self.state.status = "terminated"
        self.logger.info(f"Node {self.node_name} stopped")
    
    async def _start_server(self) -> None:
        """Start in server mode (listen for connections)"""
        self.logger.info(f"Starting QSIN server on {self.config.host}:{self.config.port}")
        
        try:
            # Start WebSocket server
            self.server = await websockets.serve(
                self._handle_connection,
                self.config.host,
                self.config.port
            )
            
            self.tasks.add(asyncio.create_task(self._server_maintenance_loop()))
            
            self.logger.info(f"QSIN server started on port {self.config.port}")
            
        except Exception as e:
            self.logger.error(f"Error starting server: {e}")
            raise
    
    async def _start_client(self) -> None:
        """Start in client mode (connect to server)"""
        self.logger.info(f"Starting QSIN client connecting to {self.config.server_url}")
        
        try:
            # Connect to server
            self.client_connection = await websockets.connect(self.config.server_url)
            
            # Register with server
            register_msg = NetworkMessage(
                message_type="register",
                sender_id=self.node_id,
                content={
                    "node_name": self.node_name,
                    "energy": self.state.energy,
                    "capabilities": ["compute", "storage", "entanglement"]
                }
            )
            
            await self.client_connection.send(json.dumps(register_msg.to_dict()))
            
            # Wait for registration acknowledgment
            response = await self.client_connection.recv()
            data = json.loads(response)
            
            if data.get("message_type") == "register_ack":
                self.connected = True
                self.logger.info(f"Connected to QSIN server as {self.node_name}")
                
                # Start message processing
                self.tasks.add(asyncio.create_task(self._client_message_loop()))
                self.tasks.add(asyncio.create_task(self._client_reconnect_loop()))
                
            else:
                self.logger.error(f"Registration failed: {data}")
                await self.client_connection.close()
                raise RuntimeError("Registration failed")
                
        except Exception as e:
            self.logger.error(f"Error starting client: {e}")
            raise
    
    async def _handle_connection(self, websocket, path) -> None:
        """Handle incoming connection in server mode"""
        try:
            # Receive initial message
            message = await websocket.recv()
            data = json.loads(message)
            
            # Convert to NetworkMessage
            if isinstance(data, dict) and "message_type" in data:
                msg = NetworkMessage.from_dict(data)
            else:
                # Legacy message format
                msg_type = data.get("type", "unknown")
                msg = NetworkMessage(
                    message_type=msg_type,
                    sender_id=data.get("node_id", "unknown"),
                    content=data
                )
            
            # Process based on message type
            if msg.message_type == "register":
                # Node registration
                node_id = msg.sender_id
                node_name = msg.content.get("node_name", f"Node-{node_id[:8]}")
                
                self.logger.info(f"Node registration: {node_name} ({node_id})")
                
                # Store connection
                self.connections[node_id] = websocket
                
                # Add to network graph
                self.network_graph.add_node(
                    node_id,
                    name=node_name,
                    type="client",
                    energy=msg.content.get("energy", 0.0),
                    last_seen=time.time()
                )
                
                # Acknowledge registration
                ack_msg = NetworkMessage(
                    message_type="register_ack",
                    sender_id=self.node_id,
                    receiver_id=node_id,
                    content={
                        "status": "success",
                        "server_name": self.node_name,
                        "timestamp": time.time()
                    }
                )
                
                await websocket.send(json.dumps(ack_msg.to_dict()))
                
                # Process messages from this node
                await self._process_node_messages(node_id, websocket)
                
            elif msg.message_type == "discovery_ping":
                # Network discovery ping
                source_id = msg.sender_id
                source_name = msg.content.get("source_name", f"Node-{source_id[:8]}")
                
                self.logger.debug(f"Discovery ping from {source_name} ({source_id})")
                
                # Respond with pong
                pong_msg = NetworkMessage(
                    message_type="discovery_pong",
                    sender_id=self.node_id,
                    receiver_id=source_id,
                    content={
                        "node_id": self.node_id,
                        "node_name": self.node_name,
                        "capabilities": ["compute", "storage", "entanglement"],
                        "timestamp": time.time()
                    }
                )
                
                await websocket.send(json.dumps(pong_msg.to_dict()))
                
            else:
                # Unknown message type
                self.logger.warning(f"Unknown initial message type: {msg.message_type}")
        
        except Exception as e:
            self.logger.error(f"Error handling connection: {e}")
            await websocket.close()
    
    async def _process_node_messages(self, node_id: str, websocket) -> None:
        """Process messages from a connected node"""
        try:
            async for message in websocket:
                # Parse message
                data = json.loads(message)
                
                # Convert to NetworkMessage
                if isinstance(data, dict) and "message_type" in data:
                    msg = NetworkMessage.from_dict(data)
                else:
                    # Legacy message format
                    msg_type = data.get("type", "unknown")
                    msg = NetworkMessage(
                        message_type=msg_type,
                        sender_id=node_id,
                        content=data
                    )
                
                # Handle based on message type
                response = await self._handle_message(msg)
                
                # Send response if any
                if response:
                    await websocket.send(json.dumps(response.to_dict()))
                
                # Update node in graph
                if node_id in self.network_graph:
                    self.network_graph.nodes[node_id]['last_seen'] = time.time()
                    
                    # Update energy if provided
                    if msg.message_type == "energy_update":
                        self.network_graph.nodes[node_id]['energy'] = msg.content.get("energy", 0.0)
                
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"Connection closed for node {node_id}")
        except Exception as e:
            self.logger.error(f"Error processing messages from {node_id}: {e}")
        finally:
            # Clean up connection
            if node_id in self.connections:
                del self.connections[node_id]
    
    async def _handle_message(self, msg: NetworkMessage) -> Optional[NetworkMessage]:
        """Handle a network message"""
        if msg.message_type == "ping":
            # Respond to ping
            return NetworkMessage(
                message_type="pong",
                sender_id=self.node_id,
                receiver_id=msg.sender_id,
                content={"timestamp": time.time()}
            )
        
        elif msg.message_type == "energy_update":
            # Node energy update
            node_id = msg.sender_id
            energy = msg.content.get("energy", 0.0)
            
            if node_id in self.network_graph:
                self.network_graph.nodes[node_id]["energy"] = energy
            
            return None
        
        elif msg.message_type == "quantum_state_update":
            # Quantum state update
            node_id = msg.sender_id
            
            if "state" in msg.content:
                try:
                    # Deserialize quantum state
                    remote_state = QuantumState.deserialize(msg.content["state"])
                    
                    # Process quantum state update
                    # For demonstration, we'll just calculate coherence
                    coherence = 0.0
                    if self.quantum_state and remote_state:
                        coherence = np.abs(np.vdot(self.quantum_state.state, remote_state.state))**2
                    
                    # Store coherence in graph edge
                    if node_id in self.network_graph:
                        if not self.network_graph.has_edge(self.node_id, node_id):
                            self.network_graph.add_edge(self.node_id, node_id, weight=coherence)
                        else:
                            self.network_graph[self.node_id][node_id]["weight"] = coherence
                        
                        # Update entanglement if coherence is high
                        if coherence > ENTANGLEMENT_THRESHOLD:
                            self.quantum_state.entangle_with(node_id)
                            self.state.entangled_nodes.add(node_id)
                    
                    return NetworkMessage(
                        message_type="quantum_state_ack",
                        sender_id=self.node_id,
                        receiver_id=msg.sender_id,
                        content={"coherence": coherence}
                    )
                    
                except Exception as e:
                    self.logger.error(f"Error processing quantum state update: {e}")
            
            return None
        
        elif msg.message_type == "task_request":
            # Process task request
            task_id = msg.content.get("task_id", str(uuid.uuid4()))
            task_type = msg.content.get("task_type", "optimization")
            
            self.logger.info(f"Received task request: {task_id} ({task_type})")
            
            # Process task in background
            asyncio.create_task(self._process_task(task_id, task_type, msg.sender_id))
            
            return NetworkMessage(
                message_type="task_ack",
                sender_id=self.node_id,
                receiver_id=msg.sender_id,
                content={
                    "task_id": task_id,
                    "status": "processing"
                }
            )
        
        elif msg.message_type == "task_result":
            # Process task result
            task_id = msg.content.get("task_id", "unknown")
            result = msg.content.get("result", 0.0)
            
            self.logger.info(f"Received task result from {msg.sender_id}: {task_id} = {result}")
            
            # Store result (in a real system, would have a task manager)
            # Acknowledge result
            return NetworkMessage(
                message_type="result_ack",
                sender_id=self.node_id,
                receiver_id=msg.sender_id,
                content={
                    "task_id": task_id,
                    "status": "received"
                }
            )
        
        elif msg.message_type == "entanglement_request":
            # Request for quantum entanglement
            target_id = msg.sender_id
            
            self.logger.info(f"Entanglement request from {target_id}")
            
            # Create entangled state
            success, coherence = await self._create_entanglement(target_id)
            
            if success:
                return NetworkMessage(
                    message_type="entanglement_success",
                    sender_id=self.node_id,
                    receiver_id=target_id,
                    content={
                        "coherence": coherence,
                        "timestamp": time.time()
                    }
                )
            else:
                return NetworkMessage(
                    message_type="entanglement_failure",
                    sender_id=self.node_id,
                    receiver_id=target_id,
                    content={
                        "reason": "Failed to create entanglement"
                    }
                )
        
        elif msg.message_type == "replication_request":
            # Request to replicate to new host
            target_host = msg.content.get("target_host")
            if not target_host:
                return NetworkMessage(
                    message_type="replication_failure",
                    sender_id=self.node_id,
                    receiver_id=msg.sender_id,
                    content={
                        "reason": "No target host specified"
                    }
                )
            
            # Start replication in background
            asyncio.create_task(self._replicate_to_host(target_host))
            
            return NetworkMessage(
                message_type="replication_ack",
                sender_id=self.node_id,
                receiver_id=msg.sender_id,
                content={
                    "target_host": target_host,
                    "status": "started"
                }
            )
        
        # Unknown message type
        return None
    
    async def _process_task(self, task_id: str, task_type: str, sender_id: str) -> None:
        """Process a compute task"""
        self.logger.info(f"Processing task {task_id} of type {task_type}")
        
        try:
            # Different task types
            if task_type == "optimization":
                # Use swarm optimization
                result = self.swarm_operator.solve_optimization_task(
                    task_id,
                    self.node_id,
                    self.quantum_state
                )
                
                # Convert result to message format
                result_msg = NetworkMessage(
                    message_type="task_result",
                    sender_id=self.node_id,
                    receiver_id=sender_id,
                    content={
                        "task_id": task_id,
                        "task_type": task_type,
                        "result_value": result.result_value,
                        "confidence": result.confidence,
                        "computation_time": result.computation_time,
                        "metadata": result.metadata
                    }
                )
                
                # Consume energy for task
                task_energy_cost = 10.0 + 0.1 * result.computation_time
                with self.lock:
                    self.state.energy -= task_energy_cost
                    self.state.energy = max(0.0, self.state.energy)
                    self.state.processed_tasks += 1
                
                # Send result
                if sender_id in self.connections:
                    await self.connections[sender_id].send(json.dumps(result_msg.to_dict()))
                elif self.client_connection and self.connected:
                    await self.client_connection.send(json.dumps(result_msg.to_dict()))
                
            elif task_type == "measurement":
                # Quantum measurement task
                observable_type = task_id.split("_")[0] if "_" in task_id else "random"
                
                # Get appropriate observable
                if observable_type == "energy":
                    observable = self.observable_gen.get_energy_observable()
                elif observable_type == "coherence":
                    observable = self.observable_gen.get_coherence_observable()
                else:
                    observable = self.observable_gen.get_random_observable()
                
                # Perform measurement (with collapse)
                measurement, new_state = self.quantum_state.measure_with_collapse(observable)
                
                # Send measurement result
                result_msg = NetworkMessage(
                    message_type="task_result",
                    sender_id=self.node_id,
                    receiver_id=sender_id,
                    content={
                        "task_id": task_id,
                        "task_type": task_type,
                        "result_value": float(measurement),
                        "confidence": float(new_state.fidelity),
                        "collapse_status": new_state.collapse_status.name
                    }
                )
                
                # Send result
                if sender_id in self.connections:
                    await self.connections[sender_id].send(json.dumps(result_msg.to_dict()))
                elif self.client_connection and self.connected:
                    await self.client_connection.send(json.dumps(result_msg.to_dict()))
                
                # Replace state with collapsed state
                self.quantum_state = new_state
            
            else:
                # Unknown task type
                self.logger.warning(f"Unknown task type: {task_type}")
                
                # Send error
                error_msg = NetworkMessage(
                    message_type="task_error",
                    sender_id=self.node_id,
                    receiver_id=sender_id,
                    content={
                        "task_id": task_id,
                        "error": f"Unknown task type: {task_type}"
                    }
                )
                
                # Send error
                if sender_id in self.connections:
                    await self.connections[sender_id].send(json.dumps(error_msg.to_dict()))
                elif self.client_connection and self.connected:
                    await self.client_connection.send(json.dumps(error_msg.to_dict()))
        
        except Exception as e:
            self.logger.error(f"Error processing task {task_id}: {e}")
            
            # Send error
            error_msg = NetworkMessage(
                message_type="task_error",
                sender_id=self.node_id,
                receiver_id=sender_id,
                content={
                    "task_id": task_id,
                    "error": str(e)
                }
            )
            
            # Send error
            if sender_id in self.connections:
                await self.connections[sender_id].send(json.dumps(error_msg.to_dict()))
            elif self.client_connection and self.connected:
                await self.client_connection.send(json.dumps(error_msg.to_dict()))
    
    async def _create_entanglement(self, target_id: str) -> Tuple[bool, float]:
        """Create quantum entanglement with another node"""
        self.logger.info(f"Creating entanglement with {target_id}")
        
        try:
            # Generate entanglement observable
            entanglement_obs = self.observable_gen.get_entanglement_observable(self.node_id, target_id)
            
            # Measure to create entanglement
            measurement, new_state = self.quantum_state.measure_with_collapse(entanglement_obs)
            
            # Set state to entangled
            new_state.entangle_with(target_id)
            new_state.collapse_status = WavefunctionCollapse.ENTANGLED
            
            # Replace quantum state
            self.quantum_state = new_state
            
            # Update node state
            self.state.entangled_nodes.add(target_id)
            
            # Calculate coherence (proxy for entanglement quality)
            coherence = measurement
            
            # Add to network graph
            if target_id in self.network_graph:
                if not self.network_graph.has_edge(self.node_id, target_id):
                    self.network_graph.add_edge(self.node_id, target_id, weight=coherence)
                else:
                    self.network_graph[self.node_id][target_id]["weight"] = coherence
                    self.network_graph[self.node_id][target_id]["entangled"] = True
            
            return True, coherence
            
        except Exception as e:
            self.logger.error(f"Error creating entanglement with {target_id}: {e}")
            return False, 0.0
    
    async def _quantum_evolution_loop(self) -> None:
        """Quantum state evolution background task"""
        self.logger.info("Starting quantum evolution loop")
        
        while not self.shutdown_event.is_set():
            try:
                # Update node mapping
                self.node_mapping = {n: i for i, n in enumerate(self.network_graph.nodes())}
                
                # Create Hamiltonian
                hamiltonian = self.hamiltonian_gen.get_evolution_hamiltonian(
                    self.network_graph,
                    self.node_mapping
                )
                
                # Evolve quantum state
                self.quantum_state.evolve(hamiltonian, dt=0.1)
                
                # Apply decoherence
                self.quantum_state.apply_noise(dt=0.1)
                
                # Occasionally share state with connected nodes
                if random.random() < 0.2:  # 20% chance each cycle
                    await self._share_quantum_state()
                
                # Wait before next evolution step
                await asyncio.sleep(1.0)
                
            except Exception as e:
                self.logger.error(f"Error in quantum evolution loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _share_quantum_state(self) -> None:
        """Share quantum state with connected nodes"""
        # Serialize quantum state
        state_data = self.quantum_state.serialize()
        
        # Create message
        state_msg = NetworkMessage(
            message_type="quantum_state_update",
            sender_id=self.node_id,
            content={
                "state": state_data
            }
        )
        
        # Convert to JSON
        state_json = json.dumps(state_msg.to_dict())
        
        # Share with all connected nodes in server mode
        if self.server_mode:
            # Send to all connections
            for node_id, conn in list(self.connections.items()):
                try:
                    await conn.send(state_json)
                except:
                    # Connection error, will be cleaned up in maintenance loop
                    pass
        
        # Share with server in client mode
        elif self.client_connection and self.connected:
            try:
                await self.client_connection.send(state_json)
            except:
                # Connection error, will attempt reconnect
                self.connected = False
    
    async def _energy_processing_loop(self) -> None:
        """Energy processing and growth background task"""
        self.logger.info("Starting energy processing loop")
        
        while not self.shutdown_event.is_set():
            try:
                with self.lock:
                    # Calculate energy gain based on quantum state
                    # Higher coherence = higher energy gain
                    base_gain = 1.0
                    coherence_factor = self.quantum_state.fidelity
                    entanglement_bonus = 0.5 * len(self.state.entangled_nodes)
                    
                    energy_gain = base_gain * coherence_factor + entanglement_bonus
                    
                    # Apply energy gain
                    self.state.energy += energy_gain
                    
                    # Check for replication
                    if self.state.energy >= self.config.replication_threshold:
                        # Start replication in background
                        asyncio.create_task(self._replicate_node())
                
                # Report energy if in client mode
                if not self.server_mode and self.client_connection and self.connected:
                    # Send energy update
                    energy_msg = NetworkMessage(
                        message_type="energy_update",
                        sender_id=self.node_id,
                        content={
                            "energy": self.state.energy,
                            "timestamp": time.time()
                        }
                    )
                    
                    try:
                        await self.client_connection.send(json.dumps(energy_msg.to_dict()))
                    except:
                        # Connection error, will attempt reconnect
                        self.connected = False
                
                # Wait before next processing
                await asyncio.sleep(5.0)
                
            except Exception as e:
                self.logger.error(f"Error in energy processing loop: {e}")
                await asyncio.sleep(5.0)
    
    async def _discovery_loop(self) -> None:
        """Network discovery background task"""
        self.logger.info("Starting network discovery loop")
        
        discovery_attempts = 0
        
        # Initial delay to allow node to stabilize
        await asyncio.sleep(10.0)
        
        while not self.shutdown_event.is_set() and discovery_attempts < self.config.max_discovery_attempts:
            try:
                # Skip if client mode and not connected
                if not self.server_mode and not self.connected:
                    await asyncio.sleep(10.0)
                    continue
                
                # Only server mode or well-connected client nodes should discover
                if self.server_mode or len(self.state.connected_nodes) > 2:
                    # Discover nodes on network
                    discovered = await self.network_discovery.discover_nodes()
                    
                    if discovered:
                        self.logger.info(f"Discovered {len(discovered)} nodes")
                        
                        # Add to discovered nodes
                        for node in discovered:
                            node_id = node["node_id"]
                            
                            # Skip self
                            if node_id == self.node_id:
                                continue
                                
                            # Store discovered node
                            self.discovered_nodes[node_id] = node
                            
                            # Try to connect
                            if self.server_mode:
                                # Connect in background
                                asyncio.create_task(self._connect_to_node(node))
                
                discovery_attempts += 1
                
                # Longer wait between discovery attempts
                await asyncio.sleep(30.0)
                
            except Exception as e:
                self.logger.error(f"Error in discovery loop: {e}")
                await asyncio.sleep(30.0)
    
    async def _connect_to_node(self, node_info: Dict[str, Any]) -> None:
        """Connect to a discovered node"""
        node_id = node_info["node_id"]
        ip_address = node_info["ip_address"]
        port = node_info.get("port", DEFAULT_PORT)
        
        # Skip if already connected
        if node_id in self.connections or node_id in self.state.connected_nodes:
            return
        
        try:
            # Connect to node
            ws_url = f"ws://{ip_address}:{port}/qsin"
            websocket = await websockets.connect(ws_url)
            
            # Register with node
            register_msg = NetworkMessage(
                message_type="register",
                sender_id=self.node_id,
                receiver_id=node_id,
                content={
                    "node_name": self.node_name,
                    "energy": self.state.energy,
                    "capabilities": ["compute", "storage", "entanglement"]
                }
            )
            
            await websocket.send(json.dumps(register_msg.to_dict()))
            
            # Wait for registration acknowledgment
            response = await websocket.recv()
            data = json.loads(response)
            
            if data.get("message_type") == "register_ack":
                # Store connection
                self.connections[node_id] = websocket
                self.state.connected_nodes.add(node_id)
                
                # Add to network graph
                self.network_graph.add_node(
                    node_id,
                    name=node_info.get("node_name", f"Node-{node_id[:8]}"),
                    type="peer",
                    energy=0.0,
                    last_seen=time.time()
                )
                
                # Start message processing
                asyncio.create_task(self._process_node_messages(node_id, websocket))
                
                self.logger.info(f"Connected to node {node_id}")
            else:
                # Registration failed
                await websocket.close()
                self.logger.warning(f"Failed to register with node {node_id}")
                
        except Exception as e:
            self.logger.error(f"Error connecting to node {node_id}: {e}")
    
    async def _replicate_node(self) -> None:
        """Replicate node to another host"""
        self.logger.info("Attempting node replication")
        
        with self.lock:
            # Check energy again to avoid race conditions
            if self.state.energy < self.config.replication_threshold:
                self.logger.warning("Insufficient energy for replication")
                return
            
            # Use half energy for replication
            replication_energy = self.state.energy / 2
            self.state.energy -= replication_energy
        
        try:
            # Find suitable hosts for replication
            if self.server_mode:
                # Use network discovery to find hosts
                subnet = self.network_discovery._get_local_subnet()
                live_hosts = self.network_discovery.scan_subnet(subnet)
                
                # Filter out hosts that already have nodes
                candidate_hosts = []
                for host in live_hosts:
                    # Skip localhost
                    if host == "127.0.0.1" or host == "localhost":
                        continue
                    
                    # Skip hosts with known nodes
                    known_host = False
                    for node_info in self.discovered_nodes.values():
                        if node_info.get("ip_address") == host:
                            known_host = True
                            break
                    
                    if not known_host:
                        candidate_hosts.append(host)
                
                # Try to replicate to a random host
                if candidate_hosts:
                    target_host = random.choice(candidate_hosts)
                    
                    # Deploy node to target host
                    success = await self.ssh_deployer.deploy_node(target_host)
                    
                    if success:
                        self.logger.info(f"Successfully replicated to {target_host}")
                        self.state.successful_replications += 1
                    else:
                        self.logger.warning(f"Failed to replicate to {target_host}")
                        
                        # Restore some energy
                        with self.lock:
                            self.state.energy += replication_energy * 0.7
                
                else:
                    self.logger.warning("No suitable hosts found for replication")
                    
                    # Restore some energy
                    with self.lock:
                        self.state.energy += replication_energy * 0.7
            
            else:
                # Client mode - ask server to replicate
                if self.client_connection and self.connected:
                    # Send replication request
                    replication_msg = NetworkMessage(
                        message_type="replication_request",
                        sender_id=self.node_id,
                        content={
                            "energy": replication_energy,
                            "timestamp": time.time()
                        }
                    )
                    
                    await self.client_connection.send(json.dumps(replication_msg.to_dict()))
                    
                    # Count as successful (server will handle actual replication)
                    self.state.successful_replications += 1
                    
                else:
                    self.logger.warning("Not connected to server for replication")
                    
                    # Restore some energy
                    with self.lock:
                        self.state.energy += replication_energy * 0.7
        
        except Exception as e:
            self.logger.error(f"Error in replication: {e}")
            
            # Restore some energy on error
            with self.lock:
                self.state.energy += replication_energy * 0.5
    
    async def _replicate_to_host(self, target_host: str) -> None:
        """Replicate node to a specific host"""
        self.logger.info(f"Replicating to host {target_host}")
        
        try:
            # Deploy node to target host
            success = await self.ssh_deployer.deploy_node(target_host)
            
            if success:
                self.logger.info(f"Successfully replicated to {target_host}")
                self.state.successful_replications += 1
            else:
                self.logger.warning(f"Failed to replicate to {target_host}")
                
        except Exception as e:
            self.logger.error(f"Error replicating to host {target_host}: {e}")
    
    async def _server_maintenance_loop(self) -> None:
        """Server maintenance background task"""
        self.logger.info("Starting server maintenance loop")
        
        while not self.shutdown_event.is_set():
            try:
                # Check for stale nodes
                now = time.time()
                stale_nodes = []
                
                for node_id, attrs in self.network_graph.nodes(data=True):
                    if node_id == self.node_id:
                        continue
                        
                    last_seen = attrs.get("last_seen", 0)
                    if now - last_seen > 60:  # 60 seconds timeout
                        stale_nodes.append(node_id)
                
                # Remove stale nodes
                for node_id in stale_nodes:
                    if node_id in self.network_graph:
                        self.network_graph.remove_node(node_id)
                    
                    if node_id in self.connections:
                        await self.connections[node_id].close()
                        del self.connections[node_id]
                    
                    if node_id in self.state.connected_nodes:
                        self.state.connected_nodes.remove(node_id)
                    
                    if node_id in self.state.entangled_nodes:
                        self.state.entangled_nodes.remove(node_id)
                    
                    self.logger.info(f"Removed stale node {node_id}")
                
                # Update total uptime
                self.state.total_uptime = time.time() - self.state.last_active
                
                await asyncio.sleep(30.0)
                
            except Exception as e:
                self.logger.error(f"Error in server maintenance loop: {e}")
                await asyncio.sleep(30.0)
    
    async def _client_message_loop(self) -> None:
        """Client message processing background task"""
        self.logger.info("Starting client message loop")
        
        while not self.shutdown_event.is_set() and self.connected:
            try:
                # Receive message
                message = await self.client_connection.recv()
                data = json.loads(message)
                
                # Convert to NetworkMessage
                if isinstance(data, dict) and "message_type" in data:
                    msg = NetworkMessage.from_dict(data)
                else:
                    # Legacy message format
                    msg_type = data.get("type", "unknown")
                    msg = NetworkMessage(
                        message_type=msg_type,
                        sender_id=data.get("sender_id", "unknown"),
                        content=data
                    )
                
                # Handle message
                response = await self._handle_message(msg)
                
                # Send response if any
                if response:
                    await self.client_connection.send(json.dumps(response.to_dict()))
                
            except websockets.exceptions.ConnectionClosed:
                self.logger.warning("Connection to server closed")
                self.connected = False
                break
            except Exception as e:
                self.logger.error(f"Error in client message loop: {e}")
                self.connected = False
                break
    
    async def _client_reconnect_loop(self) -> None:
        """Client reconnection background task"""
        self.logger.info("Starting client reconnect loop")
        
        while not self.shutdown_event.is_set():
            # Skip if already connected
            if self.connected:
                await asyncio.sleep(5.0)
                continue
            
            try:
                self.logger.info("Attempting to reconnect to server")
                
                # Connect to server
                self.client_connection = await websockets.connect(self.config.server_url)
                
                # Register with server
                register_msg = NetworkMessage(
                    message_type="register",
                    sender_id=self.node_id,
                    content={
                        "node_name": self.node_name,
                        "energy": self.state.energy,
                        "capabilities": ["compute", "storage", "entanglement"]
                    }
                )
                
                await self.client_connection.send(json.dumps(register_msg.to_dict()))
                
                # Wait for registration acknowledgment
                response = await self.client_connection.recv()
                data = json.loads(response)
                
                if data.get("message_type") == "register_ack":
                    self.connected = True
                    self.logger.info(f"Reconnected to server")
                    
                    # Start message processing
                    self.tasks.add(asyncio.create_task(self._client_message_loop()))
                    
                else:
                    self.logger.error(f"Registration failed: {data}")
                    await self.client_connection.close()
                
            except Exception as e:
                self.logger.error(f"Reconnection error: {e}")
                await asyncio.sleep(10.0)  # Wait longer between reconnect attempts
            
            # Wait before next attempt if not connected
            if not self.connected:
                await asyncio.sleep(10.0)

# ============================
# === Main Implementation ====
# ============================

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Quantum Swarm Intelligence Network")
    
    # Node configuration
    parser.add_argument("--name", type=str, help="Node name")
    parser.add_argument("--id", type=str, help="Node ID (UUID)")
    
    # Networking
    parser.add_argument("--server", action="store_true", help="Run in server mode")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--connect", type=str, help="Server URL to connect to")
    
    # Discovery
    parser.add_argument("--no-discovery", action="store_true", help="Disable network discovery")
    parser.add_argument("--scan", type=str, help="Subnet to scan (e.g., 192.168.1.0/24)")
    
    # Replication
    parser.add_argument("--ssh-user", type=str, help="SSH username for replication")
    parser.add_argument("--ssh-pass", type=str, help="SSH password for replication")
    parser.add_argument("--ssh-key", type=str, help="SSH key file for replication")
    
    # Quantum configuration
    parser.add_argument("--dimension", type=int, default=HILBERT_SPACE_DIM, help="Quantum Hilbert space dimension")
    parser.add_argument("--energy", type=float, default=20.0, help="Initial energy level")
    parser.add_argument("--replication-threshold", type=float, default=REPLICATION_ENERGY_THRESHOLD, 
                      help="Energy threshold for replication")
    
    # Persistence
    parser.add_argument("--persistence-path", type=str, default="./qsin_data", help="Path for persistent storage")
    
    # Other
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--log-level", type=str, default="INFO", 
                      choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                      help="Logging level")
    
    return parser.parse_args()

async def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Configure logging
    log_level = getattr(logging, args.log_level)
    logging.basicConfig(level=log_level, 
                     format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    
    # Load config from file if provided
    config = None
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config_data = json.load(f)
            config = QSINConfig.from_dict(config_data)
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
    
    # Create config from arguments if not loaded from file
    if not config:
        config = QSINConfig(
            node_id=args.id or str(uuid.uuid4()),
            node_name=args.name or f"QSIN-Node-{uuid.uuid4().hex[:8]}",
            server_mode=args.server,
            port=args.port,
            host=args.host,
            server_url=args.connect or f"ws://localhost:{args.port}",
            dimension=args.dimension,
            initial_energy=args.energy,
            replication_threshold=args.replication_threshold,
            discovery_enabled=not args.no_discovery,
            persistence_path=args.persistence_path,
            ssh_username=args.ssh_user or "",
            ssh_password=args.ssh_pass or "",
            ssh_key_path=args.ssh_key or ""
        )
    
    # Create node
    node = QSINNode(config)
    
    try:
        # Start node
        await node.start()
        
        # Wait for shutdown signal
        while True:
            await asyncio.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("Shutting down due to keyboard interrupt")
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
    finally:
        # Stop node
        await node.stop()

if __name__ == "__main__":
    asyncio.run(main())