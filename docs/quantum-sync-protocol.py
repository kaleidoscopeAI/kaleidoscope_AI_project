#!/usr/bin/env python3
"""
Quantum-Inspired Neural Network Synchronization System
Synchronization Protocol Implementation for Kaleidoscope AI Integration

This module handles the complex synchronization between the quantum network and
the Kaleidoscope AI ServerFleet Orchestrator, ensuring efficient communication,
fault tolerance, and self-optimization capabilities.
"""

import asyncio
import websockets
import numpy as np
import json
import logging
import time
import uuid
import os
import sys
import signal
import threading
import subprocess
import paramiko
import socket
import tempfile
import hashlib
import networkx as nx
import argparse
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Set, Any, Optional, Union, Callable

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger("quantum-sync")

# Constants for synchronization
SYNC_INTERVAL = 5  # seconds
NODE_TIMEOUT = 60  # seconds
MAX_SYNC_ATTEMPTS = 5
EDGE_WEIGHT_DECAY = 0.95  # Decay factor for edge weights
COHERENCE_THRESHOLD = 0.6  # Minimum coherence for node synchronization
TELEPORTATION_BUFFER_SIZE = 10 * 1024 * 1024  # 10 MB buffer for quantum teleportation

@dataclass
class KaleidoscopeNode:
    """Representation of a Kaleidoscope AI node in the fleet"""
    hostname: str
    ip_address: str
    role: str
    status: str = "uninitialized"
    quantum_node_id: Optional[str] = None
    last_sync: float = 0.0
    capabilities: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        """Initialize capabilities based on role"""
        if self.role == "controller":
            self.capabilities = {"orchestration", "entanglement_routing", "global_sync"}
        elif self.role == "node":
            self.capabilities = {"computation", "local_sync", "teleportation"}
        elif self.role == "memory":
            self.capabilities = {"storage", "quantum_buffer", "state_persistence"}
        elif self.role == "insight":
            self.capabilities = {"visualization", "pattern_detection", "coherence_analysis"}

@dataclass
class QuantumSynchronizationStats:
    """Statistics for quantum synchronization"""
    total_nodes: int = 0
    active_nodes: int = 0
    synchronization_attempts: int = 0
    successful_syncs: int = 0
    failed_syncs: int = 0
    average_coherence: float = 1.0
    entangled_pairs: int = 0
    teleportation_operations: int = 0
    last_global_sync: float = 0.0

class QuantumStateDiffusion:
    """
    Implements quantum state diffusion algorithm for synchronizing
    distributed quantum states across nodes with minimal information loss
    """
    
    def __init__(self, dim: int = 32):
        self.dimension = dim
        self.diffusion_matrix = np.eye(dim, dtype=complex)
        self.coherence_threshold = COHERENCE_THRESHOLD
    
    def initialize_diffusion_matrix(self, network_graph: nx.Graph) -> None:
        """Initialize diffusion matrix based on network topology"""
        # Create matrix based on graph Laplacian
        lap = nx.laplacian_matrix(network_graph).toarray().astype(complex)
        
        # Normalize Laplacian
        if network_graph.number_of_nodes() > 1:
            # Add small complex perturbation for quantum effects
            perturbation = np.random.normal(0, 0.01, lap.shape) + 1j * np.random.normal(0, 0.01, lap.shape)
            lap = lap + perturbation
            
            # Make Hermitian
            lap = 0.5 * (lap + lap.conj().T)
            
            # Compute diffusion matrix: D = exp(-i * L * dt)
            self.diffusion_matrix = np.exp(-0.1j * lap)
        else:
            self.diffusion_matrix = np.eye(self.dimension, dtype=complex)
    
    def diffuse_state(self, state_vector: np.ndarray, neighbor_states: List[np.ndarray],
                      weights: List[float]) -> np.ndarray:
        """
        Diffuse quantum state based on neighbor states and connection weights
        
        Args:
            state_vector: Local quantum state
            neighbor_states: List of neighbor quantum states
            weights: List of connection weights corresponding to neighbors
            
        Returns:
            Updated quantum state after diffusion
        """
        if not neighbor_states:
            return state_vector
        
        # Normalize weights
        total_weight = sum(weights) + 1.0  # Include self-weight of 1.0
        norm_weights = [w / total_weight for w in weights]
        self_weight = 1.0 / total_weight
        
        # Initialize diffused state with weighted self-contribution
        diffused_state = self_weight * state_vector
        
        # Add weighted contributions from neighbors
        for i, (neighbor_state, weight) in enumerate(zip(neighbor_states, norm_weights)):
            # Adjust phase to maintain coherence
            phase_factor = self._calculate_phase_alignment(state_vector, neighbor_state)
            adjusted_neighbor = phase_factor * neighbor_state
            diffused_state += weight * adjusted_neighbor
        
        # Normalize the final state
        diffused_state = diffused_state / np.linalg.norm(diffused_state)
        
        return diffused_state
    
    def _calculate_phase_alignment(self, state1: np.ndarray, state2: np.ndarray) -> complex:
        """Calculate optimal phase factor to align two quantum states"""
        # <ψ1|ψ2> = |<ψ1|ψ2>|e^(iθ)
        # We want to find e^(-iθ) to align the states
        overlap = np.vdot(state1, state2)
        
        if abs(overlap) < 1e-10:
            # States are nearly orthogonal, return identity
            return 1.0
            
        phase = np.angle(overlap)
        return np.exp(-1j * phase)
    
    def calculate_coherence(self, state1: np.ndarray, state2: np.ndarray) -> float:
        """Calculate coherence between two quantum states"""
        fidelity = np.abs(np.vdot(state1, state2))**2
        return fidelity

class QuantumTeleportationProtocol:
    """
    Implements secure data transmission using quantum teleportation principles
    for high-throughput, secure communication between nodes
    """
    
    def __init__(self, buffer_size: int = TELEPORTATION_BUFFER_SIZE):
        self.buffer_size = buffer_size
        self.teleportation_buffer = {}  # buffer_id -> buffer_data
        self.entanglement_pairs = {}  # pair_id -> (node1_id, node2_id, entanglement_strength)
    
    def create_entanglement_resource(self, node1_id: str, node2_id: str, strength: float) -> str:
        """Create an entanglement resource for teleportation"""
        pair_id = str(uuid.uuid4())
        self.entanglement_pairs[pair_id] = (node1_id, node2_id, strength)
        return pair_id
    
    def encode_data(self, data: Any, pair_id: str) -> Tuple[str, Dict[str, Any]]:
        """
        Encode data for quantum teleportation
        
        Args:
            data: Data to teleport
            pair_id: ID of entanglement pair to use
            
        Returns:
            buffer_id and qubits for transmission
        """
        if pair_id not in self.entanglement_pairs:
            raise ValueError(f"Entanglement pair {pair_id} not found")
        
        # Serialize data
        if isinstance(data, (dict, list, tuple)):
            data_bytes = json.dumps(data).encode('utf-8')
        elif isinstance(data, str):
            data_bytes = data.encode('utf-8')
        elif isinstance(data, bytes):
            data_bytes = data
        else:
            data_bytes = str(data).encode('utf-8')
        
        # Create buffer ID
        buffer_id = hashlib.sha256(data_bytes + str(time.time()).encode()).hexdigest()
        
        # Store in buffer
        self.teleportation_buffer[buffer_id] = {
            "data": data_bytes,
            "timestamp": time.time(),
            "pair_id": pair_id,
            "size": len(data_bytes),
            "checksum": hashlib.md5(data_bytes).hexdigest()
        }
        
        # In real quantum teleportation, we would perform a Bell measurement
        # Here we simulate by creating "qubits" that represent measurement outcomes
        _, _, entanglement = self.entanglement_pairs[pair_id]
        
        # Create synthetic measurement outcomes
        # Higher entanglement = more reliable teleportation
        error_rate = max(0, 0.5 - 0.5 * entanglement)
        qubits = []
        
        # Generate "qubits" based on data bytes and entanglement quality
        for i in range(min(8, len(data_bytes))):
            bit_value = data_bytes[i] & 0x01
            # Flip bit with probability based on error rate
            if np.random.random() < error_rate:
                bit_value = 1 - bit_value
            qubits.append(bit_value)
        
        return buffer_id, {
            "qubits": qubits,
            "buffer_id": buffer_id,
            "entanglement": entanglement
        }
    
    def decode_data(self, teleport_data: Dict[str, Any]) -> Tuple[bool, Any]:
        """
        Decode teleported data
        
        Args:
            teleport_data: Teleportation data including buffer_id
            
        Returns:
            Success status and decoded data
        """
        buffer_id = teleport_data.get("buffer_id")
        if not buffer_id or buffer_id not in self.teleportation_buffer:
            return False, None
        
        buffer_data = self.teleportation_buffer[buffer_id]
        data_bytes = buffer_data["data"]
        
        # In real quantum teleportation, receiver would apply corrections
        # Here we simulate successful reception with probability based on entanglement
        pair_id = buffer_data["pair_id"]
        if pair_id in self.entanglement_pairs:
            _, _, entanglement = self.entanglement_pairs[pair_id]
            success_prob = entanglement
        else:
            # Pair no longer exists (maybe consumed)
            success_prob = 0.5  # Random chance
        
        # Determine if teleportation succeeds
        if np.random.random() < success_prob:
            # Teleportation successful
            try:
                # Try to decode as JSON
                data_str = data_bytes.decode('utf-8')
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    # Not JSON, return as string
                    data = data_str
                
                # Clean up buffer
                del self.teleportation_buffer[buffer_id]
                
                return True, data
            except:
                # Return raw bytes if decoding fails
                return True, data_bytes
        else:
            # Teleportation failed
            return False, None
    
    def consume_entanglement(self, pair_id: str) -> bool:
        """Consume entanglement resource after use"""
        if pair_id in self.entanglement_pairs:
            del self.entanglement_pairs[pair_id]
            return True
        return False

class QuantumSynchronizationProtocol:
    """
    Core synchronization protocol that integrates Kaleidoscope AI with
    the quantum-inspired neural network system
    """
    
    def __init__(self, server_url: str, kaleidoscope_config: Dict[str, Any]):
        self.server_url = server_url
        self.kaleidoscope_config = kaleidoscope_config
        
        # Kaleidoscope nodes
        self.nodes = {}  # hostname -> KaleidoscopeNode
        
        # Quantum nodes
        self.quantum_nodes = {}  # node_id -> quantum_node_info
        
        # Network graph
        self.network_graph = nx.Graph()
        
        # Synchronization and teleportation protocols
        self.state_diffusion = QuantumStateDiffusion()
        self.teleportation = QuantumTeleportationProtocol()
        
        # Synchronization statistics
        self.stats = QuantumSynchronizationStats()
        
        # Connection pool
        self.ssh_connections = {}  # hostname -> ssh_client
        
        # WebSocket connection to quantum server
        self.websocket = None
        self.connected = False
        
        # Protocol node identity
        self.node_id = str(uuid.uuid4())
        self.node_name = f"Kaleidoscope-Sync-{self.node_id[:8]}"
        
        # Shutdown signal
        self.shutdown_event = asyncio.Event()
        
        # Background tasks
        self.tasks = set()
    
    def load_kaleidoscope_nodes(self) -> None:
        """Load Kaleidoscope nodes from configuration"""
        if "servers" in self.kaleidoscope_config:
            for server in self.kaleidoscope_config["servers"]:
                node = KaleidoscopeNode(
                    hostname=server.get("hostname", "unknown"),
                    ip_address=server.get("ip_address", "127.0.0.1"),
                    role=server.get("roles", ["node"])[0] if isinstance(server.get("roles"), list) else "node",
                    status="configured"
                )
                self.nodes[node.hostname] = node
                self.network_graph.add_node(node.hostname, role=node.role)
            
            # Add edges from neighbors if specified
            for server in self.kaleidoscope_config["servers"]:
                hostname = server.get("hostname", "unknown")
                neighbors = server.get("neighbors", [])
                
                for neighbor in neighbors:
                    if hostname in self.nodes and neighbor in self.nodes:
                        self.network_graph.add_edge(hostname, neighbor, weight=1.0)
        
        logger.info(f"Loaded {len(self.nodes)} Kaleidoscope nodes")
    
    async def connect_to_quantum_server(self) -> bool:
        """Connect to the quantum network server"""
        try:
            logger.info(f"Connecting to quantum server: {self.server_url}")
            self.websocket = await websockets.connect(self.server_url)
            
            # Register with the server
            await self.send_message({
                "type": "register",
                "node_id": self.node_id,
                "name": self.node_name
            })
            
            # Wait for welcome message
            welcome = await self.receive_message()
            if welcome and welcome.get("type") == "welcome":
                self.connected = True
                logger.info(f"Connected to quantum network with {welcome.get('network_size', 0)} nodes")
                return True
            else:
                logger.error(f"Unexpected welcome message: {welcome}")
                return False
                
        except Exception as e:
            logger.error(f"Connection error: {e}")
            self.connected = False
            return False
    
    async def send_message(self, message: Dict[str, Any]) -> bool:
        """Send a message to the quantum server"""
        if not self.connected or not self.websocket:
            return False
        
        try:
            await self.websocket.send(json.dumps(message))
            return True
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            self.connected = False
            return False
    
    async def receive_message(self) -> Optional[Dict[str, Any]]:
        """Receive a message from the quantum server"""
        if not self.connected or not self.websocket:
            return None
        
        try:
            message = await self.websocket.recv()
            data = json.loads(message)
            return data
        except Exception as e:
            logger.error(f"Error receiving message: {e}")
            self.connected = False
            return None
    
    def _create_ssh_client(self, node: KaleidoscopeNode) -> Optional[paramiko.SSHClient]:
        """Create SSH client for a Kaleidoscope node"""
        if node.hostname in self.ssh_connections:
            # Check if connection is still active
            ssh = self.ssh_connections[node.hostname]
            transport = ssh.get_transport()
            if transport and transport.is_active():
                return ssh
        
        try:
            # Create new SSH connection
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # Get SSH key path from config
            ssh_key_path = self.kaleidoscope_config.get("ssh_key_path", None)
            
            # Connect to the node
            ssh.connect(
                hostname=node.ip_address,
                port=22,
                username=self.kaleidoscope_config.get("username", "ubuntu"),
                key_filename=ssh_key_path,
                timeout=10
            )
            
            # Store connection
            self.ssh_connections[node.hostname] = ssh
            return ssh
            
        except Exception as e:
            logger.error(f"SSH connection error for {node.hostname}: {e}")
            return None
    
    def _execute_command(self, ssh_client: paramiko.SSHClient, command: str) -> Tuple[bool, str]:
        """Execute command on remote node"""
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
    
    async def deploy_quantum_client(self, node: KaleidoscopeNode) -> bool:
        """Deploy quantum client on a Kaleidoscope node"""
        logger.info(f"Deploying quantum client on {node.hostname}")
        
        # Create SSH connection
        ssh = self._create_ssh_client(node)
        if not ssh:
            return False
        
        try:
            # Check if Python and pip are installed
            success, _ = self._execute_command(ssh, "which python3")
            if not success:
                logger.error(f"Python3 not found on {node.hostname}")
                return False
            
            # Install dependencies
            install_cmd = (
                "pip3 install --user websockets numpy"
            )
            success, output = self._execute_command(ssh, install_cmd)
            if not success:
                logger.error(f"Failed to install dependencies on {node.hostname}: {output}")
                return False
            
            # Create temporary directory for client
            success, temp_dir = self._execute_command(ssh, "mktemp -d")
            if not success:
                logger.error(f"Failed to create temp directory on {node.hostname}")
                return False
            
            # Generate client configuration
            client_config = {
                "server_url": self.server_url,
                "node_name": f"Kaleidoscope-{node.hostname}",
                "role": node.role,
                "capabilities": list(node.capabilities)
            }
            
            config_json = json.dumps(client_config, indent=2)
            config_cmd = f"echo '{config_json}' > {temp_dir}/client_config.json"
            success, _ = self._execute_command(ssh, config_cmd)
            if not success:
                logger.error(f"Failed to create client config on {node.hostname}")
                return False
            
            # Create SFTP client
            sftp = ssh.open_sftp()
            
            # Get client script path
            client_script_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "quantum_client.py"
            )
            
            # Upload client script
            remote_script_path = f"{temp_dir}/quantum_client.py"
            sftp.put(client_script_path, remote_script_path)
            sftp.chmod(remote_script_path, 0o755)
            sftp.close()
            
            # Start client in background
            start_cmd = (
                f"cd {temp_dir} && "
                f"nohup python3 {remote_script_path} "
                f"--server {self.server_url} "
                f"--config client_config.json "
                f"> quantum_client.log 2>&1 &"
            )
            
            success, output = self._execute_command(ssh, start_cmd)
            if not success:
                logger.error(f"Failed to start quantum client on {node.hostname}: {output}")
                return False
            
            # Get client process ID
            success, pid = self._execute_command(ssh, "pgrep -f quantum_client.py")
            if success and pid:
                logger.info(f"Quantum client started on {node.hostname} with PID {pid}")
                return True
            else:
                logger.warning(f"Quantum client started on {node.hostname} but couldn't get PID")
                return True
                
        except Exception as e:
            logger.error(f"Error deploying quantum client on {node.hostname}: {e}")
            return False
    
    async def verify_client_status(self, node: KaleidoscopeNode) -> bool:
        """Verify quantum client status on a node"""
        # Create SSH connection
        ssh = self._create_ssh_client(node)
        if not ssh:
            return False
        
        try:
            # Check if client process is running
            success, pid = self._execute_command(ssh, "pgrep -f quantum_client.py")
            if success and pid:
                # Get client log
                success, log_tail = self._execute_command(ssh, "find /tmp -name quantum_client.log -exec tail -10 {} \\;")
                if success and "Connected to quantum network" in log_tail:
                    logger.info(f"Quantum client on {node.hostname} is running properly")
                    return True
                else:
                    logger.warning(f"Quantum client on {node.hostname} is running but may have issues")
                    return True
            else:
                logger.warning(f"Quantum client not running on {node.hostname}")
                return False
                
        except Exception as e:
            logger.error(f"Error verifying client status on {node.hostname}: {e}")
            return False
    
    async def deploy_all_clients(self) -> None:
        """Deploy quantum clients on all Kaleidoscope nodes"""
        logger.info("Deploying quantum clients on all nodes")
        
        deployment_tasks = []
        
        for hostname, node in self.nodes.items():
            task = asyncio.create_task(self.deploy_quantum_client(node))
            deployment_tasks.append(task)
        
        # Wait for all deployments to complete
        results = await asyncio.gather(*deployment_tasks, return_exceptions=True)
        
        success_count = sum(1 for r in results if r is True)
        fail_count = len(results) - success_count
        
        logger.info(f"Deployment complete: {success_count} succeeded, {fail_count} failed")
    
    async def synchronization_loop(self) -> None:
        """Main synchronization loop for maintaining network coherence"""
        logger.info("Starting synchronization loop")
        
        while not self.shutdown_event.is_set():
            try:
                # Update network graph based on current node states
                self._update_network_graph()
                
                # Update state diffusion matrix
                self.state_diffusion.initialize_diffusion_matrix(self.network_graph)
                
                # Get current nodes from quantum server
                if self.connected:
                    await self.send_message({"type": "get_network_stats"})
                
                # Synchronize each Kaleidoscope node
                for hostname, node in self.nodes.items():
                    if node.status == "active":
                        # Only sync active nodes
                        sync_needed = time.time() - node.last_sync > SYNC_INTERVAL
                        
                        if sync_needed:
                            success = await self._synchronize_node(node)
                            if success:
                                self.stats.successful_syncs += 1
                                node.last_sync = time.time()
                            else:
                                self.stats.failed_syncs += 1
                
                # Update global stats
                self.stats.total_nodes = len(self.nodes)
                self.stats.active_nodes = sum(1 for n in self.nodes.values() if n.status == "active")
                
                # Wait before next synchronization cycle
                await asyncio.sleep(SYNC_INTERVAL)
                
            except Exception as e:
                logger.error(f"Error in synchronization loop: {e}")
                await asyncio.sleep(SYNC_INTERVAL)
    
    def _update_network_graph(self) -> None:
        """Update network graph with current node states"""
        # Update node attributes
        for hostname, node in self.nodes.items():
            if hostname in self.network_graph:
                self.network_graph.nodes[hostname]['status'] = node.status
                self.network_graph.nodes[hostname]['last_sync'] = node.last_sync
        
        # Update edge weights based on sync times
        current_time = time.time()
        for u, v, data in self.network_graph.edges(data=True):
            # Decay edge weight based on time since last sync
            node_u = self.nodes.get(u)
            node_v = self.nodes.get(v)
            
            if node_u and node_v:
                # Calculate time since last sync for both nodes
                time_u = current_time - node_u.last_sync if node_u.last_sync > 0 else SYNC_INTERVAL
                time_v = current_time - node_v.last_sync if node_v.last_sync > 0 else SYNC_INTERVAL
                
                # Calculate normalized sync time (lower is better)
                norm_time = min(1.0, (time_u + time_v) / (2 * SYNC_INTERVAL * 10))
                
                # Update edge weight
                old_weight = data.get('weight', 1.0)
                new_weight = old_weight * EDGE_WEIGHT_DECAY + (1.0 - norm_time) * (1.0 - EDGE_WEIGHT_DECAY)
                
                # Ensure weight stays within reasonable bounds
                data['weight'] = max(0.1, min(1.0, new_weight))
    
    async def _synchronize_node(self, node: KaleidoscopeNode) -> bool:
        """Synchronize a Kaleidoscope node with the quantum network"""
        logger.info(f"Synchronizing node {node.hostname}")
        self.stats.synchronization_attempts += 1
        
        # Basic implementation: verify client is running
        try:
            return await self.verify_client_status(node)
        except Exception as e:
            logger.error(f"Error synchronizing node {node.hostname}: {e}")
            return False
    
    async def health_check_loop(self) -> None:
        """Loop for checking health of Kaleidoscope nodes"""
        logger.info("Starting health check loop")
        
        while not self.shutdown_event.is_set():
            try:
                active_count = 0
                for hostname, node in self.nodes.items():
                    ssh = self._create_ssh_client(node)
                    if ssh:
                        # Check if node is responsive
                        success, _ = self._execute_command(ssh, "uptime")
                        if success:
                            node.status = "active"
                            active_count += 1
                        else:
                            node.status = "unreachable"
                    else:
                        node.status = "unreachable"
                
                logger.info(f"Health check: {active_count}/{len(self.nodes)} nodes active")
                
                # Wait before next health check
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(30)
    
    async def message_handler(self) -> None:
        """Handle incoming messages from the quantum server"""
        logger.info("Starting message handler")
        
        while not self.shutdown_event.is_set():
            try:
                if not self.connected:
                    # Try to reconnect if not connected
                    await self.connect_to_quantum_server()
                    if not self.connected:
                        await asyncio.sleep(5)
                        continue
                
                # Receive message
                message = await self.receive_message()
                if not message:
                    # Connection lost
                    logger.warning("Connection to quantum server lost, reconnecting...")
                    self.connected = False
                    continue
                
                # Process message based on type
                msg_type = message.get("type", "")
                
                if msg_type == "pong":
                    # Heartbeat response, nothing to do
                    pass
                
                elif msg_type == "entanglement_created":
                    # Notified about entanglement between nodes
                    node1_id = message.get("node1_id")
                    node2_id = message.get("node2_id")
                    entanglement = message.get("entanglement", 0.0)
                    
                    logger.info(f"Entanglement created between {node1_id} and {node2_id}")
                    
                    # Create entanglement resource for teleportation
                    self.teleportation.create_entanglement_resource(node1_id, node2_id, entanglement)
                    self.stats.entangled_pairs += 1
                
                elif msg_type == "global_measurement":
                    # Global quantum measurement, update stats
                    observable = message.get("observable")
                    value = message.get("value", 0.0)
                    fidelity = message.get("fidelity", 0.0)
                    
                    # Update coherence stats
                    self.stats.average_coherence = 0.9 * self.stats.average_coherence + 0.1 * fidelity
                
                elif msg_type == "network_stats":
                    # Update quantum network stats
                    active_nodes = message.get("active_nodes", 0)
                    total_nodes = message.get("total_nodes", 0)
                    entangled_pairs = message.get("entangled_pairs", 0)
                    
                    logger.info(f"Quantum network stats: {active_nodes}/{total_nodes} nodes, {entangled_pairs