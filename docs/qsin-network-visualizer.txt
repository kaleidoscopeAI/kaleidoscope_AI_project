#!/usr/bin/env python3
"""
QSIN Network Visualizer and Control Dashboard

Advanced visualization and management interface for the Quantum Swarm Intelligence Network.
Provides real-time monitoring, control, and analysis of the quantum swarm.

Features:
- Real-time network topology visualization
- Quantum state analysis and visualization
- Energy flow monitoring
- Task distribution and performance metrics
- Node deployment and replication management
"""

import os
import sys
import json
import time
import asyncio
import argparse
import logging
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch
import websockets
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Dict, List, Tuple, Any, Optional, Set
import concurrent.futures
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
import uuid
from io import BytesIO
from PIL import Image, ImageTk
import colorsys
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("qsin-visualizer")

# ==============================
# === Data Structures ===========
# ==============================

@dataclass
class VisualizerConfig:
    """Configuration for the QSIN Visualizer"""
    connect_url: str = "ws://localhost:8765"
    refresh_interval: float = 1.0  # seconds
    max_history: int = 300  # data points to keep in history
    auto_connect: bool = True
    node_display_limit: int = 50  # Maximum nodes to display
    theme: str = "dark"  # 'dark' or 'light'
    show_animations: bool = True
    plot_update_interval: float = 500  # milliseconds
    log_level: str = "INFO"
    persistence_path: str = "./qsin_viz_data"
    
    @classmethod
    def from_file(cls, filepath: str) -> 'VisualizerConfig':
        """Load configuration from file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            return cls(**data)
        except Exception as e:
            logger.warning(f"Error loading config from {filepath}: {e}")
            return cls()
    
    def save_to_file(self, filepath: str) -> None:
        """Save configuration to file"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(asdict(self), f, indent=2)
        except Exception as e:
            logger.error(f"Error saving config to {filepath}: {e}")

@dataclass
class NodeData:
    """Data for a single node in the network"""
    node_id: str
    node_name: str
    status: str = "unknown"
    energy: float = 0.0
    last_seen: float = 0.0
    quantum_state: Optional[Any] = None
    connections: Set[str] = field(default_factory=set)
    entangled_nodes: Set[str] = field(default_factory=set)
    coordinates: Tuple[float, float, float] = field(default_factory=lambda: (0.0, 0.0, 0.0))
    metrics: Dict[str, Any] = field(default_factory=dict)
    history: Dict[str, List[float]] = field(default_factory=lambda: {
        "energy": [],
        "connections": [],
        "entanglements": [],
        "tasks_processed": []
    })

@dataclass
class NetworkData:
    """Data for the entire network"""
    nodes: Dict[str, NodeData] = field(default_factory=dict)
    edges: List[Tuple[str, str, float]] = field(default_factory=list)
    entanglements: List[Tuple[str, str, float]] = field(default_factory=list)
    global_metrics: Dict[str, float] = field(default_factory=dict)
    latest_update: float = 0.0
    history: Dict[str, List[float]] = field(default_factory=lambda: {
        "total_energy": [],
        "node_count": [],
        "edge_count": [],
        "entanglement_count": [],
        "average_coherence": []
    })
    timestamps: List[float] = field(default_factory=list)

# ==============================
# === Network Communication ====
# ==============================

class NetworkClient:
    """Client for communicating with QSIN server"""
    
    def __init__(self, connect_url: str):
        self.connect_url = connect_url
        self.websocket = None
        self.connected = False
        self.client_id = f"visualizer-{uuid.uuid4().hex[:8]}"
        self.connection_lock = threading.Lock()
        self.message_callbacks = []
        self.shutdown_event = asyncio.Event()
    
    def add_message_callback(self, callback) -> None:
        """Add callback for incoming messages"""
        self.message_callbacks.append(callback)
    
    def remove_message_callback(self, callback) -> None:
        """Remove callback for incoming messages"""
        if callback in self.message_callbacks:
            self.message_callbacks.remove(callback)
    
    async def connect(self) -> bool:
        """Connect to QSIN server"""
        with self.connection_lock:
            if self.connected:
                return True
            
            try:
                logger.info(f"Connecting to QSIN server at {self.connect_url}")
                self.websocket = await websockets.connect(self.connect_url)
                
                # Register as visualizer
                register_msg = {
                    "message_type": "register",
                    "sender_id": self.client_id,
                    "content": {
                        "node_name": f"QSIN-Visualizer-{self.client_id[-8:]}",
                        "type": "visualizer",
                        "capabilities": ["monitor", "control", "analyze"]
                    }
                }
                
                await self.websocket.send(json.dumps(register_msg))
                
                # Wait for response
                response = await self.websocket.recv()
                data = json.loads(response)
                
                if data.get("message_type") == "register_ack":
                    self.connected = True
                    logger.info("Connected to QSIN server")
                    
                    # Start message listener in background
                    asyncio.create_task(self._message_listener())
                    
                    return True
                else:
                    logger.error(f"Registration failed: {data}")
                    await self.websocket.close()
                    return False
                
            except Exception as e:
                logger.error(f"Connection error: {e}")
                self.connected = False
                return False
    
    async def disconnect(self) -> None:
        """Disconnect from QSIN server"""
        with self.connection_lock:
            if not self.connected:
                return
            
            try:
                # Send disconnect message
                disconnect_msg = {
                    "message_type": "disconnect",
                    "sender_id": self.client_id,
                    "content": {
                        "reason": "User disconnected"
                    }
                }
                
                await self.websocket.send(json.dumps(disconnect_msg))
                await self.websocket.close()
                
            except Exception as e:
                logger.error(f"Error disconnecting: {e}")
            
            finally:
                self.connected = False
                logger.info("Disconnected from QSIN server")
    
    async def send_message(self, message: Dict[str, Any]) -> bool:
        """Send message to QSIN server"""
        if not self.connected:
            logger.warning("Not connected to server")
            return False
        
        try:
            # Add sender ID if not present
            if "sender_id" not in message:
                message["sender_id"] = self.client_id
            
            await self.websocket.send(json.dumps(message))
            return True
            
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            self.connected = False
            return False
    
    async def request_network_state(self) -> Dict[str, Any]:
        """Request current network state"""
        if not self.connected:
            logger.warning("Not connected to server")
            return {}
        
        try:
            request_msg = {
                "message_type": "network_state_request",
                "sender_id": self.client_id,
                "content": {
                    "include_quantum_states": True,
                    "include_metrics": True
                }
            }
            
            await self.websocket.send(json.dumps(request_msg))
            
            # Wait for response with timeout
            response = await asyncio.wait_for(
                self.websocket.recv(),
                timeout=5.0
            )
            
            data = json.loads(response)
            
            if data.get("message_type") == "network_state_response":
                return data.get("content", {})
            else:
                logger.warning(f"Unexpected response: {data}")
                return {}
            
        except asyncio.TimeoutError:
            logger.error("Timeout waiting for network state")
            return {}
            
        except Exception as e:
            logger.error(f"Error requesting network state: {e}")
            self.connected = False
            return {}
    
    async def _message_listener(self) -> None:
        """Background task for listening to incoming messages"""
        logger.info("Starting message listener")
        
        while self.connected and not self.shutdown_event.is_set():
            try:
                message = await self.websocket.recv()
                data = json.loads(message)
                
                # Process message
                for callback in self.message_callbacks:
                    try:
                        callback(data)
                    except Exception as e:
                        logger.error(f"Error in message callback: {e}")
                
            except websockets.exceptions.ConnectionClosed:
                logger.warning("Connection closed by server")
                self.connected = False
                break
                
            except Exception as e:
                logger.error(f"Error in message listener: {e}")
                # Don't break loop on recoverable errors
        
        logger.info("Message listener stopped")
    
    def shutdown(self) -> None:
        """Signal shutdown"""
        self.shutdown_event.set()

# ==============================
# === Data Manager =============
# ==============================

class NetworkDataManager:
    """Manager for QSIN network data"""
    
    def __init__(self, max_history: int = 300):
        self.network_data = NetworkData()
        self.max_history = max_history
        self.data_lock = threading.RLock()
        self.data_updated_callbacks = []
    
    def add_data_updated_callback(self, callback) -> None:
        """Add callback for data updates"""
        self.data_updated_callbacks.append(callback)
    
    def remove_data_updated_callback(self, callback) -> None:
        """Remove callback for data updates"""
        if callback in self.data_updated_callbacks:
            self.data_updated_callbacks.remove(callback)
    
    def update_from_network_state(self, state_data: Dict[str, Any]) -> None:
        """Update network data from network state response"""
        with self.data_lock:
            timestamp = time.time()
            self.network_data.latest_update = timestamp
            
            # Extract nodes
            nodes_data = state_data.get("nodes", {})
            
            # Track existing nodes for removal of old ones
            current_nodes = set()
            
            for node_id, node_info in nodes_data.items():
                current_nodes.add(node_id)
                
                # Add or update node
                if node_id not in self.network_data.nodes:
                    self.network_data.nodes[node_id] = NodeData(
                        node_id=node_id,
                        node_name=node_info.get("name", f"Node-{node_id[:8]}")
                    )
                
                node = self.network_data.nodes[node_id]
                
                # Update node data
                node.status = node_info.get("status", node.status)
                node.energy = node_info.get("energy", node.energy)
                node.last_seen = node_info.get("last_seen", timestamp)
                
                # Update quantum state if present
                if "quantum_state" in node_info:
                    node.quantum_state = node_info["quantum_state"]
                
                # Update connections
                if "connections" in node_info:
                    node.connections = set(node_info["connections"])
                
                # Update entanglements
                if "entangled_nodes" in node_info:
                    node.entangled_nodes = set(node_info["entangled_nodes"])
                
                # Update metrics
                if "metrics" in node_info:
                    node.metrics.update(node_info["metrics"])
                
                # Update history
                node.history["energy"].append(node.energy)
                node.history["connections"].append(len(node.connections))
                node.history["entanglements"].append(len(node.entangled_nodes))
                node.history["tasks_processed"].append(
                    node.metrics.get("processed_tasks", 0)
                )
                
                # Trim history if too long
                for key in node.history:
                    if len(node.history[key]) > self.max_history:
                        node.history[key] = node.history[key][-self.max_history:]
            
            # Remove old nodes
            for node_id in list(self.network_data.nodes.keys()):
                if node_id not in current_nodes:
                    del self.network_data.nodes[node_id]
            
            # Extract edges
            self.network_data.edges = []
            edges_data = state_data.get("edges", [])
            
            for edge in edges_data:
                if len(edge) >= 3:
                    self.network_data.edges.append((edge[0], edge[1], edge[2]))
            
            # Extract entanglements
            self.network_data.entanglements = []
            entanglements_data = state_data.get("entanglements", [])
            
            for entanglement in entanglements_data:
                if len(entanglement) >= 3:
                    self.network_data.entanglements.append(
                        (entanglement[0], entanglement[1], entanglement[2])
                    )
            
            # Extract global metrics
            if "global_metrics" in state_data:
                self.network_data.global_metrics.update(state_data["global_metrics"])
            
            # Update network history
            self.network_data.timestamps.append(timestamp)
            
            total_energy = sum(node.energy for node in self.network_data.nodes.values())
            self.network_data.history["total_energy"].append(total_energy)
            self.network_data.history["node_count"].append(len(self.network_data.nodes))
            self.network_data.history["edge_count"].append(len(self.network_data.edges))
            self.network_data.history["entanglement_count"].append(len(self.network_data.entanglements))
            
            average_coherence = self.network_data.global_metrics.get("average_coherence", 0.0)
            self.network_data.history["average_coherence"].append(average_coherence)
            
            # Trim history if too long
            if len(self.network_data.timestamps) > self.max_history:
                self.network_data.timestamps = self.network_data.timestamps[-self.max_history:]
                
                for key in self.network_data.history:
                    if len(self.network_data.history[key]) > self.max_history:
                        self.network_data.history[key] = self.network_data.history[key][-self.max_history:]
            
            # Calculate node coordinates using spring layout
            self._calculate_node_positions()
            
            # Notify callbacks
            for callback in self.data_updated_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Error in data updated callback: {e}")
    
    def _calculate_node_positions(self) -> None:
        """Calculate node positions using networkx spring layout"""
        # Create graph from nodes and edges
        G = nx.Graph()
        
        # Add nodes
        for node_id, node in self.network_data.nodes.items():
            G.add_node(node_id)
        
        # Add edges
        for src, dst, weight in self.network_data.edges:
            G.add_edge(src, dst, weight=weight)
        
        # Also add entanglements as edges
        for src, dst, strength in self.network_data.entanglements:
            if G.has_edge(src, dst):
                # Already has edge, update weight
                current_weight = G[src][dst]['weight']
                # Combine weights
                G[src][dst]['weight'] = max(current_weight, strength)
            else:
                # Add new edge
                G.add_edge(src, dst, weight=strength)
        
        # If graph is empty, return
        if not G.nodes():
            return
        
        # Calculate positions with spring layout
        # Use previous positions as starting point if available
        pos_2d = nx.spring_layout(
            G,
            k=0.15,  # Optimal distance between nodes
            iterations=50,  # Number of iterations
            weight='weight'  # Use edge weights
        )
        
        # Add Z coordinate based on node energy
        pos_3d = {}
        for node_id, (x, y) in pos_2d.items():
            if node_id in self.network_data.nodes:
                # Z coordinate based on energy
                energy = self.network_data.nodes[node_id].energy
                # Normalize to range [0, 1]
                z = min(energy / 200.0, 1.0)  # Assume max energy is 200
                pos_3d[node_id] = (x, y, z)
        
        # Update node coordinates
        for node_id, (x, y, z) in pos_3d.items():
            if node_id in self.network_data.nodes:
                self.network_data.nodes[node_id].coordinates = (x, y, z)
    
    def get_network_data(self) -> NetworkData:
        """Get current network data"""
        with self.data_lock:
            # Return a copy to avoid threading issues
            return self.network_data
    
    def get_node_data(self, node_id: str) -> Optional[NodeData]:
        """Get data for a specific node"""
        with self.data_lock:
            return self.network_data.nodes.get(node_id)
    
    def get_energy_history(self) -> Tuple[List[float], List[float]]:
        """Get energy history for plotting"""
        with self.data_lock:
            timestamps = self.network_data.timestamps
            energy = self.network_data.history["total_energy"]
            return timestamps, energy
    
    def get_node_count_history(self) -> Tuple[List[float], List[float]]:
        """Get node count history for plotting"""
        with self.data_lock:
            timestamps = self.network_data.timestamps
            node_count = self.network_data.history["node_count"]
            return timestamps, node_count
    
    def get_entanglement_history(self) -> Tuple[List[float], List[float]]:
        """Get entanglement count history for plotting"""
        with self.data_lock:
            timestamps = self.network_data.timestamps
            entanglement_count = self.network_data.history["entanglement_count"]
            return timestamps, entanglement_count
    
    def get_node_energy_history(self, node_id: str) -> List[float]:
        """Get energy history for a specific node"""
        with self.data_lock:
            node = self.network_data.nodes.get(node_id)
            if node:
                return node.history["energy"]
            return []
    
    def get_quantum_state(self, node_id: str) -> Optional[Any]:
        """Get quantum state for a specific node"""
        with self.data_lock:
            node = self.network_data.nodes.get(node_id)
            if node:
                return node.quantum_state
            return None
    
    def clear_data(self) -> None:
        """Clear all network data"""
        with self.data_lock:
            self.network_data = NetworkData()

# ==============================
# === Task Manager =============
# ==============================

class TaskManager:
    """Manager for QSIN tasks"""
    
    def __init__(self, network_client: NetworkClient):
        self.network_client = network_client
        self.pending_tasks = {}  # task_id -> task_info
        self.completed_tasks = {}  # task_id -> result
        self.task_lock = threading.RLock()
        self.task_result_callbacks = {}  # task_id -> callback
    
    async def create_task(self, task_type: str, target_node: Optional[str] = None, 
                        parameters: Optional[Dict[str, Any]] = None) -> str:
        """Create a new task"""
        task_id = f"task_{uuid.uuid4().hex}"
        
        task_msg = {
            "message_type": "task_request",
            "content": {
                "task_id": task_id,
                "task_type": task_type,
                "parameters": parameters or {}
            }
        }
        
        # Add target node if specified
        if target_node:
            task_msg["receiver_id"] = target_node
        
        # Send task request
        success = await self.network_client.send_message(task_msg)
        
        if success:
            with self.task_lock:
                self.pending_tasks[task_id] = {
                    "task_id": task_id,
                    "task_type": task_type,
                    "target_node": target_node,
                    "parameters": parameters or {},
                    "timestamp": time.time(),
                    "status": "pending"
                }
        
        return task_id
    
    def handle_task_result(self, message: Dict[str, Any]) -> None:
        """Handle task result message"""
        content = message.get("content", {})
        task_id = content.get("task_id")
        
        if not task_id:
            return
        
        with self.task_lock:
            # Update task status
            if task_id in self.pending_tasks:
                task_info = self.pending_tasks.pop(task_id)
                self.completed_tasks[task_id] = {
                    **task_info,
                    "result": content,
                    "completion_time": time.time(),
                    "status": "completed"
                }
                
                # Call result callback if registered
                if task_id in self.task_result_callbacks:
                    callback = self.task_result_callbacks.pop(task_id)
                    callback(content)
    
    def register_result_callback(self, task_id: str, callback) -> None:
        """Register callback for task result"""
        with self.task_lock:
            # Check if task already completed
            if task_id in self.completed_tasks:
                # Call callback immediately
                callback(self.completed_tasks[task_id]["result"])
            else:
                # Register callback for future result
                self.task_result_callbacks[task_id] = callback
    
    def get_pending_tasks(self) -> List[Dict[str, Any]]:
        """Get list of pending tasks"""
        with self.task_lock:
            return list(self.pending_tasks.values())
    
    def get_completed_tasks(self) -> List[Dict[str, Any]]:
        """Get list of completed tasks"""
        with self.task_lock:
            return list(self.completed_tasks.values())
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a task"""
        with self.task_lock:
            if task_id in self.pending_tasks:
                return self.pending_tasks[task_id]
            elif task_id in self.completed_tasks:
                return self.completed_tasks[task_id]
            else:
                return None
    
    def clear_completed_tasks(self) -> None:
        """Clear completed tasks"""
        with self.task_lock:
            self.completed_tasks = {}

# ==============================
# === Network Visualizer =======
# ==============================

class NetworkVisualizer:
    """Visualizer for QSIN network"""
    
    def __init__(self, data_manager: NetworkDataManager, root: tk.Tk, config: VisualizerConfig):
        self.data_manager = data_manager
        self.root = root
        self.config = config
        
        # Set up matplotlib figure
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Set up canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.draw()
        
        # Set up toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, root)
        self.toolbar.update()
        
        # Create node scatter plot
        self.scatter = None
        self.edge_lines = None
        self.entanglement_lines = None
        self.node_labels = []
        
        # Color maps
        self.node_cmap = plt.cm.viridis
        self.edge_cmap = plt.cm.Blues
        self.entanglement_cmap = plt.cm.Reds
        
        # Set up animation
        if self.config.show_animations:
            self.anim = animation.FuncAnimation(
                self.fig, self._update_plot, interval=config.plot_update_interval
            )
        
        # Register for data updates
        self.data_manager.add_data_updated_callback(self._refresh_plot)
    
    def _refresh_plot(self) -> None:
        """Signal that plot should be refreshed"""
        if not self.config.show_animations:
            self._update_plot(None)
            self.canvas.draw()
    
    def _update_plot(self, frame) -> None:
        """Update network plot"""
        network_data = self.data_manager.get_network_data()
        
        # Clear previous plot
        self.ax.clear()
        self.node_labels = []
        
        # Extract node positions and states
        node_xs = []
        node_ys = []
        node_zs = []
        node_colors = []
        node_sizes = []
        node_ids = []
        
        for node_id, node in network_data.nodes.items():
            x, y, z = node.coordinates
            node_xs.append(x)
            node_ys.append(y)
            node_zs.append(z)
            
            # Color based on energy
            color_val = min(node.energy / 100.0, 1.0)  # Normalize to [0, 1]
            node_colors.append(color_val)
            
            # Size based on connections
            size = 50 + 10 * len(node.connections)
            node_sizes.append(size)
            
            node_ids.append(node_id)
        
        # Draw nodes
        if node_xs:
            self.scatter = self.ax.scatter(
                node_xs, node_ys, node_zs,
                c=node_colors,
                s=node_sizes,
                cmap=self.node_cmap,
                alpha=0.8,
                edgecolors='w'
            )
        
        # Draw edges
        if network_data.edges:
            for src, dst, weight in network_data.edges:
                if src in network_data.nodes and dst in network_data.nodes:
                    src_pos = network_data.nodes[src].coordinates
                    dst_pos = network_data.nodes[dst].coordinates
                    
                    # Draw line
                    self.ax.plot(
                        [src_pos[0], dst_pos[0]],
                        [src_pos[1], dst_pos[1]],
                        [src_pos[2], dst_pos[2]],
                        color=self.edge_cmap(weight),
                        alpha=weight,
                        linewidth=1 + weight * 2
                    )
        
        # Draw entanglements
        if network_data.entanglements:
            for src, dst, strength in network_data.entanglements:
                if src in network_data.nodes and dst in network_data.nodes:
                    src_pos = network_data.nodes[src].coordinates
                    dst_pos = network_data.nodes[dst].coordinates
                    
                    # Draw curved line for entanglement
                    # Calculate midpoint and offset it to create curve
                    mid_x = (src_pos[0] + dst_pos[0]) / 2
                    mid_y = (src_pos[1] + dst_pos[1]) / 2
                    mid_z = (src_pos[2] + dst_pos[2]) / 2 + 0.2  # Offset in z direction
                    
                    # Create Bezier curve points
                    t = np.linspace(0, 1, 20)
                    points = []
                    for ti in t:
                        # Quadratic Bezier curve
                        x = (1-ti)**2 * src_pos[0] + 2*(1-ti)*ti * mid_x + ti**2 * dst_pos[0]
                        y = (1-ti)**2 * src_pos[1] + 2*(1-ti)*ti * mid_y + ti**2 * dst_pos[1]
                        z = (1-ti)**2 * src_pos[2] + 2*(1-ti)*ti * mid_z + ti**2 * dst_pos[2]
                        points.append((x, y, z))
                    
                    # Convert to arrays
                    curve_x, curve_y, curve_z = zip(*points)
                    
                    # Draw entanglement curve
                    self.ax.plot(
                        curve_x, curve_y, curve_z,
                        color=self.entanglement_cmap(strength),
                        alpha=min(strength + 0.3, 1.0),
                        linewidth=1 + strength * 3,
                        linestyle='--'
                    )
        
        # Set labels and title
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Energy')
        self.ax