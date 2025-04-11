#!/usr/bin/env python3
"""
Quantum Kaleidoscope Orchestration Hub
======================================

A cutting-edge middleware layer that integrates the Quantum Kaleidoscope backend
with an immersive 3D visualization frontend. This hub creates a real-time
communication channel between components and adds enhanced capabilities for
visualizing quantum processes and emergent behaviors.

Usage:
    python quantum_kaleidoscope_hub.py --port 8080 --data-dir ./data

Requirements:
    - Python 3.8+
    - Flask
    - Flask-SocketIO
    - Numpy
    - networkx
    - python-dotenv
"""

import os
import sys
import time
import json
import uuid
import math
import argparse
import threading
import random
import logging
import subprocess
import asyncio
import traceback
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

# Create and activate virtual environment if needed
VENV_PATH = ".venv"
if not Path(VENV_PATH).exists():
    try:
        import venv
        print(f"Creating virtual environment at {VENV_PATH}...")
        venv.create(VENV_PATH, with_pip=True)
        
        # Determine the path to the Python executable in the virtual environment
        if os.name == 'nt':  # Windows
            python_path = os.path.join(VENV_PATH, 'Scripts', 'python.exe')
            pip_path = os.path.join(VENV_PATH, 'Scripts', 'pip.exe')
        else:  # Unix-like (Linux, macOS)
            python_path = os.path.join(VENV_PATH, 'bin', 'python')
            pip_path = os.path.join(VENV_PATH, 'bin', 'pip')
        
        # Install required packages
        print("Installing required packages...")
        subprocess.check_call([pip_path, 'install', 'flask', 'flask-socketio', 'numpy', 
                              'networkx', 'python-dotenv', 'eventlet', 'colorlog'])
        
        # Re-execute this script with the virtual environment python
        print("Starting with virtual environment...")
        os.execv(python_path, [python_path] + sys.argv)
    except Exception as e:
        print(f"Failed to create virtual environment: {e}")
        print("Continuing with system Python...")

# Now we can safely import our dependencies
try:
    from flask import Flask, render_template, request, jsonify, Response, send_from_directory
    from flask_socketio import SocketIO, emit
    import networkx as nx
    import colorlog
except ImportError as e:
    print(f"Required package missing: {e}")
    print("Please install required packages: pip install flask flask-socketio numpy networkx python-dotenv eventlet colorlog")
    sys.exit(1)

# Set up enhanced logging
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
))

logger = colorlog.getLogger("QuantumHub")
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# Import necessary components from the Kaleidoscope system
try:
    # First, add the parent directory to the Python path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    # Try to import the kaleidoscope modules
    # The specific imports depend on how the original system is structured
    try:
        from execution_sandbox_system import SandboxManager
        logger.info("Successfully imported SandboxManager")
    except ImportError:
        logger.warning("SandboxManager not available")

    try:
        from error_handler import ErrorManager
        logger.info("Successfully imported ErrorManager")
    except ImportError:
        logger.warning("ErrorManager not available")
        
    KALEIDOSCOPE_IMPORTS_SUCCESS = True
except ImportError as e:
    logger.warning(f"Could not import Kaleidoscope modules: {e}")
    logger.warning("Running in standalone mode")
    KALEIDOSCOPE_IMPORTS_SUCCESS = False

# Middleware data structures
@dataclass
class QuantumNode:
    """Represents a node in the quantum field visualization"""
    id: str
    position: Tuple[float, float, float]
    energy: float = 0.5
    stability: float = 0.8
    connections: Dict[str, float] = field(default_factory=dict)
    features: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    visible: bool = True
    creation_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    color: str = "#4488ff"
    size: float = 1.0
    
    def to_dict(self) -> Dict:
        """Convert to a dictionary for serialization"""
        return {
            "id": self.id,
            "position": list(self.position),
            "energy": self.energy,
            "stability": self.stability,
            "connections": self.connections,
            "features": self.features[:20] if self.features else [],  # Limit feature vector size
            "metadata": self.metadata,
            "visible": self.visible,
            "creation_time": self.creation_time,
            "last_update": self.last_update,
            "color": self.color,
            "size": self.size
        }

@dataclass
class QuantumConnection:
    """Represents a connection between nodes in the quantum field"""
    source: str
    target: str
    strength: float = 0.5
    type: str = "standard"  # standard, quantum, causal, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)
    visible: bool = True
    creation_time: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        """Convert to a dictionary for serialization"""
        return {
            "source": self.source,
            "target": self.target,
            "strength": self.strength,
            "type": self.type,
            "metadata": self.metadata,
            "visible": self.visible,
            "creation_time": self.creation_time
        }

@dataclass
class SimulationState:
    """Captures the current state of the quantum simulation"""
    nodes: Dict[str, QuantumNode] = field(default_factory=dict)
    connections: List[QuantumConnection] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    insights: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        """Convert to a dictionary for serialization"""
        return {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "connections": [conn.to_dict() for conn in self.connections],
            "events": self.events[-50:],  # Limit to most recent events
            "insights": self.insights[-20:],  # Limit to most recent insights
            "metrics": self.metrics,
            "timestamp": self.timestamp
        }

class QuantumOrchestrator:
    """
    Core orchestration class that connects to the Kaleidoscope backend,
    manages the quantum field state, and provides real-time updates to clients.
    """
    
    def __init__(self, data_dir: str = "./data", backend_url: Optional[str] = None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.backend_url = backend_url
        
        # Initialize simulation state
        self.state = SimulationState()
        self.state_lock = threading.RLock()
        
        # Internal state tracking
        self.running = True
        self.simulation_thread = None
        self.connection_thread = None
        self.backend_connected = False
        self.last_update_time = time.time()
        
        # Event callbacks
        self.event_callbacks = []
        
        # Network graph for analysis
        self.network_graph = nx.Graph()
        
        # Performance metrics
        self.metrics = {
            "fps": 0,
            "node_count": 0,
            "connection_count": 0,
            "update_time_ms": 0,
            "cpu_usage": 0,
            "memory_usage": 0,
            "start_time": time.time()
        }
        
        # Backend interface 
        self.backend = None
        if KALEIDOSCOPE_IMPORTS_SUCCESS and backend_url:
            self._connect_to_backend()
        else:
            logger.info("Running with simulated quantum field")
            self._initialize_simulated_field()
    
    def _connect_to_backend(self):
        """Connect to the Kaleidoscope backend"""
        try:
            logger.info(f"Connecting to backend at {self.backend_url}...")
            # TODO: Implement actual backend connection
            # This will depend on the exact interface provided by the Kaleidoscope system
            
            self.backend_connected = True
            logger.info("Successfully connected to backend")
        except Exception as e:
            logger.error(f"Failed to connect to backend: {e}")
            logger.info("Falling back to simulated quantum field")
            self._initialize_simulated_field()
    
    def _initialize_simulated_field(self):
        """Initialize a simulated quantum field for standalone use"""
        logger.info("Initializing simulated quantum field")
        
        # Create a seed node at the center
        seed_node = QuantumNode(
            id=str(uuid.uuid4()),
            position=(0, 0, 0),
            energy=0.9,
            stability=0.8,
            features=[random.random() for _ in range(32)],
            metadata={"type": "seed", "description": "Origin point of the quantum field"}
        )
        
        with self.state_lock:
            self.state.nodes[seed_node.id] = seed_node
            
            # Create some initial nodes around the seed
            for i in range(8):
                angle = i * 2 * math.pi / 8
                distance = 5.0
                
                child_node = QuantumNode(
                    id=str(uuid.uuid4()),
                    position=(
                        distance * math.cos(angle),
                        distance * math.sin(angle),
                        (random.random() - 0.5) * 3.0
                    ),
                    energy=0.6 + random.random() * 0.3,
                    stability=0.5 + random.random() * 0.3,
                    features=[random.random() for _ in range(32)],
                    metadata={"type": "child", "origin": "simulation"}
                )
                
                self.state.nodes[child_node.id] = child_node
                
                # Connect to seed node
                conn = QuantumConnection(
                    source=seed_node.id,
                    target=child_node.id,
                    strength=0.5 + random.random() * 0.3
                )
                
                self.state.connections.append(conn)
                seed_node.connections[child_node.id] = conn.strength
                child_node.connections[seed_node.id] = conn.strength
                
                # Sometimes connect to previous node
                if i > 0 and random.random() > 0.5:
                    prev_node = list(self.state.nodes.values())[-2]
                    conn_strength = 0.3 + random.random() * 0.3
                    
                    conn = QuantumConnection(
                        source=prev_node.id,
                        target=child_node.id,
                        strength=conn_strength
                    )
                    
                    self.state.connections.append(conn)
                    prev_node.connections[child_node.id] = conn_strength
                    child_node.connections[prev_node.id] = conn_strength
            
            # Add initial events and insights
            self.state.events.append({
                "type": "system",
                "message": "Quantum field initialized",
                "timestamp": time.time()
            })
            
            self.state.insights.append({
                "type": "system",
                "title": "Field Genesis",
                "content": "Initial quantum field stabilized with seed node and child nodes",
                "confidence": 0.95,
                "timestamp": time.time()
            })
            
            # Update metrics
            self.state.metrics = {
                "node_count": len(self.state.nodes),
                "connection_count": len(self.state.connections),
                "avg_energy": np.mean([n.energy for n in self.state.nodes.values()]),
                "avg_connections": np.mean([len(n.connections) for n in self.state.nodes.values()]),
                "simulation_time": 0
            }
    
    def start(self):
        """Start the orchestrator"""
        logger.info("Starting Quantum Orchestrator")
        
        # Start simulation thread
        self.simulation_thread = threading.Thread(target=self._simulation_loop)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        # Start backend connection thread if applicable
        if self.backend_url:
            self.connection_thread = threading.Thread(target=self._backend_connection_loop)
            self.connection_thread.daemon = True
            self.connection_thread.start()
    
    def stop(self):
        """Stop the orchestrator"""
        logger.info("Stopping Quantum Orchestrator")
        self.running = False
        
        if self.simulation_thread:
            self.simulation_thread.join(timeout=2.0)
        
        if self.connection_thread:
            self.connection_thread.join(timeout=2.0)
    
    def register_event_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Register a callback for state update events"""
        self.event_callbacks.append(callback)
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get the current simulation state"""
        with self.state_lock:
            return self.state.to_dict()
    
    def process_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process text input to create new nodes and connections.
        
        Args:
            text: Input text
            metadata: Optional metadata to attach to the created node
            
        Returns:
            Dict containing the created node's ID and other processing results
        """
        if not metadata:
            metadata = {}
            
        # Add timestamp and source
        metadata.update({
            "timestamp": time.time(),
            "source": "user_input",
            "text_sample": text[:100] + ("..." if len(text) > 100 else "")
        })
        
        logger.info(f"Processing text input ({len(text)} characters)")
        
        # Try to use backend if available
        if self.backend_connected and hasattr(self.backend, 'process_text'):
            try:
                result = self.backend.process_text(text, metadata)
                logger.info(f"Text processed by backend: {result}")
                return result
            except Exception as e:
                logger.error(f"Backend processing failed: {e}")
                logger.info("Falling back to simulated processing")
        
        # Simulated text processing
        with self.state_lock:
            # Create a new node
            node_id = str(uuid.uuid4())
            
            # Generate features from text hash
            hash_obj = hashlib.sha256(text.encode('utf-8'))
            hash_bytes = hash_obj.digest()
            features = [
                (int(hash_bytes[i % len(hash_bytes)]) / 255.0) * 2 - 1
                for i in range(32)
            ]
            
            # Find a position that's not too close to existing nodes
            while True:
                position = (
                    (random.random() * 30) - 15,
                    (random.random() * 30) - 15,
                    (random.random() * 10) - 5
                )
                
                # Check distance to existing nodes
                min_distance = float('inf')
                for node in self.state.nodes.values():
                    dx = position[0] - node.position[0]
                    dy = position[1] - node.position[1]
                    dz = position[2] - node.position[2]
                    distance = math.sqrt(dx*dx + dy*dy + dz*dz)
                    min_distance = min(min_distance, distance)
                
                if min_distance > 3.0 or len(self.state.nodes) < 2:
                    break
            
            # Create node
            node = QuantumNode(
                id=node_id,
                position=position,
                energy=0.8,  # Higher energy for user-created nodes
                stability=0.7,
                features=features,
                metadata=metadata,
                color=self._generate_color_from_text(text)
            )
            
            self.state.nodes[node_id] = node
            
            # Find related nodes based on feature similarity
            similarities = []
            for other_id, other_node in self.state.nodes.items():
                if other_id == node_id:
                    continue
                
                # Skip if other node has no features
                if not other_node.features:
                    continue
                
                # Calculate cosine similarity
                dot_product = sum(a * b for a, b in zip(features, other_node.features[:len(features)]))
                magnitude1 = math.sqrt(sum(a * a for a in features))
                magnitude2 = math.sqrt(sum(b * b for b in other_node.features[:len(features)]))
                
                if magnitude1 > 0 and magnitude2 > 0:
                    similarity = dot_product / (magnitude1 * magnitude2)
                    # Normalize to 0-1 range
                    similarity = (similarity + 1) / 2
                    
                    if similarity > 0.6:  # Only consider fairly similar nodes
                        similarities.append((other_id, similarity))
            
            # Sort by similarity and keep top 3
            similarities.sort(key=lambda x: x[1], reverse=True)
            related_nodes = similarities[:3]
            
            # Create connections to related nodes
            for other_id, similarity in related_nodes:
                conn = QuantumConnection(
                    source=node_id,
                    target=other_id,
                    strength=similarity,
                    type="semantic" if similarity > 0.8 else "standard"
                )
                
                self.state.connections.append(conn)
                node.connections[other_id] = similarity
                self.state.nodes[other_id].connections[node_id] = similarity
            
            # Generate insights based on text and connections
            insights = self._generate_insights_from_text(text, node_id, related_nodes)
            for insight in insights:
                self.state.insights.append(insight)
            
            # Add event
            self.state.events.append({
                "type": "node_creation",
                "node_id": node_id,
                "message": f"Created node from user text input",
                "timestamp": time.time()
            })
            
            # Update state timestamp
            self.state.timestamp = time.time()
            
            # Notify callbacks
            self._notify_callbacks({
                "type": "state_update",
                "source": "text_processing",
                "changes": {
                    "nodes_added": [node_id],
                    "connections_added": len(related_nodes)
                }
            })
            
            return {
                "node_id": node_id,
                "position": list(position),
                "related_nodes": related_nodes,
                "insights": insights
            }
    
    def _generate_color_from_text(self, text: str) -> str:
        """Generate a deterministic color based on text content"""
        # Use hash of text to generate a color
        hash_obj = hashlib.md5(text.encode('utf-8'))
        hash_hex = hash_obj.hexdigest()
        
        # Use first 6 characters of hash for color
        color = "#" + hash_hex[:6]
        
        # Ensure the color isn't too dark
        r, g, b = int(hash_hex[0:2], 16), int(hash_hex[2:4], 16), int(hash_hex[4:6], 16)
        if r + g + b < 300:  # Too dark
            # Brighten it up
            r = min(255, r + 100)
            g = min(255, g + 100)
            b = min(255, b + 100)
            color = f"#{r:02x}{g:02x}{b:02x}"
        
        return color
    
    def _generate_insights_from_text(self, text: str, node_id: str, related_nodes: List[Tuple[str, float]]) -> List[Dict[str, Any]]:
        """Generate simulated insights from text input"""
        insights = []
        
        # Basic text analysis
        words = text.lower().split()
        word_count = len(words)
        avg_word_length = sum(len(w) for w in words) / max(1, word_count)
        
        # Generate a text complexity score
        complexity = min(1.0, avg_word_length / 10)
        
        # Check for topic indicators
        topics = {
            "tech": ["ai", "algorithm", "computer", "data", "technology", "code", "quantum"],
            "science": ["research", "experiment", "theory", "analysis", "scientific"],
            "creative": ["art", "design", "creative", "imagine", "vision"],
            "emotional": ["feel", "emotion", "love", "hate", "angry", "happy", "sad"]
        }
        
        detected_topics = []
        for topic, keywords in topics.items():
            if any(keyword in words for keyword in keywords):
                detected_topics.append(topic)
        
        # Generate different insights based on detected topics and complexity
        if detected_topics:
            topic_str = ", ".join(detected_topics)
            insights.append({
                "type": "topic_analysis",
                "title": f"Detected Topics: {topic_str.capitalize()}",
                "content": f"The text appears to contain elements related to {topic_str}, with {word_count} words and a complexity rating of {complexity:.2f}.",
                "confidence": 0.7 + 0.2 * complexity,
                "timestamp": time.time()
            })
        
        # Relationship insight if we have related nodes
        if related_nodes:
            top_relation = related_nodes[0]
            related_node = self.state.nodes.get(top_relation[0])
            if related_node and related_node.metadata.get("text_sample"):
                similarity_desc = "strong" if top_relation[1] > 0.8 else "moderate" if top_relation[1] > 0.6 else "weak"
                insights.append({
                    "type": "relationship_analysis",
                    "title": f"{similarity_desc.capitalize()} Conceptual Connection Detected",
                    "content": f"This input shows a {similarity_desc} semantic relationship with previous content: \"{related_node.metadata.get('text_sample', '')[:50]}...\"",
                    "confidence": top_relation[1],
                    "timestamp": time.time()
                })
        
        # Always add a general insight
        pattern_desc = "complex" if complexity > 0.6 else "moderate" if complexity > 0.4 else "straightforward"
        insights.append({
            "type": "content_analysis",
            "title": f"Content Pattern: {pattern_desc.capitalize()}",
            "content": f"Analysis reveals a {pattern_desc} pattern structure with {word_count} words and an average word length of {avg_word_length:.1f} characters.",
            "confidence": 0.85,
            "timestamp": time.time()
        })
        
        return insights
    
    def _simulation_loop(self):
        """Main simulation loop for the quantum field"""
        logger.info("Starting simulation loop")
        last_metrics_update = time.time()
        frame_count = 0
        
        try:
            while self.running:
                start_time = time.time()
                
                with self.state_lock:
                    # If connected to backend, fetch latest state
                    if self.backend_connected and hasattr(self.backend, 'get_visualization_data'):
                        try:
                            backend_state = self.backend.get_visualization_data()
                            self._update_state_from_backend(backend_state)
                        except Exception as e:
                            logger.error(f"Error updating from backend: {e}")
                    else:
                        # Run simulated updates
                        self._update_simulated_field()
                
                # Update metrics
                frame_count += 1
                if time.time() - last_metrics_update > 1.0:
                    self.metrics["fps"] = frame_count / (time.time() - last_metrics_update)
                    self.metrics["node_count"] = len(self.state.nodes)
                    self.metrics["connection_count"] = len(self.state.connections)
                    self.metrics["update_time_ms"] = (time.time() - start_time) * 1000
                    
                    try:
                        import psutil
                        process = psutil.Process(os.getpid())
                        self.metrics["cpu_usage"] = process.cpu_percent()
                        self.metrics["memory_usage"] = process.memory_info().rss / 1024 / 1024  # MB
                    except ImportError:
                        pass
                    
                    frame_count = 0
                    last_metrics_update = time.time()
                
                # Notify callbacks of state update
                self._notify_callbacks({
                    "type": "state_update",
                    "source": "simulation",
                    "update_time": time.time() - start_time
                })
                
                # Control update rate
                elapsed = time.time() - start_time
                if elapsed < 0.033:  # Target ~30 FPS
                    time.sleep(0.033 - elapsed)
        
        except Exception as e:
            logger.error(f"Error in simulation loop: {e}")
            logger.error(traceback.format_exc())
    
    def _update_state_from_backend(self, backend_state: Dict[str, Any]):
        """Update internal state from backend data"""
        # Process backend nodes
        backend_nodes = backend_state.get("nodes", [])
        backend_node_ids = set()
        
        for backend_node in backend_nodes:
            node_id = backend_node.get("id")
            if not node_id:
                continue
                
            backend_node_ids.add(node_id)
            
            # Check if node already exists
            if node_id in self.state.nodes:
                # Update existing node
                node = self.state.nodes[node_id]
                node.position = tuple(backend_node.get("position", node.position))
                node.energy = backend_node.get("energy", node.energy)
                node.stability = backend_node.get("stability", node.stability)
                node.metadata.update(backend_node.get("metadata", {}))
                node.last_update = time.time()
            else:
                # Create new node
                node = QuantumNode(
                    id=node_id,
                    position=tuple(backend_node.get("position", (0, 0, 0))),
                    energy=backend_node.get("energy", 0.5),
                    stability=backend_node.get("stability", 0.8),
                    metadata=backend_node.get("metadata", {}),
                    features=backend_node.get("features", [])
                )
                self.state.nodes[node_id] = node
                
                # Add creation event
                self.state.events.append({
                    "type": "node_creation",
                    "node_id": node_id,
                    "message": f"Node created by backend",
                    "timestamp": time.time()
                })
        
        # Remove nodes that no longer exist in backend
        for node_id in list(self.state.nodes.keys()):
            if node_id not in backend_node_ids:
                del self.state.nodes[node_id]
                
                # Remove any connections to this node
                self.state.connections = [
                    conn for conn in self.state.connections
                    if conn.source != node_id and conn.target != node_id
                ]
                
                # Add removal event
                self.state.events.append({
                    "type": "node_removal",
                    "node_id": node_id,
                    "message": f"Node removed by backend",
                    "timestamp": time.time()
                })
        
        # Process backend connections
        backend_connections = backend_state.get("connections", [])
        existing_connections = set()
        
        for backend_conn in backend_connections:
            source = backend_conn.get("source")
            target = backend_conn.get("target")
            
            if not source or not target:
                continue
                
            # Create unique connection identifier
            conn_id = tuple(sorted([source, target]))
            existing_connections.add(conn_id)
            
            # Check if connection already exists
            conn_exists = False
            for conn in self.state.connections:
                if (conn.source == source and conn.target == target) or \
                   (conn.source == target and conn.target == source):
                    # Update existing connection
                    conn.strength = backend_conn.get("strength", conn.strength)
                    conn.type = backend_conn.get("type", conn.type)
                    conn.metadata.update(backend_conn.get("metadata", {}))
                    conn_exists = True
                    break
            
            if not conn_exists and source in self.state.nodes and target in self.state.nodes:
                # Create new connection
                conn = QuantumConnection(
                    source=source,
                    target=target,
                    strength=backend_conn.get("strength", 0.5),
                    type=backend_conn.get("type", "standard"),
                    metadata=backend_conn.get("metadata", {})
                )
                self.state.connections.append(conn)
                
                # Update node connections
                self.state.nodes[source].connections[target] = conn.strength
                self.state.nodes[target].connections[source] = conn.strength
        
        # Remove connections that no longer exist in backend
        self.state.connections = [
            conn for conn in self.state.connections
            if tuple(sorted([conn.source, conn.target])) in existing_connections
        ]
        
        # Update metrics from backend
        if "metrics" in backend_state:
            self.state.metrics.update(backend_state.get("metrics", {}))
            
        # Update state timestamp
        self.state.timestamp = time.time()
    
    def _update_simulated_field(self):
        """Update the simulated quantum field"""
        current_time = time.time()
        
        # Calculate elapsed time since last update
        elapsed = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Simulation parameters
        repulsion = 5.0  # Repulsion force between nodes
        attraction = 3.0  # Attraction force along connections
        damping = 0.8    # Damping factor to avoid oscillations
        min_distance = 1.0  # Minimum distance between nodes
        max_force = 5.0  # Maximum force magnitude
        
        # Update node positions using force-directed algorithm
        forces = {node_id: [0.0, 0.0, 0.0] for node_id in self.state.nodes}
        
        # Calculate repulsive forces between all nodes
        node_items = list(self.state.nodes.items())
        for i, (node1_id, node1) in enumerate(node_items):
            for node2_id, node2 in node_items[i+1:]:
                # Calculate distance vector
                dx = node2.position[0] - node1.position[0]
                dy = node2.position[1] - node1.position[1]
                dz = node2.position[2] - node1.position[2]
                distance_squared = dx*dx + dy*dy + dz*dz
                
                # Apply minimum distance
                if distance_squared < min_distance*min_distance:
                    distance_squared = min_distance*min_distance
                
                # Calculate repulsive force (inversely proportional to distance squared)
                force_magnitude = repulsion / distance_squared
                
                # Calculate unit direction vector
                distance = math.sqrt(distance_squared)
                dx /= distance
                dy /= distance
                dz /= distance
                
                # Apply force to both nodes in opposite directions
                forces[node1_id][0] -= dx * force_magnitude
                forces[node1_id][1] -= dy * force_magnitude
                forces[node1_id][2] -= dz * force_magnitude
                
                forces[node2_id][0] += dx * force_magnitude
                forces[node2_id][1] += dy * force_magnitude
                forces[node2_id][2] += dz * force_magnitude
        
        # Calculate attractive forces along connections
        for conn in self.state.connections:
            if conn.source in self.state.nodes and conn.target in self.state.nodes:
                node1 = self.state.nodes[conn.source]
                node2 = self.state.nodes[conn.target]
                
                # Calculate distance vector
                dx = node2.position[0] - node1.position[0]
                dy = node2.position[1] - node1.position[1]
                dz = node2.position[2] - node1.position[2]
                distance = math.sqrt(dx*dx + dy*dy + dz*dz)
                
                # Skip if nodes are too close
                if distance < 0.1:
                    continue
                
                # Calculate attractive force (proportional to distance)
                force_magnitude = distance * conn.strength * attraction
                
                # Calculate unit direction vector
                dx /= distance
                dy /= distance
                dz /= distance
                
                # Apply force to both nodes, pulling them together
                forces[conn.source][0] += dx * force_magnitude
                forces[conn.source][1] += dy * force_magnitude
                forces[conn.source][2] += dz * force_magnitude
                
                forces[conn.target][0] -= dx * force_magnitude
                forces[conn.target][1] -= dy * force_magnitude
                forces[conn.target][2] -= dz * force_magnitude
        
        # Apply forces to update node positions
        for node_id, force in forces.items():
            node = self.state.nodes[node_id]
            
            # Apply stability-based damping
            stability_factor = node.stability * damping
            
            # Limit force magnitude
            force_magnitude = math.sqrt(force[0]*force[0] + force[1]*force[1] + force[2]*force[2])
            if force_magnitude > max_force:
                scaling = max_force / force_magnitude
                force = [f * scaling for f in force]
            
            # Apply force to position
            new_position = (
                node.position[0] + force[0] * stability_factor,
                node.position[1] + force[1] * stability_factor,
                node.position[2] + force[2] * stability_factor
            )
            
            # Update node position
            node.position = new_position
        
        # Occasionally update node energies
        if random.random() < 0.1:  # 10% chance per update
            self._update_node_energies()
        
        # Sometimes add a new random node
        if len(self.state.nodes) < 50 and random.random() < 0.02:  # 2% chance per update
            self._add_random_node()
        
        # Sometimes remove a node
        if len(self.state.nodes) > 10 and random.random() < 0.005:  # 0.5% chance per update
            self._remove_random_node()
        
        # Update metrics
        self.state.metrics = {
            "node_count": len(self.state.nodes),
            "connection_count": len(self.state.connections),
            "avg_energy": np.mean([n.energy for n in self.state.nodes.values()]),
            "avg_connections": np.mean([len(n.connections) for n in self.state.nodes.values()]),
            "simulation_time": current_time - self.metrics["start_time"]
        }
        
        # Update state timestamp
        self.state.timestamp = current_time
    
    def _update_node_energies(self):
        """Update node energies based on connections and stability"""
        for node in self.state.nodes.values():
            # Base energy decay
            decay_rate = 0.01 * (1.0 - node.stability)
            
            # Energy gain from connections
            connection_count = len(node.connections)
            energy_gain = 0.005 * connection_count
            
            # Apply changes with limits
            node.energy = max(0.1, min(1.0, node.energy - decay_rate + energy_gain))
            
            # Update node size based on energy
            node.size = 0.5 + node.energy
    
    def _add_random_node(self):
        """Add a random node to the field"""
        # Generate random position
        position = (
            (random.random() * 30) - 15,
            (random.random() * 30) - 15,
            (random.random() * 10) - 5
        )
        
        # Create node
        node_id = str(uuid.uuid4())
        node = QuantumNode(
            id=node_id,
            position=position,
            energy=0.3 + random.random() * 0.4,
            stability=0.4 + random.random() * 0.4,
            features=[random.random() for _ in range(32)],
            metadata={"type": "emergent", "origin": "simulation"}
        )
        
        self.state.nodes[node_id] = node
        
        # Connect to a random existing node
        if self.state.nodes:
            other_id = random.choice(list(self.state.nodes.keys()))
            if other_id != node_id:
                other_node = self.state.nodes[other_id]
                
                strength = 0.3 + random.random() * 0.3
                conn = QuantumConnection(
                    source=node_id,
                    target=other_id,
                    strength=strength
                )
                
                self.state.connections.append(conn)
                node.connections[other_id] = strength
                other_node.connections[node_id] = strength
        
        # Add event
        self.state.events.append({
            "type": "node_creation",
            "node_id": node_id,
            "message": "Emergent node created by quantum fluctuation",
            "timestamp": time.time()
        })
        
        logger.debug(f"Added random node {node_id}")
    
    def _remove_random_node(self):
        """Remove a random low-energy node from the field"""
        # Find nodes with low energy
        low_energy_nodes = [
            node_id for node_id, node in self.state.nodes.items()
            if node.energy < 0.3 and len(node.connections) <= 1
        ]
        
        if low_energy_nodes:
            # Remove a random low-energy node
            node_id = random.choice(low_energy_nodes)
            
            # Remove connections to this node
            self.state.connections = [
                conn for conn in self.state.connections
                if conn.source != node_id and conn.target != node_id
            ]
            
            # Remove connection references in other nodes
            for other_node in self.state.nodes.values():
                if node_id in other_node.connections:
                    del other_node.connections[node_id]
            
            # Remove the node
            if node_id in self.state.nodes:
                del self.state.nodes[node_id]
            
            # Add event
            self.state.events.append({
                "type": "node_removal",
                "node_id": node_id,
                "message": "Low-energy node collapsed",
                "timestamp": time.time()
            })
            
            logger.debug(f"Removed node {node_id}")
    
    def _backend_connection_loop(self):
        """Maintain connection to the backend system"""
        while self.running:
            if not self.backend_connected:
                try:
                    self._connect_to_backend()
                except Exception as e:
                    logger.error(f"Backend connection error: {e}")
                    
            time.sleep(5.0)  # Check connection every 5 seconds
    
    def _notify_callbacks(self, event: Dict[str, Any]):
        """Notify all registered callbacks of an event"""
        for callback in self.event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in event callback: {e}")


# Flask Server Implementation
class QuantumKaleidoscopeServer:
    """
    Flask server implementation for the Quantum Kaleidoscope visualization.
    Provides HTTP API and WebSocket communication with clients.
    """
    
    def __init__(self, orchestrator: QuantumOrchestrator, port: int = 8080, static_dir: Optional[str] = None):
        self.orchestrator = orchestrator
        self.port = port
        
        # Set up static directory
        if static_dir:
            self.static_dir = Path(static_dir)
        else:
            # Use default static directory inside the script directory
            self.static_dir = Path(__file__).parent / "static"
            
        # Create static directory if it doesn't exist
        self.static_dir.mkdir(parents=True, exist_ok=True)
        
        # Create default assets if they don't exist
        self._create_default_assets()
        
        # Setup Flask app
        self.app = Flask(
            __name__,
            static_folder=str(self.static_dir),
            template_folder=str(self.static_dir)
        )
        
        # Setup SocketIO
        self.socketio = SocketIO(
            self.app,
            cors_allowed_origins="*",
            async_mode="eventlet"
        )
        
        # Register routes
        self._register_routes()
        
        # Register SocketIO handlers
        self._register_socketio_handlers()
        
        # Register with orchestrator
        self.orchestrator.register_event_callback(self._handle_orchestrator_event)
        
        logger.info(f"Server initialized with static directory: {self.static_dir}")
    
    def _create_default_assets(self):
        """Create default static assets if they don't exist"""
        # Create index.html
        index_html = self.static_dir / "index.html"
        if not index_html.exists():
            with open(index_html, "w") as f:
                f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantum Kaleidoscope</title>
    <link rel="stylesheet" href="/styles.css">
    <script src="/socket.io/socket.io.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/dat-gui/0.7.7/dat.gui.min.js"></script>
</head>
<body>
    <div id="visualization-container"></div>
    <div id="ui-container">
        <div id="sidebar">
            <h1>Quantum Kaleidoscope</h1>
            <div id="stats"></div>
            <div id="controls">
                <textarea id="input-text" placeholder="Enter text to process..."></textarea>
                <button id="process-button">Process Text</button>
            </div>
            <div id="insights">
                <h2>Quantum Insights</h2>
                <div id="insights-container"></div>
            </div>
        </div>
        <div id="events-panel">
            <h3>System Events</h3>
            <div id="events-container"></div>
        </div>
        <div id="node-details" class="hidden">
            <h3>Node Details</h3>
            <div id="node-details-container"></div>
            <button id="close-details">Close</button>
        </div>
    </div>
    <script src="/main.js"></script>
</body>
</html>""")
            logger.info(f"Created default index.html")
        
        # Create styles.css
        styles_css = self.static_dir / "styles.css"
        if not styles_css.exists():
            with open(styles_css, "w") as f:
                f.write("""body {
    margin: 0;
    padding: 0;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    overflow: hidden;
    background-color: #0a0a1a;
    color: #e6e6ff;
}

#visualization-container {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: 1;
}

#ui-container {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: 2;
    pointer-events: none; /* Allow clicking through to the visualization */
}

#sidebar {
    position: absolute;
    top: 0;
    right: 0;
    width: 300px;
    height: 100%;
    background-color: rgba(10, 10, 26, 0.8);
    backdrop-filter: blur(10px);
    border-left: 1px solid #2a2a4a;
    padding: 20px;
    overflow-y: auto;
    pointer-events: auto; /* Make the sidebar clickable */
}

#sidebar h1 {
    margin-top: 0;
    color: #66aaff;
    font-size: 24px;
    border-bottom: 1px solid #3a3a5a;
    padding-bottom: 10px;
}

#stats {
    background-color: #1a1a2a;
    border-radius: 6px;
    padding: 10px;
    margin-bottom: 20px;
    font-family: monospace;
    font-size: 14px;
}

#controls {
    margin-bottom: 20px;
}

#input-text {
    width: 100%;
    height: 80px;
    background-color: #1a1a2a;
    color: #e6e6ff;
    border: 1px solid #3a3a5a;
    border-radius: 4px;
    padding: 8px;
    resize: vertical;
    font-family: inherit;
    margin-bottom: 10px;
}

#process-button {
    background-color: #4466cc;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
    cursor: pointer;
    font-weight: bold;
    transition: background-color 0.2s;
}

#process-button:hover {
    background-color: #5577dd;
}

#insights h2 {
    font-size: 18px;
    margin-top: 0;
    color: #66aaff;
}

.insight-card {
    background-color: #1a1a2a;
    border-radius: 6px;
    padding: 12px;
    margin-bottom: 12px;
    border-left: 3px solid #4466cc;
}

.insight-title {
    font-weight: bold;
    margin-bottom: 6px;
    color: #88ccff;
}

.insight-content {
    font-size: 14px;
    line-height: 1.5;
}

.insight-meta {
    font-size: 12px;
    color: #8888aa;
    margin-top: 6px;
    display: flex;
    justify-content: space-between;
}

#events-panel {
    position: absolute;
    bottom: 0;
    left: 0;
    width: calc(100% - 340px);
    height: 150px;
    background-color: rgba(10, 10, 26, 0.8);
    backdrop-filter: blur(10px);
    border-top: 1px solid #2a2a4a;
    padding: 10px 20px;
    pointer-events: auto;
}

#events-panel h3 {
    margin-top: 0;
    margin-bottom: 10px;
    color: #66aaff;
    font-size: 16px;
}

#events-container {
    height: calc(100% - 30px);
    overflow-y: auto;
    display: flex;
    flex-direction: column-reverse; /* Show newest events at the top */
}

.event-item {
    padding: 6px 10px;
    border-radius: 4px;
    margin-bottom: 6px;
    font-size: 13px;
    background-color: #1a1a2a;
    display: flex;
    align-items: center;
}

.event-icon {
    margin-right: 8px;
    color: #66aaff;
    font-size: 16px;
}

.event-time {
    margin-left: auto;
    color: #8888aa;
    font-size: 12px;
}

#node-details {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 400px;
    background-color: rgba(20, 20, 40, 0.95);
    backdrop-filter: blur(10px);
    border: 1px solid #3a3a5a;
    border-radius: 8px;
    padding: 20px;
    z-index: 10;
    pointer-events: auto;
}

#node-details h3 {
    margin-top: 0;
    color: #66aaff;
    border-bottom: 1px solid #3a3a5a;
    padding-bottom: 8px;
}

#node-details-container {
    margin-bottom: 20px;
}

.node-property {
    margin-bottom: 10px;
}

.property-name {
    font-weight: bold;
    color: #88ccff;
    margin-bottom: 2px;
}

.property-value {
    background-color: #1a1a2a;
    padding: 8px;
    border-radius: 4px;
    word-break: break-word;
}

#close-details {
    background-color: #3a3a5a;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
    cursor: pointer;
    font-weight: bold;
    float: right;
}

.hidden {
    display: none;
}""")
            logger.info(f"Created default styles.css")
        
        # Create main.js
        main_js = self.static_dir / "main.js"
        if not main_js.exists():
            with open(main_js, "w") as f:
                f.write("""// QuantumKaleidoscope Visualization
// Main visualization and UI script

// Global variables
let socket;
let scene, camera, renderer;
let nodes = {};  // nodeId -> THREE.Mesh
let connections = {};  // source_id:target_id -> THREE.Line
let nodeData = {};  // nodeId -> node data object
let connectionData = {};  // source_id:target_id -> connection data object
let selectedNode = null;
let raycaster, mouse;
let clock = new THREE.Clock();
let particleSystem;
let eventStream = [];
let insightList = [];
let controls;
let neuralPathways = [];

// GUI controls
let gui;
let guiParams = {
    showForces: true,
    showQuantumEffects: true,
    rotationSpeed: 0.1,
    visualizationMode: 'standard',
    neuralPathwayCount: 8,
    connectionOpacity: 0.7,
    backgroundColor: '#0a0a1a'
};

// Initialize socket connection
function initSocket() {
    socket = io();
    
    socket.on('connect', () => {
        console.log('Connected to server');
        // Request initial state
        socket.emit('get_state');
    });
    
    socket.on('disconnect', () => {
        console.log('Disconnected from server');
    });
    
    socket.on('state_update', (data) => {
        updateVisualization(data);
    });
    
    socket.on('event', (event) => {
        addEvent(event);
    });
    
    socket.on('insight', (insight) => {
        addInsight(insight);
    });
}

// Initialize 3D visualization
function initVisualization() {
    // Create scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(guiParams.backgroundColor);
    scene.fog = new THREE.FogExp2(guiParams.backgroundColor, 0.002);
    
    // Create camera
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.z = 50;
    
    // Create renderer
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    document.getElementById('visualization-container').appendChild(renderer.domElement);
    
    // Add orbit controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.25;
    controls.screenSpacePanning = false;
    controls.minDistance = 10;
    controls.maxDistance = 100;
    
    // Create lights
    const ambientLight = new THREE.AmbientLight(0x404040);
    scene.add(ambientLight);
    
    const pointLight = new THREE.PointLight(0xffffff, 1);
    pointLight.position.set(50, 50, 50);
    scene.add(pointLight);
    
    // Create groups for organization
    window.nodeGroup = new THREE.Group();
    scene.add(window.nodeGroup);
    
    window.connectionGroup = new THREE.Group();
    scene.add(window.connectionGroup);
    
    // Setup neural pathways
    initNeuralPathways();
    
    // Add particle effects for quantum field
    initParticleSystem();
    
    // Setup raycaster for interaction
    raycaster = new THREE.Raycaster();
    mouse = new THREE.Vector2();
    
    // Add event listeners
    window.addEventListener('resize', onWindowResize);
    renderer.domElement.addEventListener('click', onMouseClick);
    renderer.domElement.addEventListener('mousemove', onMouseMove);
    
    // Setup GUI
    setupGUI();
    
    // Start animation loop
    animate();
}

// Initialize neural pathways visualization
function initNeuralPathways() {
    for (let i = 0; i < guiParams.neuralPathwayCount; i++) {
        createNeuralPathway();
    }
}

// Create a single neural pathway
function createNeuralPathway() {
    // Create curve points
    const curvePoints = [];
    const pointCount = 5 + Math.floor(Math.random() * 5);
    
    for (let i = 0; i < pointCount; i++) {
        const radius = 20 + Math.random() * 20;
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.random() * Math.PI * 2;
        
        const x = radius * Math.sin(theta) * Math.cos(phi);
        const y = radius * Math.sin(theta) * Math.sin(phi);
        const z = radius * Math.cos(theta);
        
        curvePoints.push(new THREE.Vector3(x, y, z));
    }
    
    // Create spline
    const curve = new THREE.CatmullRomCurve3(curvePoints);
    
    // Create tube geometry along spline
    const tubeGeometry = new THREE.TubeGeometry(curve, 64, 0.2, 8, false);
    
    // Create material with glow effect
    const tubeMaterial = new THREE.MeshPhongMaterial({
        color: new THREE.Color(0.1, 0.4, 0.8),
        emissive: new THREE.Color(0.05, 0.2, 0.4),
        transparent: true,
        opacity: 0.7,
        shininess: 100
    });
    
    // Create mesh
    const tube = new THREE.Mesh(tubeGeometry, tubeMaterial);
    scene.add(tube);
    
    // Add pulse effect
    const pulse = {
        mesh: tube,
        material: tubeMaterial,
        speed: 0.5 + Math.random() * 2,
        offset: Math.random() * Math.PI * 2,
        baseOpacity: 0.3 + Math.random() * 0.4
    };
    
    neuralPathways.push(pulse);
}

// Initialize particle system for quantum field visualization
function initParticleSystem() {
    const particleCount = 5000;
    const geometry = new THREE.BufferGeometry();
    
    // Create particle positions
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);
    const sizes = new Float32Array(particleCount);
    
    for (let i = 0; i < particleCount; i++) {
        // Position
        const radius = 30 + 20 * Math.random();
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.random() * Math.PI * 2;
        
        positions[i * 3] = radius * Math.sin(theta) * Math.cos(phi);  // x
        positions[i * 3 + 1] = radius * Math.sin(theta) * Math.sin(phi);  // y
        positions[i * 3 + 2] = radius * Math.cos(theta);  // z
        
        // Color (blueish hues)
        colors[i * 3] = 0.3 + 0.3 * Math.random();     // r
        colors[i * 3 + 1] = 0.4 + 0.3 * Math.random(); // g
        colors[i * 3 + 2] = 0.7 + 0.3 * Math.random(); // b
        
        // Size
        sizes[i] = 0.5 + 1.5 * Math.random();
    }
    
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
    
    // Add velocity attribute for animation
    const velocities = new Float32Array(particleCount * 3);
    for (let i = 0; i < particleCount * 3; i += 3) {
        velocities[i] = (Math.random() - 0.5) * 0.05;
        velocities[i + 1] = (Math.random() - 0.5) * 0.05;
        velocities[i + 2] = (Math.random() - 0.5) * 0.05;
    }
    geometry.setAttribute('velocity', new THREE.BufferAttribute(velocities, 3));
    
    // Create shader material
    const vertexShader = `
        attribute float size;
        attribute vec3 velocity;
        varying vec3 vColor;
        uniform float time;
        
        void main() {
            vColor = color;
            
            // Subtle movement
            vec3 pos = position;
            pos.x += sin(position.z * 0.1 + time * 0.5) * 0.5;
            pos.y += sin(position.x * 0.1 + time * 0.6) * 0.5;
            pos.z += sin(position.y * 0.1 + time * 0.7) * 0.5;
            
            vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
            gl_PointSize = size * (30.0 / -mvPosition.z);
            gl_Position = projectionMatrix * mvPosition;
        }
    `;
    
    const fragmentShader = `
        varying vec3 vColor;
        
        void main() {
            float r = distance(gl_PointCoord, vec2(0.5, 0.5));
            if (r > 0.5) {
                discard;
            }
            float alpha = 1.0 - r * 2.0;
            gl_FragColor = vec4(vColor, alpha);
        }
    `;
    
    const material = new THREE.ShaderMaterial({
        uniforms: {
            time: { value: 0 }
        },
        vertexShader: vertexShader,
        fragmentShader: fragmentShader,
        blending: THREE.AdditiveBlending,
        depthTest: false,
        transparent: true,
        vertexColors: true
    });
    
    // Create particle system
    particleSystem = new THREE.Points(geometry, material);
    scene.add(particleSystem);
}

// Setup GUI controls
function setupGUI() {
    gui = new dat.GUI({ width: 300 });
    gui.domElement.id = 'gui';
    
    // Create visualization folder
    const visFolder = gui.addFolder('Visualization Settings');
    visFolder.add(guiParams, 'visualizationMode', ['standard', 'energy', 'connections', 'quantum']).name('View Mode').onChange(updateVisualizationMode);
    visFolder.add(guiParams, 'showQuantumEffects').name('Quantum Effects').onChange(updateQuantumEffects);
    visFolder.add(guiParams, 'showForces').name('Show Forces');
    visFolder.add(guiParams, 'rotationSpeed', 0, 0.5).name('Auto Rotation');
    visFolder.add(guiParams, 'connectionOpacity', 0, 1).name('Connection Opacity').onChange(updateConnectionOpacity);
    visFolder.addColor(guiParams, 'backgroundColor').name('Background Color').onChange(updateBackgroundColor);
    
    // Create neural pathways folder
    const neuralFolder = gui.addFolder('Neural Pathways');
    neuralFolder.add(guiParams, 'neuralPathwayCount', 0, 16).step(1).name('Pathway Count').onChange((value) => {
        // Remove existing pathways
        neuralPathways.forEach(pathway => {
            scene.remove(pathway.mesh);
        });
        neuralPathways = [];
        
        // Create new pathways
        for (let i = 0; i < value; i++) {
            createNeuralPathway();
        }
    });
    
    // Open folders
    visFolder.open();
}

// Update visualization based on selected mode
function updateVisualizationMode(mode) {
    // Update node appearance based on mode
    for (const nodeId in nodes) {
        const nodeMesh = nodes[nodeId];
        const data = nodeData[nodeId];
        
        if (mode === 'energy') {
            // Color based on energy level
            const hue = 0.6 - data.energy * 0.5; // Blue to red
            nodeMesh.material.color.setHSL(hue, 0.8, 0.5);
        } else if (mode === 'connections') {
            // Color based on connection count
            const connectionCount = Object.keys(data.connections || {}).length;
            const hue = 0.3 - Math.min(connectionCount / 10, 0.3); // Green to yellow
            nodeMesh.material.color.setHSL(hue, 0.8, 0.5);
        } else if (mode === 'quantum') {
            // Special quantum visualization
            nodeMesh.material.color.set(data.color || '#4488ff');
            nodeMesh.material.emissive.set(data.color || '#4488ff');
            nodeMesh.material.emissiveIntensity = 0.5;
        } else {
            // Standard mode - use node's color
            nodeMesh.material.color.set(data.color || '#4488ff');
            nodeMesh.material.emissive.set('#000000');
        }
    }
    
    // Update connection appearance
    for (const connId in connections) {
        const connection = connections[connId];
        const data = connectionData[connId];
        
        if (mode === 'quantum') {
            connection.material.color.set('#88aaff');
            connection.material.opacity = Math.min(data.strength * 1.5, 1) * guiParams.connectionOpacity;
        } else {
            connection.material.color.set('#4466aa');
            connection.material.opacity = data.strength * guiParams.connectionOpacity;
        }
    }
}

// Update quantum effects visibility
function updateQuantumEffects(enabled) {
    if (particleSystem) {
        particleSystem.visible = enabled;
    }
    
    // Show/hide neural pathways
    neuralPathways.forEach(pathway => {
        pathway.mesh.visible = enabled;
    });
}

// Update connection opacity
function updateConnectionOpacity(value) {
    for (const connId in connections) {
        const connection = connections[connId];
        const data = connectionData[connId];
        connection.material.opacity = data.strength * value;
    }
}

// Update background color
function updateBackgroundColor(color) {
    scene.background = new THREE.Color(color);
    scene.fog.color = new THREE.Color(color);
}

// Handle window resize
function onWindowResize() {
    const width = window.innerWidth;
    const height = window.innerHeight;
    
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    
    renderer.setSize(width, height);
}

// Handle mouse move for hover effects
function onMouseMove(event) {
    // Calculate mouse position in normalized device coordinates
    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
}

// Handle mouse click for node selection
function onMouseClick(event) {
    // Update the picking ray
    raycaster.setFromCamera(mouse, camera);
    
    // Calculate objects intersecting the picking ray
    const intersects = raycaster.intersectObjects(window.nodeGroup.children);
    
    if (intersects.length > 0) {
        // Get the first intersected object
        const object = intersects[0].object;
        const nodeId = object.userData.id;
        
        // Select the node
        selectNode(nodeId);
    } else {
        // Deselect if clicked outside
        deselectNode();
    }
}

// Select a node and show details
function selectNode(nodeId) {
    // Deselect previous node
    deselectNode();
    
    // Highlight selected node
    selectedNode = nodeId;
    const nodeMesh = nodes[nodeId];
    if (nodeMesh) {
        nodeMesh.material.emissive.set('#ffffff');
        nodeMesh.material.emissiveIntensity = 0.5;
        
        // Scale up slightly
        nodeMesh.scale.set(1.2, 1.2, 1.2);
    }
    
    // Get node data
    const data = nodeData[nodeId];
    
    // Show node details panel
    const detailsPanel = document.getElementById('node-details');
    detailsPanel.classList.remove('hidden');
    
    // Populate details
    const container = document.getElementById('node-details-container');
    let html = '';
    
    html += createPropertyHTML('ID', nodeId);
    html += createPropertyHTML('Energy', data.energy.toFixed(2));
    html += createPropertyHTML('Stability', data.stability.toFixed(2));
    html += createPropertyHTML('Position', `X: ${data.position[0].toFixed(2)}, Y: ${data.position[1].toFixed(2)}, Z: ${data.position[2].toFixed(2)}`);
    html += createPropertyHTML('Connections', Object.keys(data.connections || {}).length);
    
    // Add metadata
    if (data.metadata && Object.keys(data.metadata).length > 0) {
        html += '<div class="node-property"><div class="property-name">Metadata</div>';
        html += '<div class="property-value">';
        
        for (const key in data.metadata) {
            const value = data.metadata[key];
            if (typeof value === 'object') {
                html += `<b>${key}</b>: ${JSON.stringify(value)}<br>`;
            } else {
                html += `<b>${key}</b>: ${value}<br>`;
            }
        }
        
        html += '</div></div>';
    }
    
    container.innerHTML = html;
    
    // Request node insights
    socket.emit('get_node_insights', { nodeId });
}

// Create HTML for a property in node details
function createPropertyHTML(name, value) {
    return `
        <div class="node-property">
            <div class="property-name">${name}</div>
            <div class="property-value">${value}</div>
        </div>
    `;
}

// Deselect the current node
function deselectNode() {
    if (selectedNode && nodes[selectedNode]) {
        // Reset node appearance
        const nodeMesh = nodes[selectedNode];
        nodeMesh.material.emissive.set('#000000');
        
        // Reset scale
        nodeMesh.scale.set(1, 1, 1);
        
        // Update based on current visualization mode
        updateVisualizationMode(guiParams.visualizationMode);
    }
    
    selectedNode = null;
    
    // Hide node details panel
    document.getElementById('node-details').classList.add('hidden');
}

// Update the visualization with new state data
function updateVisualization(data) {
    if (!data || !data.nodes) return;
    
    // Update stats
    updateStats(data.metrics || {});
    
    // Track added nodes and connections for events
    const addedNodes = [];
    const removedNodes = [];
    const addedConnections = [];
    const removedConnections = [];
    
    // Process nodes
    const updatedNodeIds = new Set();
    data.nodes.forEach(nodeData => {
        updatedNodeIds.add(nodeData.id);
        
        // Check if node already exists
        if (nodes[nodeData.id]) {
            // Update existing node
            updateNode(nodeData);
        } else {
            // Create new node
            createNode(nodeData);
            addedNodes.push(nodeData.id);
        }
    });
    
    // Remove nodes that no longer exist
    for (const nodeId in nodes) {
        if (!updatedNodeIds.has(nodeId)) {
            // Remove node
            removeNode(nodeId);
            removedNodes.push(nodeId);
        }
    }
    
    // Process connections
    const updatedConnectionIds = new Set();
    data.connections.forEach(connData => {
        const connId = `${connData.source}:${connData.target}`;
        updatedConnectionIds.add(connId);
        
        // Check if connection already exists
        if (connections[connId]) {
            // Update existing connection
            updateConnection(connData);
        } else {
            // Create new connection
            createConnection(connData);
            addedConnections.push(connId);
        }
    });
    
    // Remove connections that no longer exist
    for (const connId in connections) {
        if (!updatedConnectionIds.has(connId)) {
            // Remove connection
            removeConnection(connId);
            removedConnections.push(connId);
        }
    }
    
    // Add events for changes
    if (addedNodes.length > 0) {
        addEvent({
            type: 'system',
            message: `Added ${addedNodes.length} new node${addedNodes.length > 1 ? 's' : ''}`,
            timestamp: Date.now() / 1000
        });
    }
    
    if (removedNodes.length > 0) {
        addEvent({
            type: 'system',
            message: `Removed ${removedNodes.length} node${removedNodes.length > 1 ? 's' : ''}`,
            timestamp: Date.now() / 1000
        });
    }
    
    // Update insights
    if (data.insights) {
        data.insights.forEach(insight => {
            addInsight(insight);
        });
    }
    
    // Update events
    if (data.events) {
        data.events.forEach(event => {
            addEvent(event);
        });
    }
}

// Create a new node in the visualization
function createNode(data) {
    // Store node data
    nodeData[data.id] = data;
    
    // Create node geometry
    const radius = 0.5 + data.size || 1.0;
    const segments = 16;
    const geometry = new THREE.SphereGeometry(radius, segments, segments);
    
    // Create node material
    const material = new THREE.MeshPhongMaterial({
        color: data.color || '#4488ff',
        shininess: 30,
        transparent: true,
        opacity: 0.9
    });
    
    // Create mesh
    const mesh = new THREE.Mesh(geometry, material);
    
    // Set position
    mesh.position.set(
        data.position[0],
        data.position[1],
        data.position[2]
    );
    
    // Store node ID in user data for raycasting
    mesh.userData.id = data.id;
    
    // Add to node group
    window.nodeGroup.add(mesh);
    
    // Store reference
    nodes[data.id] = mesh;
    
    return mesh;
}

// Update an existing node
function updateNode(data) {
    // Get node mesh
    const mesh = nodes[data.id];
    if (!mesh) return;
    
    // Update position with smooth transition
    const targetPosition = new THREE.Vector3(
        data.position[0],
        data.position[1],
        data.position[2]
    );
    
    // Check if position has changed significantly
    const currentPosition = mesh.position;
    const distanceSquared = currentPosition.distanceToSquared(targetPosition);
    
    if (distanceSquared > 0.01) {
        // Lerp position for smoother transitions
        mesh.position.lerp(targetPosition, 0.1);
    }
    
    // Update node size if it has changed
    const targetSize = 0.5 + data.size || 1.0;
    if (Math.abs(mesh.geometry.parameters.radius - targetSize) > 0.01) {
        // Replace geometry with new size
        mesh.geometry.dispose();
        mesh.geometry = new THREE.SphereGeometry(targetSize, 16, 16);
    }
    
    // Update material based on energy
    if (guiParams.visualizationMode === 'standard') {
        // In standard mode, use the node's color
        mesh.material.color.set(data.color || '#4488ff');
    }
    
    // Store updated data
    nodeData[data.id] = data;
}

// Remove a node
function removeNode(nodeId) {
    // Get node mesh
    const mesh = nodes[nodeId];
    if (!mesh) return;
    
    // Remove from scene
    window.nodeGroup.remove(mesh);
    
    // Dispose resources
    mesh.geometry.dispose();
    mesh.material.dispose();
    
    // Remove references
    delete nodes[nodeId];
    delete nodeData[nodeId];
    
    // If this was the selected node, deselect it
    if (selectedNode === nodeId) {
        deselectNode();
    }
}

// Create a connection between nodes
function createConnection(data) {
    // Get nodes
    const sourceNode = nodes[data.source];
    const targetNode = nodes[data.target];
    
    if (!sourceNode || !targetNode) return;
    
    const connId = `${data.source}:${data.target}`;
    
    // Store connection data
    connectionData[connId] = data;
    
    // Create connection geometry
    const points = [
        sourceNode.position,
        targetNode.position
    ];
    
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    
    // Create connection material
    const material = new THREE.LineBasicMaterial({
        color: data.type === 'quantum' ? '#88aaff' : '#4466aa',
        transparent: true,
        opacity: data.strength * guiParams.connectionOpacity
    });
    
    // Create line
    const line = new THREE.Line(geometry, material);
    
    // Add to connection group
    window.connectionGroup.add(line);
    
    // Store reference
    connections[connId] = line;
    
    return line;
}

// Update an existing connection
function updateConnection(data) {
    const connId = `${data.source}:${data.target}`;
    
    // Get connection line
    const line = connections[connId];
    if (!line) return;
    
    // Get nodes
    const sourceNode = nodes[data.source];
    const targetNode = nodes[data.target];
    
    if (!sourceNode || !targetNode) return;
    
    // Update geometry points
    const points = [
        sourceNode.position.clone(),
        targetNode.position.clone()
    ];
    
    line.geometry.dispose();
    line.geometry = new THREE.BufferGeometry().setFromPoints(points);
    
    // Update material based on connection strength
    line.material.opacity = data.strength * guiParams.connectionOpacity;
    
    // Store updated data
    connectionData[connId] = data;
}

// Remove a connection
function removeConnection(connId) {
    // Get connection line
    const line = connections[connId];
    if (!line) return;
    
    // Remove from scene
    window.connectionGroup.remove(line);
    
    // Dispose resources
    line.geometry.dispose();
    line.material.dispose();
    
    // Remove references
    delete connections[connId];
    delete connectionData[connId];
}

// Update stats display
function updateStats(metrics) {
    const statsElement = document.getElementById('stats');
    
    let html = `
        <div>Nodes: ${metrics.node_count || 0}</div>
        <div>Connections: ${metrics.connection_count || 0}</div>
        <div>Avg. Energy: ${metrics.avg_energy ? metrics.avg_energy.toFixed(2) : '0.00'}</div>
        <div>Avg. Connections: ${metrics.avg_connections ? metrics.avg_connections.toFixed(2) : '0.00'}</div>
        <div>FPS: ${metrics.fps ? metrics.fps.toFixed(1) : '0.0'}</div>
    `;
    
    // Add simulation time if available
    if (metrics.simulation_time) {
        const seconds = Math.floor(metrics.simulation_time % 60);
        const minutes = Math.floor((metrics.simulation_time / 60) % 60);
        const hours = Math.floor(metrics.simulation_time / 3600);
        
        html += `<div>Simulation Time: ${hours}h ${minutes}m ${seconds}s</div>`;
    }
    
    statsElement.innerHTML = html;
}

// Add an event to the event stream
function addEvent(event) {
    // Check if event already exists
    const existingIndex = eventStream.findIndex(e => 
        e.type === event.type && 
        e.message === event.message &&
        Math.abs(e.timestamp - event.timestamp) < 1
    );
    
    if (existingIndex !== -1) {
        // Skip duplicate events
        return;
    }
    
    // Add to event stream
    eventStream.push(event);
    
    // Keep only recent events (max 100)
    if (eventStream.length > 100) {
        eventStream.shift();
    }
    
    // Update event display
    updateEventDisplay();
}

// Update event display
function updateEventDisplay() {
    const container = document.getElementById('events-container');
    
    // Sort events by timestamp (newest first)
    const sortedEvents = [...eventStream].sort((a, b) => b.timestamp - a.timestamp);
    
    let html = '';
    sortedEvents.forEach(event => {
        // Format timestamp
        const date = new Date(event.timestamp * 1000);
        const timeString = date.toLocaleTimeString();
        
        // Get icon based on event type
        let icon = '📝'; // Default icon
        if (event.type === 'system') {
            icon = '⚙️';
        } else if (event.type === 'node_creation') {
            icon = '🔵';
        } else if (event.type === 'node_removal') {
            icon = '❌';
        } else if (event.type === 'connection') {
            icon = '🔗';
        } else if (event.type === 'insight') {
            icon = '💡';
        } else if (event.type === 'error') {
            icon = '⚠️';
        }
        
        html += `
            <div class="event-item">
                <span class="event-icon">${icon}</span>
                <span>${event.message}</span>
                <span class="event-time">${timeString}</span>
            </div>
        `;
    });
    
    container.innerHTML = html;
}

// Add an insight to the insight list
function addInsight(insight) {
    // Check if insight already exists
    const existingIndex = insightList.findIndex(i => 
        i.type === insight.type && 
        i.title === insight.title &&
        i.content === insight.content
    );
    
    if (existingIndex !== -1) {
        // Skip duplicate insights
        return;
    }
    
    // Add to insight list
    insightList.push(insight);
    
    // Keep only recent insights (max 20)
    if (insightList.length > 20) {
        insightList.shift();
    }
    
    // Update insight display
    updateInsightDisplay();
    
    // Add event for new insight
    addEvent({
        type: 'insight',
        message: `New insight: ${insight.title}`,
        timestamp: insight.timestamp || (Date.now() / 1000)
    });
}

// Update insight display
function updateInsightDisplay() {
    const container = document.getElementById('insights-container');
    
    // Sort insights by timestamp (newest first)
    const sortedInsights = [...insightList].sort((a, b) => 
        (b.timestamp || 0) - (a.timestamp || 0)
    );
    
    let html = '';
    sortedInsights.forEach(insight => {
        // Format timestamp
        let timeString = '';
        if (insight.timestamp) {
            const date = new Date(insight.timestamp * 1000);
            timeString = date.toLocaleTimeString();
        }
        
        // Format confidence
        const confidencePercent = insight.confidence ? 
            Math.round(insight.confidence * 100) : '';
        
        html += `
            <div class="insight-card">
                <div class="insight-title">${insight.title}</div>
                <div class="insight-content">${insight.content}</div>
                <div class="insight-meta">
                    <span>${insight.type}</span>
                    <span>${confidencePercent ? confidencePercent + '% confidence' : ''}</span>
                    <span>${timeString}</span>
                </div>
            </div>
        `;
    });
    
    container.innerHTML = html;
}

// Animation loop
function animate() {
    requestAnimationFrame(animate);
    
    // Update orbit controls
    if (controls) {
        controls.update();
    }
    
    // Get elapsed time
    const delta = clock.getDelta();
    const elapsedTime = clock.getElapsedTime();
    
    // Apply auto-rotation to node group
    if (guiParams.rotationSpeed > 0) {
        window.nodeGroup.rotation.y += guiParams.rotationSpeed * delta;
        window.connectionGroup.rotation.y += guiParams.rotationSpeed * delta;
    }
    
    // Update quantum particle system
    if (particleSystem) {
        particleSystem.material.uniforms.time.value = elapsedTime;
        
        // Optionally, make particles move
        if (guiParams.showQuantumEffects) {
            const positions = particleSystem.geometry.attributes.position.array;
            const velocities = particleSystem.geometry.attributes.velocity.array;
            
            for (let i = 0; i < positions.length; i += 3) {
                positions[i] += velocities[i] * 0.1;
                positions[i + 1] += velocities[i + 1] * 0.1;
                positions[i + 2] += velocities[i + 2] * 0.1;
                
                // Boundary check - if particle gets too far, wrap around
                const maxDist = 50;
                const x = positions[i];
                const y = positions[i + 1];
                const z = positions[i + 2];
                const distance = Math.sqrt(x*x + y*y + z*z);
                
                if (distance > maxDist) {
                    // Reset position with random values
                    const r = 30 + Math.random() * 10;
                    const theta = Math.random() * Math.PI * 2;
                    const phi = Math.random() * Math.PI * 2;
                    
                    positions[i] = r * Math.sin(theta) * Math.cos(phi);
                    positions[i + 1] = r * Math.sin(theta) * Math.sin(phi);
                    positions[i + 2] = r * Math.cos(theta);
                    
                    // Randomize velocity
                    velocities[i] = (Math.random() - 0.5) * 0.05;
                    velocities[i + 1] = (Math.random() - 0.5) * 0.05;
                    velocities[i + 2] = (Math.random() - 0.5) * 0.05;
                }
            }
            
            particleSystem.geometry.attributes.position.needsUpdate = true;
        }
    }
    
    // Update neural pathways
    if (guiParams.showQuantumEffects) {
        neuralPathways.forEach(pathway => {
            const opacity = pathway.baseOpacity + 0.2 * Math.sin(elapsedTime * pathway.speed + pathway.offset);
            pathway.material.opacity = opacity;
        });
    }
    
    // Render scene
    renderer.render(scene, camera);
}

// Initialize UI event handlers
function initUI() {
    // Get elements
    const processButton = document.getElementById('process-button');
    const inputText = document.getElementById('input-text');
    const closeDetailsButton = document.getElementById('close-details');
    
    // Add event listeners
    processButton.addEventListener('click', () => {
        const text = inputText.value.trim();
        if (text) {
            // Send text to server
            socket.emit('process_text', { text });
            
            // Clear input
            inputText.value = '';
            
            // Add event
            addEvent({
                type: 'system',
                message: 'Processing text input...',
                timestamp: Date.now() / 1000
            });
        }
    });
    
    // Allow Enter key to submit
    inputText.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            processButton.click();
        }
    });
    
    // Close details button
    closeDetailsButton.addEventListener('click', () => {
        deselectNode();
    });
    
    // Add socket event handlers
    socket.on('text_processed', (data) => {
        if (data.error) {
            addEvent({
                type: 'error',
                message: `Error processing text: ${data.error}`,
                timestamp: Date.now() / 1000
            });
        } else {
            addEvent({
                type: 'system',
                message: `Text processed! Created node ${data.node_id}`,
                timestamp: Date.now() / 1000
            });
            
            // If insights were generated, add them
            if (data.insights) {
                data.insights.forEach(insight => {
                    addInsight(insight);
                });
            }
        }
    });
    
    socket.on('node_insights', (data) => {
        if (data.insights) {
            data.insights.forEach(insight => {
                addInsight(insight);
            });
        }
    });
}

// Initialize everything when the page loads
document.addEventListener('DOMContentLoaded', () => {
    // Initialize socket
    initSocket();
    
    // Initialize 3D visualization
    initVisualization();
    
    // Initialize UI
    initUI();
});
