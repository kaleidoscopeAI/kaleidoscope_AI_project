#!/usr/bin/env python3
"""
Enhanced Quantum Kaleidoscope System
====================================

An advanced integration of quantum-inspired neural networks with real-time
visualization capabilities. This system extends the original Kaleidoscope
with improved data processing, enhanced visualization, and better integration
between components.

Features:
- Improved 3D visualization with adaptive resolution
- Advanced quantum state simulation with enhanced coherence
- Real-time insight generation with natural language processing
- Self-optimizing network topology
- Integrated metrics and performance monitoring

Author: Jacob's Assistant (Claude)
"""

import os
import sys
import time
import json
import asyncio
import logging
import argparse
import threading
import uuid
import math
import random
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import traceback

# Setup virtual environment if needed
def setup_virtual_environment():
    """Set up and activate a virtual environment if it doesn't exist."""
    venv_dir = ".venv"
    if not os.path.exists(venv_dir):
        import subprocess
        print(f"Creating virtual environment in {venv_dir}...")
        subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)
        
        # Install dependencies
        if os.name == 'nt':  # Windows
            pip_path = os.path.join(venv_dir, "Scripts", "pip")
        else:  # Unix/Linux/Mac
            pip_path = os.path.join(venv_dir, "bin", "pip")
            
        print("Installing dependencies...")
        subprocess.run([pip_path, "install", "numpy", "websockets", "flask", "psutil", "networkx"], check=True)
    
    # Activate the virtual environment
    if os.name == 'nt':  # Windows
        activate_script = os.path.join(venv_dir, "Scripts", "activate")
    else:  # Unix/Linux/Mac
        activate_script = os.path.join(venv_dir, "bin", "activate")
    
    # Return the activation command for the user
    return f"source {activate_script}" if os.name != 'nt' else activate_script

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("quantum_kaleidoscope.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("QuantumKaleidoscope")

#=====================================================================
# Core Data Structures
#=====================================================================

@dataclass
class QuantumState:
    """Represents a quantum state in the neural field."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    vector: np.ndarray = field(default_factory=lambda: np.zeros(128))
    energy: float = 0.5
    coherence: float = 0.8
    entanglement: Dict[str, float] = field(default_factory=dict)
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    metadata: Dict[str, Any] = field(default_factory=dict)
    creation_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the quantum state to a dictionary."""
        return {
            "id": self.id,
            "vector": self.vector.tolist(),
            "energy": self.energy,
            "coherence": self.coherence,
            "entanglement": self.entanglement,
            "position": self.position,
            "metadata": self.metadata,
            "creation_time": self.creation_time,
            "last_update": self.last_update
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QuantumState':
        """Create a quantum state from a dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            vector=np.array(data.get("vector", [])),
            energy=data.get("energy", 0.5),
            coherence=data.get("coherence", 0.8),
            entanglement=data.get("entanglement", {}),
            position=data.get("position", [0.0, 0.0, 0.0]),
            metadata=data.get("metadata", {}),
            creation_time=data.get("creation_time", time.time()),
            last_update=data.get("last_update", time.time())
        )


@dataclass
class Insight:
    """Represents an insight generated from the quantum field."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = "pattern"  # pattern, correlation, prediction, anomaly
    source_nodes: List[str] = field(default_factory=list)
    content: str = ""
    confidence: float = 0.5
    creation_time: float = field(default_factory=time.time)
    vector: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the insight to a dictionary."""
        return {
            "id": self.id,
            "type": self.type,
            "source_nodes": self.source_nodes,
            "content": self.content,
            "confidence": self.confidence,
            "creation_time": self.creation_time,
            "vector": self.vector.tolist() if self.vector is not None else None,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Insight':
        """Create an insight from a dictionary."""
        vector_data = data.get("vector")
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            type=data.get("type", "pattern"),
            source_nodes=data.get("source_nodes", []),
            content=data.get("content", ""),
            confidence=data.get("confidence", 0.5),
            creation_time=data.get("creation_time", time.time()),
            vector=np.array(vector_data) if vector_data else None,
            metadata=data.get("metadata", {})
        )


@dataclass
class VisualizationSettings:
    """Settings for the visualization system."""
    node_size_scale: float = 1.0
    connection_scale: float = 1.0
    color_mode: str = "energy"  # energy, coherence, entanglement, type
    glow_intensity: float = 0.5
    background_color: str = "#0a0a1a"
    node_color_palette: Dict[str, str] = field(default_factory=lambda: {
        "default": "#4488ff",
        "high_energy": "#ff4488",
        "high_coherence": "#44ff88",
        "entangled": "#ffaa22"
    })
    show_labels: bool = False
    auto_rotate: bool = True
    render_quality: str = "medium"  # low, medium, high, ultra
    max_visible_nodes: int = 1000
    max_visible_connections: int = 2000


#=====================================================================
# Quantum Field Simulator
#=====================================================================

class QuantumFieldSimulator:
    """
    Advanced quantum-inspired neural field simulator.
    
    This component simulates the behavior of quantum states in a neural field,
    including entanglement, coherence decay, and energy propagation.
    """
    
    def __init__(self, dimension: int = 128, data_dir: str = "./data"):
        self.dimension = dimension
        self.data_dir = data_dir
        self.quantum_states: Dict[str, QuantumState] = {}
        self.insights: List[Insight] = []
        
        # Simulation parameters
        self.coherence_decay_rate = 0.01
        self.entanglement_threshold = 0.7
        self.energy_transfer_rate = 0.05
        self.spontaneous_insight_rate = 0.02
        self.mutation_rate = 0.1
        
        # Simulation state
        self.simulation_step = 0
        self.start_time = time.time()
        self.last_mutation_time = time.time()
        self.simulation_lock = threading.RLock()
        
        # Performance metrics
        self.metrics = {
            "simulation_speed": 0,  # steps per second
            "avg_energy": 0,
            "total_entanglements": 0,
            "insight_generation_rate": 0,
            "active_nodes": 0
        }
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # Load existing state if available
        self._load_state()
        
        logger.info(f"Quantum Field Simulator initialized with dimension {dimension}")
    
    def _load_state(self) -> None:
        """Load the simulator state from disk."""
        state_file = os.path.join(self.data_dir, "quantum_state.json")
        insights_file = os.path.join(self.data_dir, "insights.json")
        
        try:
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    states_data = json.load(f)
                    
                for state_data in states_data:
                    state = QuantumState.from_dict(state_data)
                    self.quantum_states[state.id] = state
                
                logger.info(f"Loaded {len(self.quantum_states)} quantum states from {state_file}")
            
            if os.path.exists(insights_file):
                with open(insights_file, 'r') as f:
                    insights_data = json.load(f)
                    
                for insight_data in insights_data:
                    insight = Insight.from_dict(insight_data)
                    self.insights.append(insight)
                
                logger.info(f"Loaded {len(self.insights)} insights from {insights_file}")
        
        except Exception as e:
            logger.error(f"Error loading state: {e}")
            traceback.print_exc()
    
    def _save_state(self) -> None:
        """Save the simulator state to disk."""
        state_file = os.path.join(self.data_dir, "quantum_state.json")
        insights_file = os.path.join(self.data_dir, "insights.json")
        
        try:
            # Save quantum states
            states_data = [state.to_dict() for state in self.quantum_states.values()]
            with open(state_file, 'w') as f:
                json.dump(states_data, f, indent=2)
            
            # Save insights
            insights_data = [insight.to_dict() for insight in self.insights]
            with open(insights_file, 'w') as f:
                json.dump(insights_data, f, indent=2)
            
            logger.info(f"Saved {len(self.quantum_states)} quantum states and {len(self.insights)} insights")
        
        except Exception as e:
            logger.error(f"Error saving state: {e}")
            traceback.print_exc()
    
    def create_quantum_state(self, 
                           vector: Optional[np.ndarray] = None,
                           energy: float = 0.5,
                           coherence: float = 0.8,
                           position: Optional[List[float]] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new quantum state in the field.
        
        Args:
            vector: State vector (random if None)
            energy: Initial energy level (0.0 to 1.0)
            coherence: Initial coherence (0.0 to 1.0)
            position: Position in 3D space (random if None)
            metadata: Additional metadata
            
        Returns:
            ID of the created state
        """
        with self.simulation_lock:
            # Generate random vector if not provided
            if vector is None:
                vector = np.random.normal(0, 1, self.dimension)
                # Normalize the vector
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm
            
            # Ensure vector has the right dimension
            if len(vector) != self.dimension:
                if len(vector) > self.dimension:
                    vector = vector[:self.dimension]
                else:
                    # Pad with zeros
                    padded = np.zeros(self.dimension)
                    padded[:len(vector)] = vector
                    vector = padded
            
            # Generate random position if not provided
            if position is None:
                position = [
                    random.uniform(-10, 10),
                    random.uniform(-10, 10),
                    random.uniform(-10, 10)
                ]
            
            # Create quantum state
            state = QuantumState(
                vector=vector,
                energy=min(1.0, max(0.0, energy)),
                coherence=min(1.0, max(0.0, coherence)),
                position=position,
                metadata=metadata or {}
            )
            
            # Store the state
            self.quantum_states[state.id] = state
            
            # Update metrics
            self.metrics["active_nodes"] = len(self.quantum_states)
            
            logger.info(f"Created quantum state {state.id} with energy {energy:.2f}, coherence {coherence:.2f}")
            
            return state.id
    
    def delete_quantum_state(self, state_id: str) -> bool:
        """
        Delete a quantum state from the field.
        
        Args:
            state_id: ID of the state to delete
            
        Returns:
            True if deleted, False if not found
        """
        with self.simulation_lock:
            if state_id not in self.quantum_states:
                return False
            
            # Remove from quantum states
            del self.quantum_states[state_id]
            
            # Remove entanglement references
            for state in self.quantum_states.values():
                if state_id in state.entanglement:
                    del state.entanglement[state_id]
            
            # Update metrics
            self.metrics["active_nodes"] = len(self.quantum_states)
            
            logger.info(f"Deleted quantum state {state_id}")
            
            return True
    
    def entangle_states(self, state1_id: str, state2_id: str, strength: Optional[float] = None) -> bool:
        """
        Create an entanglement between two quantum states.
        
        Args:
            state1_id: ID of the first state
            state2_id: ID of the second state
            strength: Entanglement strength (0.0 to 1.0), calculated from state vectors if None
            
        Returns:
            True if entanglement created, False if either state doesn't exist
        """
        with self.simulation_lock:
            if state1_id not in self.quantum_states or state2_id not in self.quantum_states:
                return False
            
            state1 = self.quantum_states[state1_id]
            state2 = self.quantum_states[state2_id]
            
            # Calculate entanglement strength if not provided
            if strength is None:
                # Cosine similarity between state vectors
                dot_product = np.dot(state1.vector, state2.vector)
                norm1 = np.linalg.norm(state1.vector)
                norm2 = np.linalg.norm(state2.vector)
                
                if norm1 > 0 and norm2 > 0:
                    cosine_sim = dot_product / (norm1 * norm2)
                    # Convert from [-1, 1] to [0, 1]
                    strength = (cosine_sim + 1) / 2
                else:
                    strength = 0.5
            
            # Create bidirectional entanglement
            state1.entanglement[state2_id] = strength
            state2.entanglement[state1_id] = strength
            
            # Update metrics
            self.metrics["total_entanglements"] += 1
            
            logger.info(f"Entangled states {state1_id} and {state2_id} with strength {strength:.2f}")
            
            return True
    
    def remove_entanglement(self, state1_id: str, state2_id: str) -> bool:
        """
        Remove an entanglement between two quantum states.
        
        Args:
            state1_id: ID of the first state
            state2_id: ID of the second state
            
        Returns:
            True if entanglement removed, False if not found
        """
        with self.simulation_lock:
            if state1_id not in self.quantum_states or state2_id not in self.quantum_states:
                return False
            
            state1 = self.quantum_states[state1_id]
            state2 = self.quantum_states[state2_id]
            
            # Remove entanglement references
            removed = False
            if state2_id in state1.entanglement:
                del state1.entanglement[state2_id]
                removed = True
            
            if state1_id in state2.entanglement:
                del state2.entanglement[state1_id]
                removed = True
            
            if removed:
                logger.info(f"Removed entanglement between {state1_id} and {state2_id}")
            
            return removed
    
    def process_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process text input to create quantum states and generate insights.
        
        Args:
            text: Input text
            metadata: Additional metadata
            
        Returns:
            Dictionary with processing results
        """
        with self.simulation_lock:
            # Start timing
            start_time = time.time()
            
            # Create metadata
            if metadata is None:
                metadata = {}
            
            metadata.update({
                "type": "text",
                "timestamp": time.time(),
                "length": len(text),
                "source": "user_input"
            })
            
            # Convert text to vector representation (simple encoding)
            vector = self._text_to_vector(text)
            
            # Create quantum state with higher energy for user input
            state_id = self.create_quantum_state(
                vector=vector,
                energy=0.9,  # High initial energy
                coherence=0.8,
                metadata={**metadata, "text": text[:100] + "..." if len(text) > 100 else text}
            )
            
            # Find related states
            related_states = self._find_related_states(state_id, max_count=5, min_similarity=0.6)
            
            # Create entanglements with related states
            for related_id, similarity in related_states:
                self.entangle_states(state_id, related_id, similarity)
            
            # Generate insights
            insights = self._generate_insights_from_text(text, state_id, related_states)
            
            # Create response
            result = {
                "state_id": state_id,
                "related_states": related_states,
                "insights": [insight.to_dict() for insight in insights],
                "processing_time": time.time() - start_time
            }
            
            # Save state periodically
            if random.random() < 0.1:
                self._save_state()
            
            return result
    
    def _text_to_vector(self, text: str) -> np.ndarray:
        """
        Convert text to a vector representation.
        
        This implementation uses a hashbased approach for simplicity.
        A more advanced implementation would use a proper text embedding model.
        
        Args:
            text: Input text
            
        Returns:
            Vector representation of the text
        """
        import hashlib
        
        # Initialize vector
        vector = np.zeros(self.dimension)
        
        # Normalize and tokenize text
        text = text.lower().strip()
        words = text.split()
        
        # Process each word
        for i, word in enumerate(words):
            # Hash the word
            word_hash = hashlib.md5(word.encode()).digest()
            
            # Convert hash to sequence of floats
            for j in range(min(16, len(word_hash))):
                idx = (i + j) % self.dimension
                # Use hash byte value to influence the vector
                byte_val = word_hash[j]
                vector[idx] += (byte_val / 255.0) * 2 - 1  # Scale to [-1, 1]
        
        # Apply non-linearity
        vector = np.tanh(vector)
        
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def _find_related_states(self, 
                           state_id: str, 
                           max_count: int = 5, 
                           min_similarity: float = 0.5) -> List[Tuple[str, float]]:
        """
        Find states related to the given state based on vector similarity.
        
        Args:
            state_id: ID of the reference state
            max_count: Maximum number of related states to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of (state_id, similarity) tuples for related states
        """
        if state_id not in self.quantum_states:
            return []
        
        reference_state = self.quantum_states[state_id]
        
        # Calculate similarity with all other states
        similarities = []
        for other_id, other_state in self.quantum_states.items():
            if other_id == state_id:
                continue
            
            # Cosine similarity
            similarity = self._calculate_similarity(reference_state.vector, other_state.vector)
            
            if similarity >= min_similarity:
                similarities.append((other_id, similarity))
        
        # Sort by similarity (descending) and take top matches
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:max_count]
    
    def _calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        dot_product = np.dot(vec1, vec2)
        similarity = dot_product / (norm1 * norm2)
        
        # Normalize to [0, 1] range
        return (similarity + 1) / 2
    
    def _generate_insights_from_text(self, 
                                   text: str, 
                                   state_id: str,
                                   related_states: List[Tuple[str, float]]) -> List[Insight]:
        """
        Generate insights based on the input text and related states.
        
        Args:
            text: Input text
            state_id: ID of the state created from the text
            related_states: List of related states found
            
        Returns:
            List of generated insights
        """
        insights = []
        
        # Simple text analysis for pattern detection
        words = text.lower().split()
        word_count = len(words)
        unique_words = len(set(words))
        
        # Create content analysis insight
        content_insight = Insight(
            type="content_analysis",
            source_nodes=[state_id],
            content=f"This input contains {word_count} words with {unique_words} unique terms, "
                   f"showing a {unique_words/word_count:.1%} vocabulary density.",
            confidence=0.85
        )
        insights.append(content_insight)
        
        # Create connection insight if related states found
        if related_states:
            related_ids = [rs[0] for rs in related_states]
            avg_similarity = sum(rs[1] for rs in related_states) / len(related_states)
            
            connection_insight = Insight(
                type="connection_analysis",
                source_nodes=[state_id] + related_ids,
                content=f"This input shows {len(related_states)} significant connections to existing "
                       f"knowledge with {avg_similarity:.1%} average similarity.",
                confidence=min(0.95, avg_similarity + 0.3)
            )
            insights.append(connection_insight)
            
            # Create deeper insight by analyzing the most related state
            if related_states:
                most_related_id, similarity = related_states[0]
                most_related_state = self.quantum_states[most_related_id]
                
                if "text" in most_related_state.metadata:
                    related_text = most_related_state.metadata["text"]
                    synthesis_insight = Insight(
                        type="synthesis",
                        source_nodes=[state_id, most_related_id],
                        content=f"This input appears to be conceptually related to previous information "
                               f"about '{related_text}', suggesting a possible thematic connection.",
                        confidence=similarity
                    )
                    insights.append(synthesis_insight)
        
        # Create a speculative insight
        topics = [
            "data analysis", "quantum computing", "visualization techniques",
            "neural networks", "pattern recognition", "emergent behavior",
            "machine learning", "complex systems", "information theory"
        ]
        selected_topic = random.choice(topics)
        
        speculative_insight = Insight(
            type="speculation",
            source_nodes=[state_id],
            content=f"This information might have applications in {selected_topic}, "
                   f"where similar patterns could reveal novel approaches.",
            confidence=0.6
        )
        insights.append(speculative_insight)
        
        # Store insights
        self.insights.extend(insights)
        
        # Update metrics
        self.metrics["insight_generation_rate"] = len(self.insights) / (time.time() - self.start_time) * 3600
        
        return insights
    
    def run_simulation_step(self) -> Dict[str, Any]:
        """
        Run a single step of the quantum field simulation.
        
        This updates state energies, coherence, entanglements, and
        may generate spontaneous insights or mutations.
        
        Returns:
            Dictionary with simulation results for this step
        """
        with self.simulation_lock:
            step_start_time = time.time()
            self.simulation_step += 1
            
            # Skip if no states to simulate
            if not self.quantum_states:
                return {"step": self.simulation_step, "changes": 0}
            
            # Track changes
            changes = 0
            
            # 1. Update coherence for all states (gradual decay)
            for state in self.quantum_states.values():
                original_coherence = state.coherence
                # Apply coherence decay
                state.coherence *= (1 - self.coherence_decay_rate)
                state.coherence = max(0.1, state.coherence)  # Minimum coherence
                
                if abs(state.coherence - original_coherence) > 0.01:
                    changes += 1
            
            # 2. Energy propagation through entanglements
            energy_deltas = {}
            for state_id, state in self.quantum_states.items():
                # Skip states with no entanglements
                if not state.entanglement:
                    continue
                
                # Calculate energy transfer
                for target_id, strength in state.entanglement.items():
                    if target_id in self.quantum_states:
                        target = self.quantum_states[target_id]
                        
                        # Energy flows from higher to lower energy states
                        energy_diff = state.energy - target.energy
                        if abs(energy_diff) > 0.05:  # Only transfer significant energy differences
                            # Energy transfer is proportional to entanglement strength and difference
                            transfer = energy_diff * strength * self.energy_transfer_rate
                            
                            # Accumulate energy deltas
                            energy_deltas[state_id] = energy_deltas.get(state_id, 0) - transfer
                            energy_deltas[target_id] = energy_deltas.get(target_id, 0) + transfer
                            
                            changes += 1
            
            # Apply energy deltas
            for state_id, delta in energy_deltas.items():
                if state_id in self.quantum_states:
                    self.quantum_states[state_id].energy += delta
                    # Clamp energy to valid range
                    self.quantum_states[state_id].energy = min(1.0, max(0.1, self.quantum_states[state_id].energy))
            
            # 3. Update positions based on entanglements
            self._update_positions()
            
            # 4. Generate spontaneous insights with low probability
            if random.random() < self.spontaneous_insight_rate and len(self.quantum_states) >= 2:
                self._generate_spontaneous_insight()
                changes += 1
            
            # 5. Introduce random mutations occasionally
            current_time = time.time()
            if current_time - self.last_mutation_time > 10.0:  # Every 10 seconds
                mutations = self._introduce_mutations()
                changes += mutations
                self.last_mutation_time = current_time
            
            # Update metrics
            total_energy = sum(state.energy for state in self.quantum_states.values())
            if self.quantum_states:
                self.metrics["avg_energy"] = total_energy / len(self.quantum_states)
            
            # Calculate simulation speed
            step_duration = time.time() - step_start_time
            self.metrics["simulation_speed"] = 1.0 / max(step_duration, 0.001)  # Steps per second
            
            # Return simulation results
            return {
                "step": self.simulation_step,
                "changes": changes,
                "duration": step_duration,
                "metrics": self.metrics
            }
    
    def _update_positions(self) -> None:
        """
        Update positions of quantum states based on entanglements.
        
        This function implements a force-directed algorithm where
        entangled states attract each other proportionally to their
        entanglement strength.
        """
        # Parameters
        repulsion = 1.0  # Base repulsion force
        attraction = 5.0  # Base attraction multiplier
        damping = 0.8    # Movement damping factor
        min_distance = 0.5  # Minimum distance to prevent division by zero
        
        # Calculate forces for each state
        forces = {state_id: np.zeros(3) for state_id in self.quantum_states}
        
        # 1. Repulsive forces between all states
        states_list = list(self.quantum_states.items())
        for i, (id1, state1) in enumerate(states_list):
            pos1 = np.array(state1.position)
            
            for id2, state2 in states_list[i+1:]:
                pos2 = np.array(state2.position)
                
                # Calculate distance
                direction = pos2 - pos1
                distance = np.linalg.norm(direction)
                
                # Avoid division by zero
                if distance < min_distance:
                    distance = min_distance
                
                # Normalize direction
                if distance > 0:
                    direction = direction / distance
                
                # Repulsive force diminishes with square of distance
                force_magnitude = repulsion / (distance * distance)
                
                # Apply repulsive force
                force = direction * force_magnitude
                forces[id1] -= force
                forces[id2] += force
        
        # 2. Attractive forces through entanglements
        for id1, state1 in self.quantum_states.items():
            pos1 = np.array(state1.position)
            
            for id2, strength in state1.entanglement.items():
                if id2 in self.quantum_states:
                    state2 = self.quantum_states[id2]
                    pos2 = np.array(state2.position)
                    
                    # Calculate distance
                    direction = pos2 - pos1
                    distance = np.linalg.norm(direction)
                    
                    # Avoid division by zero
                    if distance < min_distance:
                        distance = min_distance
                    
                    # Normalize direction
                    if distance > 0:
                        direction = direction / distance
                    
                    # Attractive force proportional to distance and entanglement strength
                    force_magnitude = distance * strength * attraction
                    
                    # Apply attractive force
                    force = direction * force_magnitude
                    forces[id1] += force
        
        # 3. Apply forces to update positions
        for state_id, force in forces.items():
            state = self.quantum_states[state_id]
            
            # Apply force with damping based on coherence
            # Higher coherence = more responsive to forces
            effective_damping = damping * state.coherence
            
            # Limit maximum force to prevent extreme movements
            force_magnitude = np.linalg.norm(force)
            if force_magnitude > 5.0:
                force = force * (5.0 / force_magnitude)
            
            # Update position
            state.position[0] += force[0] * effective_damping
            state.position[1] += force[1] * effective_damping
            state.position[2] += force[2] * effective_damping
            
            # Update last_update timestamp
            state.last_update = time.time()
    
    def _generate_spontaneous_insight(self) -> Insight:
        """
        Generate a spontaneous insight from the current quantum field state.
        
        Returns:
            The generated insight
        """
        # Need at least 2 states for an interesting insight
        if len(self.quantum_states) < 2:
            return None
        
        # 1. Select random state as the focus
        focus_id = random.choice(list(self.quantum_states.keys()))
        focus_state = self.quantum_states[focus_id]
        
        # 2. Choose insight type
        insight_types = ["pattern", "correlation", "prediction", "anomaly"]
        weights = [0.4, 0.3, 0.2, 0.1]  # Pattern insights are most common
        insight_type = random.choices(insight_types, weights=weights, k=1)[0]
        
        # 3. Generate insight based on type
        if insight_type == "pattern":
            # Find states similar to focus state
            similar_states = []
            for state_id, state in self.quantum_states.items():
                if state_id != focus_id:
                    similarity = self._calculate_similarity(focus_state.vector, state.vector)
                    if similarity > 0.7:  # High similarity threshold
                        similar_states.append((state_id, similarity))
            
            # Only create insight if we found similar states
            if similar_states:
                source_ids = [focus_id] + [s[0] for s in similar_states]
                content = f"Detected a recurring pattern across {len(source_ids)} states, " \
                          f"suggesting an underlying structural similarity."
                confidence = 0.6 + 0.3 * (len(similar_states) / 10)  # More states = higher confidence
                
                insight = Insight(
                    type=insight_type,
                    source_nodes=source_ids,
                    content=content,
                    confidence=min(0.9, confidence)
                )
                
                self.insights.append(insight)
                return insight
        
        elif insight_type == "correlation":
            # Look for correlations in energy levels and entanglement
            high_entanglement = [(id2, strength) for id2, strength in focus_state.entanglement.items() 
                               if strength > 0.8 and id2 in self.quantum_states]
            
            if high_entanglement:
                source_ids = [focus_id] + [h[0] for h in high_entanglement]
                
                # Calculate average energy
                energies = [self.quantum_states[sid].energy for sid in source_ids]
                avg_energy = sum(energies) / len(energies)
                
                if avg_energy > 0.7:
                    content = f"Strong correlation between high energy states, possibly indicating " \
                              f"an emerging resonance pattern."
                else:
                    content = f"Detected correlation between entangled states with varying energy levels, " \
                              f"suggesting complex interaction patterns."
                
                insight = Insight(
                    type=insight_type,
                    source_nodes=source_ids,
                    content=content,
                    confidence=0.7
                )
                
                self.insights.append(insight)
                return insight
        
        elif insight_type == "prediction":
            # Make a prediction based on state evolution
            if len(focus_state.entanglement) > 0:
                # Predict energy transfer
                energy_receivers = []
                energy_donors = []
                
                for other_id, strength in focus_state.entanglement.items():
                    if other_id in self.quantum_states:
                        other_state = self.quantum_states[other_id]
                        if other_state.energy < focus_state.energy:
                            energy_receivers.append(other_id)
                        else:
                            energy_donors.append(other_id)
                
                if energy_receivers:
                    source_ids = [focus_id] + energy_receivers
                    content = f"Predicting energy transfer from high-energy state to " \
                              f"{len(energy_receivers)} connected lower-energy states."
                elif energy_donors:
                    source_ids = [focus_id] + energy_donors
                    content = f"Predicting energy absorption from {len(energy_donors)} " \
                              f"higher-energy connected states."
                else:
                    source_ids = [focus_id]
                    content = f"Predicting stable energy state maintenance with minimal fluctuations."
                
                insight = Insight(
                    type=insight_type,
                    source_nodes=source_ids,
                    content=content,
                    confidence=0.6
                )
                
                self.insights.append(insight)
                return insight
        
        elif insight_type == "anomaly":
            # Look for anomalous behavior
            avg_energy = sum(s.energy for s in self.quantum_states.values()) / len(self.quantum_states)
            avg_coherence = sum(s.coherence for s in self.quantum_states.values()) / len(self.quantum_states)
            
            # Check if focus state differs significantly from average
            energy_diff = abs(focus_state.energy - avg_energy)
            coherence_diff = abs(focus_state.coherence - avg_coherence)
            
            if energy_diff > 0.3 or coherence_diff > 0.3:
                content = f"Detected anomalous state behavior with " + \
                          (f"unusually high energy" if focus_state.energy > avg_energy else "unusually low energy") + \
                          " and " + \
                          (f"above-average coherence." if focus_state.coherence > avg_coherence else "below-average coherence.")
                
                insight = Insight(
                    type=insight_type,
                    source_nodes=[focus_id],
                    content=content,
                    confidence=0.5 + (energy_diff + coherence_diff)
                )
                
                self.insights.append(insight)
                return insight
        
        # If we get here, we couldn't generate an insight
        return None
    
    def _introduce_mutations(self) -> int:
        """
        Introduce random mutations in the quantum field.
        
        Returns:
            Number of mutations applied
        """
        if not self.quantum_states:
            return 0
        
        mutations = 0
        mutation_candidates = random.sample(
            list(self.quantum_states.keys()),
            min(int(len(self.quantum_states) * self.mutation_rate) + 1, len(self.quantum_states))
        )
        
        for state_id in mutation_candidates:
            state = self.quantum_states[state_id]
            
            # Choose mutation type
            mutation_type = random.choice(["vector", "energy", "coherence", "position", "entanglement"])
            
            if mutation_type == "vector":
                # Perturb the state vector slightly
                noise = np.random.normal(0, 0.1, self.dimension)
                state.vector = state.vector + noise
                # Renormalize
                norm = np.linalg.norm(state.vector)
                if norm > 0:
                    state.vector = state.vector / norm
                
                mutations += 1
                
            elif mutation_type == "energy":
                # Randomly increase or decrease energy
                change = random.uniform(-0.2, 0.2)
                state.energy = max(0.1, min(1.0, state.energy + change))
                mutations += 1
                
            elif mutation_type == "coherence":
                # Randomly increase or decrease coherence
                change = random.uniform(-0.1, 0.2)  # Bias toward increasing
                state.coherence = max(0.1, min(1.0, state.coherence + change))
                mutations += 1
                
            elif mutation_type == "position":
                # Apply random displacement
                displacement = [
                    random.uniform(-2.0, 2.0),
                    random.uniform(-2.0, 2.0),
                    random.uniform(-2.0, 2.0)
                ]
                for i in range(3):
                    state.position[i] += displacement[i]
                mutations += 1
                
            elif mutation_type == "entanglement":
                # Either create new entanglement or modify existing
                if random.random() < 0.7 and len(state.entanglement) > 0:
                    # Modify existing entanglement
                    target_id = random.choice(list(state.entanglement.keys()))
                    if target_id in self.quantum_states:
                        change = random.uniform(-0.2, 0.2)
                        new_strength = max(0.1, min(1.0, state.entanglement[target_id] + change))
                        state.entanglement[target_id] = new_strength
                        self.quantum_states[target_id].entanglement[state_id] = new_strength
                        mutations += 1
                else:
                    # Create new entanglement with random state
                    other_ids = [id for id in self.quantum_states if id != state_id and id not in state.entanglement]
                    if other_ids:
                        target_id = random.choice(other_ids)
                        strength = random.uniform(0.3, 0.8)
                        state.entanglement[target_id] = strength
                        self.quantum_states[target_id].entanglement[state_id] = strength
                        mutations += 1
        
        return mutations
    
    def get_visualization_data(self) -> Dict[str, Any]:
        """
        Get data formatted for visualization.
        
        Returns:
            Dictionary with nodes and connections for visualization
        """
        nodes = []
        connections = []
        
        # Process nodes
        for state_id, state in self.quantum_states.items():
            nodes.append({
                "id": state_id,
                "position": state.position,
                "energy": state.energy,
                "coherence": state.coherence,
                "connections": len(state.entanglement),
                "metadata": state.metadata
            })
        
        # Process connections (avoid duplicates)
        processed_pairs = set()
        
        for state_id, state in self.quantum_states.items():
            for target_id, strength in state.entanglement.items():
                # Create a canonical connection ID (ordered by node IDs)
                conn_id = tuple(sorted([state_id, target_id]))
                if conn_id in processed_pairs:
                    continue
                
                # Only include connection if other state exists
                if target_id in self.quantum_states:
                    connections.append({
                        "source": state_id,
                        "target": target_id,
                        "strength": strength
                    })
                    processed_pairs.add(conn_id)
        
        return {
            "nodes": nodes,
            "connections": connections,
            "metrics": self.metrics,
            "simulation_step": self.simulation_step,
            "timestamp": time.time()
        }
    
    def get_insights(self, limit: int = 10, state_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get insights, optionally filtered by state.
        
        Args:
            limit: Maximum number of insights to return
            state_id: Optional state ID to filter insights by
            
        Returns:
            List of insight dictionaries
        """
        # Filter insights if state_id provided
        if state_id:
            filtered_insights = [
                insight for insight in self.insights
                if state_id in insight.source_nodes
            ]
        else:
            filtered_insights = self.insights
        
        # Sort by creation time (newest first) and limit
        sorted_insights = sorted(filtered_insights, key=lambda x: x.creation_time, reverse=True)
        limited_insights = sorted_insights[:limit]
        
        # Convert to dictionaries
        return [insight.to_dict() for insight in limited_insights]


#=====================================================================
# Visualization Engine
#=====================================================================

class VisualizationEngine:
    """
    Advanced visualization engine for the quantum field simulator.
    
    This component handles rendering the quantum field state for
    visualization, providing different rendering modes and levels
    of detail.
    """
    
    def __init__(self, field_simulator: QuantumFieldSimulator, settings: Optional[VisualizationSettings] = None):
        """
        Initialize the visualization engine.
        
        Args:
            field_simulator: Quantum field simulator to visualize
            settings: Visualization settings
        """
        self.simulator = field_simulator
        self.settings = settings or VisualizationSettings()
        
        # Visualization state
        self.node_positions = {}  # node_id -> position
        self.node_colors = {}  # node_id -> color
        self.node_sizes = {}  # node_id -> size
        self.connection_opacities = {}  # (node1_id, node2_id) -> opacity
        
        # Performance monitoring
        self.render_time = 0
        self.last_update = 0
        self.frame_count = 0
        self.fps = 0
        
        logger.info("Visualization Engine initialized")
    
    def update_settings(self, settings: Dict[str, Any]) -> None:
        """Update visualization settings from a dictionary."""
        for key, value in settings.items():
            if hasattr(self.settings, key):
                setattr(self.settings, key, value)
        
        logger.info("Visualization settings updated")
    
    def prepare_frame_data(self) -> Dict[str, Any]:
        """
        Prepare data for rendering a visualization frame.
        
        Returns:
            Dictionary with visualization data
        """
        start_time = time.time()
        
        # Get raw data from simulator
        sim_data = self.simulator.get_visualization_data()
        
        # Apply visualization settings
        
        # 1. Limit number of nodes if necessary
        nodes = sim_data["nodes"]
        if len(nodes) > self.settings.max_visible_nodes:
            # Sort by importance (energy + connections)
            nodes.sort(key=lambda n: n["energy"] + (n["connections"] * 0.1), reverse=True)
            nodes = nodes[:self.settings.max_visible_nodes]
        
        # 2. Process nodes for visualization
        node_data = []
        for node in nodes:
            # Calculate node size
            size = 0.5 + (node["energy"] * self.settings.node_size_scale)
            
            # Determine node color based on color mode
            if self.settings.color_mode == "energy":
                # Color based on energy: blue (low) to red (high)
                color = self._energy_to_color(node["energy"])
            elif self.settings.color_mode == "coherence":
                # Color based on coherence: green (high) to purple (low)
                color = self._coherence_to_color(node["coherence"])
            elif self.settings.color_mode == "entanglement":
                # Color based on number of connections: blue (few) to orange (many)
                connection_ratio = min(1.0, node["connections"] / 10)
                color = self._connection_to_color(connection_ratio)
            else:
                # Default color
                color = self.settings.node_color_palette["default"]
            
            # Store position for animation
            self.node_positions[node["id"]] = node["position"]
            self.node_colors[node["id"]] = color
            self.node_sizes[node["id"]] = size
            
            # Add to node data
            node_data.append({
                "id": node["id"],
                "position": node["position"],
                "color": color,
                "size": size,
                "energy": node["energy"],
                "coherence": node.get("coherence", 0.5),
                "connections": node["connections"],
                "metadata": node.get("metadata", {})
            })
        
        # 3. Process connections
        connection_data = []
        connections = sim_data["connections"]
        
        # Limit connections if necessary
        if len(connections) > self.settings.max_visible_connections:
            # Sort by strength
            connections.sort(key=lambda c: c["strength"], reverse=True)
            connections = connections[:self.settings.max_visible_connections]
        
        for conn in connections:
            # Only include connection if both nodes are visible
            source_id = conn["source"]
            target_id = conn["target"]
            
            if source_id in self.node_positions and target_id in self.node_positions:
                # Calculate opacity based on strength
                opacity = conn["strength"] * self.settings.connection_scale
                
                # Store opacity for animation
                conn_key = tuple(sorted([source_id, target_id]))
                self.connection_opacities[conn_key] = opacity
                
                connection_data.append({
                    "source": source_id,
                    "target": target_id,
                    "strength": conn["strength"],
                    "opacity": opacity,
                    "color": "#8888ff"  # Default connection color
                })
        
        # 4. Add rendering metadata
        metadata = {
            "settings": {
                "node_size_scale": self.settings.node_size_scale,
                "connection_scale": self.settings.connection_scale,
                "color_mode": self.settings.color_mode,
                "glow_intensity": self.settings.glow_intensity,
                "background_color": self.settings.background_color,
                "show_labels": self.settings.show_labels,
                "auto_rotate": self.settings.auto_rotate,
                "render_quality": self.settings.render_quality
            },
            "metrics": sim_data["metrics"],
            "simulation_step": sim_data["simulation_step"]
        }
        
        # Update performance metrics
        self.render_time = time.time() - start_time
        self.frame_count += 1
        
        # Calculate FPS every second
        current_time = time.time()
        if current_time - self.last_update >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_update)
            self.frame_count = 0
            self.last_update = current_time
        
        metadata["performance"] = {
            "render_time": self.render_time,
            "fps": self.fps
        }
        
        return {
            "nodes": node_data,
            "connections": connection_data,
            "metadata": metadata,
            "timestamp": time.time()
        }
    
    def _energy_to_color(self, energy: float) -> str:
        """Convert energy value to color."""
        # Red (high energy) to Blue (low energy)
        r = int(255 * energy)
        g = int(50 + 50 * energy)
        b = int(255 * (1 - energy))
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def _coherence_to_color(self, coherence: float) -> str:
        """Convert coherence value to color."""
        # Green (high coherence) to Purple (low coherence)
        r = int(100 + 155 * (1 - coherence))
        g = int(100 + 155 * coherence)
        b = int(200)
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def _connection_to_color(self, connection_ratio: float) -> str:
        """Convert connection ratio to color."""
        # Blue (few connections) to Orange (many connections)
        r = int(50 + 205 * connection_ratio)
        g = int(100 + 55 * connection_ratio)
        b = int(200 - 150 * connection_ratio)
        return f"#{r:02x}{g:02x}{b:02x}"


#=====================================================================
# Web Server
#=====================================================================

from flask import Flask, request, jsonify, render_template, send_from_directory

class WebServer:
    """
    Web server for the quantum kaleidoscope system.
    
    Provides a web interface for interacting with the system,
    including visualization, data input, and insight retrieval.
    """
    
    def __init__(self, 
                simulator: QuantumFieldSimulator,
                visualizer: VisualizationEngine,
                host: str = "0.0.0.0",
                port: int = 8000,
                static_dir: str = "./static",
                template_dir: str = "./templates"):
        """
        Initialize the web server.
        
        Args:
            simulator: Quantum field simulator
            visualizer: Visualization engine
            host: Server host
            port: Server port
            static_dir: Directory for static files
            template_dir: Directory for templates
        """
        self.simulator = simulator
        self.visualizer = visualizer
        self.host = host
        self.port = port
        
        # Create required directories
        os.makedirs(static_dir, exist_ok=True)
        os.makedirs(template_dir, exist_ok=True)
        
        # Create Flask app
        self.app = Flask(__name__, 
                        static_folder=static_dir, 
                        template_folder=template_dir)
        
        # Configure routes
        self._setup_routes()
        
        # Background simulation task
        self.simulation_thread = None
        self.simulation_active = False
        
        logger.info(f"Web server initialized on {host}:{port}")
    
    def _setup_routes(self) -> None:
        """Set up the web server routes."""
        
        @self.app.route('/')
        def index():
            """Render the main page."""
            return render_template('index.html')
        
        @self.app.route('/visualization')
        def visualization():
            """Render the visualization page."""
            return render_template('visualization.html')
        
        @self.app.route('/api/status')
        def get_status():
            """Get system status."""
            return jsonify({
                "status": "active",
                "node_count": len(self.simulator.quantum_states),
                "insight_count": len(self.simulator.insights),
                "simulation_step": self.simulator.simulation_step,
                "uptime": time.time() - self.simulator.start_time,
                "uptime_formatted": self._format_time(time.time() - self.simulator.start_time),
                "simulation_active": self.simulation_active,
                "metrics": self.simulator.metrics
            })
        
        @self.app.route('/api/visualization')
        def get_visualization():
            """Get visualization data."""
            return jsonify(self.visualizer.prepare_frame_data())
        
        @self.app.route('/api/settings', methods=['GET', 'POST'])
        def visualization_settings():
            """Get or update visualization settings."""
            if request.method == 'POST':
                settings = request.json
                self.visualizer.update_settings(settings)
                return jsonify({"status": "success"})
            else:
                return jsonify(vars(self.visualizer.settings))
        
        @self.app.route('/api/process/text', methods=['POST'])
        def process_text():
            """Process text input."""
            data = request.json
            if not data or 'text' not in data:
                return jsonify({"error": "Missing text in request"}), 400
            
            result = self.simulator.process_text(
                data['text'],
                data.get('metadata', {})
            )
            
            return jsonify(result)
        
        @self.app.route('/api/insights')
        def get_insights():
            """Get insights."""
            limit = request.args.get('limit', 10, type=int)
            state_id = request.args.get('state_id')
            
            insights = self.simulator.get_insights(limit=limit, state_id=state_id)
            return jsonify({"insights": insights})
        
        @self.app.route('/api/state/<state_id>')
        def get_state(state_id):
            """Get details for a specific quantum state."""
            if state_id in self.simulator.quantum_states:
                state = self.simulator.quantum_states[state_id]
                return jsonify({
                    "id": state_id,
                    "energy": state.energy,
                    "coherence": state.coherence,
                    "position": state.position,
                    "entanglement": {k: float(v) for k, v in state.entanglement.items()},
                    "metadata": state.metadata,
                    "creation_time": state.creation_time
                })
            else:
                return jsonify({"error": "State not found"}), 404
        
        @self.app.route('/api/simulation/start', methods=['POST'])
        def start_simulation():
            """Start the background simulation."""
            if not self.simulation_active:
                self.simulation_active = True
                self.simulation_thread = threading.Thread(
                    target=self._simulation_loop,
                    daemon=True
                )
                self.simulation_thread.start()
                return jsonify({"status": "simulation_started"})
            else:
                return jsonify({"status": "already_running"})
        
        @self.app.route('/api/simulation/stop', methods=['POST'])
        def stop_simulation():
            """Stop the background simulation."""
            if self.simulation_active:
                self.simulation_active = False
                if self.simulation_thread:
                    self.simulation_thread.join(timeout=2.0)
                return jsonify({"status": "simulation_stopped"})
            else:
                return jsonify({"status": "not_running"})
    
    def _simulation_loop(self) -> None:
        """Background thread for running the simulation."""
        logger.info("Starting simulation loop")
        
        while self.simulation_active:
            try:
                # Run a simulation step
                self.simulator.run_simulation_step()
                
                # Sleep to control simulation speed
                time.sleep(0.1)  # 10 steps per second maximum
                
            except Exception as e:
                logger.error(f"Error in simulation loop: {e}")
                traceback.print_exc()
                time.sleep(1.0)
    
    def _format_time(self, seconds: float) -> str:
        """Format time in seconds to HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def start(self) -> None:
        """Start the web server."""
        # Create default templates and static files if needed
        self._ensure_templates_exist()
        
        # Start simulation
        self.simulation_active = True
        self.simulation_thread = threading.Thread(
            target=self._simulation_loop,
            daemon=True
        )
        self.simulation_thread.start()
        
        # Run Flask app
        logger.info(f"Starting web server on {self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=False, threaded=True)
    
    def _ensure_templates_exist(self) -> None:
        """Ensure that template and static files exist."""
        # Create basic