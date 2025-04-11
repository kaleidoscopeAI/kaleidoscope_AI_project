import uuid
import time
import ctypes
import json
import numpy as np
import networkx as nx
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field
import matplotlib.pyplot as plt  #For visualization
from datetime import datetime
from kaleidoscope.data_processing.standardized_data import StandardizedData  # Assuming this is available
from kaleidoscope.common.chatbot import ChatBot  # Assuming this is available


class Cube:  # Copied directly from your provided code
    def __init__(self, grid_size: int = 100):
        """
        Initializes the cube with six inner faces. Each face is populated
        with a grid of points. The cube is defined in a 3D coordinate system:
          - Front face: z = 1
          - Back face: z = grid_size
          - Left face: x = 1
          - Right face: x = grid_size
          - Bottom face: y = 1
          - Top face: y = grid_size
        """
        self.grid_size = grid_size
        self.points = {}  # Keys: (face, x, y), Values: 3D coordinate tuple
        self.edges = []   # List of tuples: (point_id, opposite_point_id)
        self._populate_faces()
        self._connect_opposite_faces()

    def _populate_faces(self):
        """Populate each of the six inner faces with a grid of points."""
        faces = ["front", "back", "left", "right", "top", "bottom"]
        for face in faces:
            for x in range(1, self.grid_size + 1):
                for y in range(1, self.grid_size + 1):
                    point_id = f"{face}_{x}_{y}"
                    coord = self._get_point_coordinate(face, x, y)
                    self.points[point_id] = coord
        # No logging in the combined version
        # logging.info(f"Populated {len(self.points)} points across {len(faces)} faces.")


    def _get_point_coordinate(self, face: str, x: int, y: int) -> Tuple[int, int, int]:
        """
        Returns the 3D coordinate for a point on a given face.
          - Front: z = 1
          - Back: z = grid_size
          - Left: x = 1
          - Right: x = grid_size
          - Top: y = grid_size
          - Bottom: y = 1
        The other two coordinates are given by (x, y) in the face's local 2D grid.
        """
        if face == "front":
            return (x, y, 1)
        elif face == "back":
            return (x, y, self.grid_size)
        elif face == "left":
            return (1, x, y)  # Using x and y from the grid for the other axes
        elif face == "right":
            return (self.grid_size, x, y)
        elif face == "top":
            return (x, self.grid_size, y)
        elif face == "bottom":
            return (x, 1, y)
        else:
            raise ValueError("Invalid face name")

    def _connect_opposite_faces(self):
        """
        For each point on one face, create an edge (string) to its corresponding
        point on the opposite face. The mapping is done based on the grid coordinates.
        For front/back, the (x, y) pair is directly used.
        For left/right, we use (x, y) in the face's coordinate system.
        For top/bottom, we use (x, y) in the face's coordinate system.
        """
        opposite = {
            "front": "back",
            "back": "front",
            "left": "right",
            "right": "left",
            "top": "bottom",
            "bottom": "top"
        }
        for face, opp_face in opposite.items():
            # To avoid duplicating edges, process only one direction.
            if face in ["front", "left", "top"]:
                for x in range(1, self.grid_size + 1):
                    for y in range(1, self.grid_size + 1):
                        point_id = f"{face}_{x}_{y}"
                        opp_point_id = f"{opp_face}_{x}_{y}"
                        if point_id in self.points and opp_point_id in self.points:
                            self.edges.append((point_id, opp_point_id))
        # No logging in combined version.
        # logging.info(f"Created {len(self.edges)} edges connecting opposite faces.")


    def get_cube_structure(self) -> Dict[str, Any]:
        """Return the cube's structure as a dictionary with points and edges."""
        return {"points": self.points, "edges": self.edges}



@dataclass
class UnifiedInterfaceNode:
    """
    A unified node that combines data storage, chatbot interaction,
    and supernode capabilities.  Uses a data cube for internal representation.
    """
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    cube: Cube = field(default_factory=Cube)  # Use the Cube class
    chatbot: ChatBot = field(default_factory=ChatBot)
    energy: float = 100.0
    traits: Dict[str, float] = field(default_factory=dict)
    stress_level: float = 0.0
    emotional_state: str = "Calm"
    recent_performance: List[float] = field(default_factory=list) #for performance tracking
    task_load: int = 0
    last_heartbeat: float = field(default_factory=time.time)
    tasks : List[Dict] = field(default_factory=list)

    def __post_init__(self):
        if not self.node_id:
            self.node_id = str(uuid.uuid4())
        self.memory_threshold: float = 5.0

    def ingest_data(self, data: Dict, source: str):
        """
        Ingests data and stores it in the data cube.

        Args:
            data: The data to ingest.  Should be a dictionary.  Keys are used to
                  determine the location on the cube.
            source: The source of the data.
        """

        # Very basic example:  Map keys to x, y, z, layer
        x = data.get('x')
        y = data.get('y')
        z = data.get('z')
        layer = data.get('layer', 1)  # Default to layer 1

        if x is None or y is None or z is None:
            print(f"Warning: Data entry missing x, y, or z coordinates. Skipping. Data: {data}")
            return

        if not (1 <= x <= self.cube.grid_size and 1 <= y <= self.cube.grid_size and (z == 1 or z == self.cube.grid_size)):
             print(f"Warning: coordinates {(x,y,z)} out of bounds")
             return

        # Determine the face based on x, y, z, and layer
        if z == 1:
            face = "front"
        elif z == self.cube.grid_size:
            face = "back"
        elif x == 1:
            face = "left"
        elif x == self.cube.grid_size:
            face = "right"
        elif y == 1:
            face = "bottom"
        elif y == self.cube.grid_size:
            face = "top"
        else:
            print(f"Warning: invalid coordinates {(x,y,z, layer)}")
            return
        
        point_id = f"{face}_{x}_{y}"

        # Store the data at the appropriate node
        if point_id in self.cube.points:
             self.cube.points[point_id]['data'] = data # Store ALL data at the point.
        else:
            print(f"Warning: point_id {point_id} not found")

    def get_data_at_point(self, x: int, y: int, z: int, layer: int) -> Optional[Dict]:
        """Retrieves data from a specific point on the cube."""
        if not (1 <= x <= self.cube.grid_size and 1 <= y <= self.cube.grid_size and (z == 1 or z == self.cube.grid_size) and 1 <= layer <= 2):
            print(f"Warning: Coordinates {(x, y, z, layer)} are out of bounds.")
            return None

        if z == 1:
            face = 'front'
        elif z == self.cube.grid_size:
            face = 'back'
        elif x == 1:
            face = 'left'
        elif x == self.cube.grid_size:
            face = 'right'
        elif y == 1:
            face = 'bottom'
        else:
            face = 'top'

        point_id = f"{face}_{x}_{y}"
        if point_id in self.cube.points:
            return self.cube.points[point_id].get('data')
        return None
    
    def get_response(self, query: str) -> str:
        """
        Provides a response to a user query via the integrated chatbot.

        Args:
            query: The user's query string.

        Returns:
            A response string from the chatbot.
        """
        return self.chatbot.get_response(query)
    
    def calculate_stress(self):
        """
        Calculate the stress level based on task load, energy levels, and recent performance.
        """
        task_factor = self.task_load / 10.0  # Normalize task load to a factor out of 10
        energy_factor = (10.0 - self.energy) / 10.0  # Invert energy level to represent stress
        performance_factor = 1.0 - np.mean(self.recent_performance) if self.recent_performance else 1.0  # Use 1.0 as default if no performance data

        # Calculate stress level, ensuring it is clipped between 0 and 1
        self.stress_level = np.clip(
            task_factor * 0.4 + energy_factor * 0.4 + performance_factor * 0.2,
            0.0,
            1.0
        )

    def update_emotional_state(self):
        """
        Update the emotional state of the node based on its current stress level.
        """
        if self.stress_level < 0.3:
            self.emotional_state = "Calm"
        elif self.stress_level < 0.6:
            self.emotional_state = "Alert"
        elif self.stress_level < 0.8:
            self.emotional_state = "Anxious"
        else:
            self.emotional_state = "Overwhelmed"

    def get_state(self) -> dict:
        """Retrieves the current state of the node."""
        state = {
            'node_id': self.node_id,
            'energy': self.energy,
            'stress_level': self.stress_level,
            'emotional_state': self.emotional_state,
            'traits': self.traits,
            'cube_data': self.cube.get_cube_structure()  # Include the cube's data
        }
        return state
