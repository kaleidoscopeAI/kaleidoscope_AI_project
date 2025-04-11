# modules/SeedManager.py
import logging
import random
import uuid
import time
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

class SeedManager:
    """
    Manages the creation and evolution of node seeds.
    Seeds are templates used to create new nodes with specific characteristics.
    """
    
    def __init__(self):
        self.seeds = {}  # Store seed templates
        self.active_seeds = set()  # Currently active seeds
        self.seed_history = []  # Track seed evolution
        logger.info("SeedManager initialized")
    
    def create_seed(self, seed_type: str, properties: Dict[str, Any]) -> str:
        """
        Creates a new seed template.
        
        Args:
            seed_type: Type of seed (e.g., "text", "visual", "capability")
            properties: Properties to initialize the seed with
            
        Returns:
            Seed ID
        """
        seed_id = f"seed_{seed_type}_{str(uuid.uuid4())[:8]}"
        
        seed = {
            "id": seed_id,
            "type": seed_type,
            "properties": properties.copy(),
            "created_at": time.time(),
            "activation_count": 0,
            "success_rate": 0.0,
            "active": False
        }
        
        self.seeds[seed_id] = seed
        logger.info(f"Created {seed_type} seed: {seed_id}")
        return seed_id
    
    def activate_seed(self, seed_id: str) -> bool:
        """Activates a seed for use in node creation."""
        if seed_id not in self.seeds:
            logger.warning(f"Cannot activate: Seed {seed_id} not found")
            return False
        
        if seed_id not in self.active_seeds:
            self.active_seeds.add(seed_id)
            self.seeds[seed_id]["active"] = True
            logger.info(f"Activated seed: {seed_id}")
        return True
    
    def deactivate_seed(self, seed_id: str) -> bool:
        """Deactivates a seed."""
        if seed_id not in self.active_seeds:
            logger.warning(f"Seed {seed_id} is not active")
            return False
        
        self.active_seeds.remove(seed_id)
        self.seeds[seed_id]["active"] = False
        logger.info(f"Deactivated seed: {seed_id}")
        return True
    
    def get_active_seeds(self, seed_type: Optional[str] = None) -> List[str]:
        """Gets IDs of all active seeds, optionally filtered by type."""
        if seed_type:
            return [
                seed_id for seed_id in self.active_seeds 
                if self.seeds[seed_id]["type"] == seed_type
            ]
        return list(self.active_seeds)
    
    def get_seed_properties(self, seed_id: str) -> Optional[Dict[str, Any]]:
        """Gets the properties of a specified seed."""
        if seed_id not in self.seeds:
            logger.warning(f"Seed {seed_id} not found")
            return None
        
        return self.seeds[seed_id]["properties"].copy()
    
    def evolve_seed(self, parent_seed_id: str, mutation_rate: float = 0.1) -> Optional[str]:
        """
        Creates a new seed by evolving an existing one.
        
        Args:
            parent_seed_id: ID of the seed to evolve
            mutation_rate: Rate of property mutation (0.0-1.0)
            
        Returns:
            New seed ID if successful, None otherwise
        """
        if parent_seed_id not in self.seeds:
            logger.warning(f"Cannot evolve: Seed {parent_seed_id} not found")
            return None
        
        parent = self.seeds[parent_seed_id]
        new_properties = parent["properties"].copy()
        
        # Mutate properties
        for key, value in new_properties.items():
            if isinstance(value, (int, float)) and random.random() < mutation_rate:
                # Numeric mutation
                if isinstance(value, int):
                    new_properties[key] = value + random.choice([-1, 1])
                else:  # float
                    new_properties[key] = value * random.uniform(0.9, 1.1)
            elif isinstance(value, str) and random.random() < mutation_rate:
                # String mutation - simple example
                if len(value) > 0:
                    chars = list(value)
                    pos = random.randint(0, len(chars) - 1)
                    chars[pos] = chr(ord(chars[pos]) + random.choice([-1, 1]))
                    new_properties[key] = ''.join(chars)
            elif isinstance(value, bool) and random.random() < mutation_rate:
                # Boolean mutation
                new_properties[key] = not value
        
        # Create the evolved seed
        new_seed_id = self.create_seed(parent["type"], new_properties)
        
        # Record the evolution
        self.seed_history.append({
            "parent": parent_seed_id,
            "child": new_seed_id,
            "timestamp": time.time(),
            "mutation_rate": mutation_rate
        })
        
        logger.info(f"Evolved seed {parent_seed_id} into {new_seed_id}")
        return new_seed_id
    
    def record_seed_usage(self, seed_id: str, success: bool = True) -> None:
        """Records usage of a seed for performance tracking."""
        if seed_id not in self.seeds:
            logger.warning(f"Cannot record usage: Seed {seed_id} not found")
            return
        
        seed = self.seeds[seed_id]
        seed["activation_count"] += 1
        
        # Update success rate with exponential moving average
        if seed["activation_count"] == 1:
            seed["success_rate"] = 1.0 if success else 0.0
        else:
            alpha = 0.1  # Weighing factor for new data
            seed["success_rate"] = (1 - alpha) * seed["success_rate"] + alpha * (1.0 if success else 0.0)
        
        logger.debug(f"Recorded seed usage: {seed_id}, Success: {success}, " 
                    f"New rate: {seed['success_rate']:.2f}, Count: {seed['activation_count']}")
