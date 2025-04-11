# core/AI_Core.py
import logging
import time
import random
import numpy as np
from .NodeManager import NodeManager # Use absolute import from core
from .laws import GrowthLaws # Use absolute import from core
from .PerspectiveManager import PerspectiveManager # Use absolute import from modules
from .SeedManager import SeedManager # Use absolute import from modules
from kaleidoscope_ai.modules import PatternRecognition # Use absolute import from modules
from kaleidoscope_ai.llm import GPTProcessor # Import the new GPT processor
from kaleidoscope_ai.nodes import TextNode # Import updated TextNode
# Import VisualNode and potentially other node types if they exist
# from nodes.VisualNode import VisualNode

logger = logging.getLogger(__name__)

class AI_Core:
    """
    Main execution loop and system orchestrator for the Unified AI Core,
    now integrating a local GPT processor.
    """
    def __init__(self, initial_nodes=3, max_nodes=50):
        logger.info("Initializing AI Core...")
        self.node_manager = NodeManager()
        self.growth_laws = GrowthLaws()
        self.perspective_manager = PerspectiveManager()
        self.seed_manager = SeedManager()
        self.mirrored_network = MirroredNetwork()
        self.memory_graph = MemoryGraph()
        self.pattern_recognizer = PatternRecognition()
        # Initialize the local GPT processor
        try:
            # Use a smaller model by default for easier setup, adjust as needed
            self.gpt_processor = GPTProcessor(model_name="EleutherAI/gpt-neo-125M")
            if not self.gpt_processor.is_ready():
                 raise RuntimeError("GPT Processor failed to load model.")
        except Exception as e:
             logger.error(f"Critical error initializing GPT Processor: {e}. AI Core cannot start.", exc_info=True)
             raise # Prevent starting if GPT fails

        self.max_nodes = max_nodes
        self.running = False

        # Initialize perspectives (example)
        self.perspective_manager.add_perspective("Reverse", lambda data: data[::-1] if isinstance(data, str) else data)

        # Initialize base nodes
        self._initialize_base_nodes(initial_nodes)
        logger.info("AI Core initialized successfully.")

    def _initialize_base_nodes(self, count):
        """Initializes starting nodes, including TextNode with GPT."""
        logger.info(f"Initializing {count} base nodes...")
        core_laws = CoreLaws() # Assuming a default set of laws
        for i in range(count):
            # Create different types of nodes, ensuring TextNode gets the GPT processor
            node_type_choice = random.choice(["text", "visual", "generic"]) # Example types
            node_id = f"NodeInit_{i}"

            if node_type_choice == "text":
                node = TextNode(core_laws=core_laws, gpt_processor=self.gpt_processor, node_id=node_id)
            # Add elif blocks for VisualNode, etc., if they exist and need specific initializers
            # elif node_type_choice == "visual":
            #    node = VisualNode(...)
            else:
                # Fallback or generic node type placeholder
                # You might need a GenericNode class or use BaseNode if applicable
                from kaleidoscope_ai.nodes import BaseNode # Import BaseNode if needed for generic nodes
                node = BaseNode(node_type="generic", core_laws=core_laws, node_id=node_id)
                # Assign placeholder capabilities or state if BaseNode is used directly
                node.state = node.CoreState() if not hasattr(node, 'state') else node.state # Ensure state exists

            self.node_manager.register_node(node)
        logger.info(f"Base nodes initialized.")


    def start(self, execution_cycles=10, interval=1):
        """Starts the main execution loop."""
        logger.info("Starting AI Core execution loop...")
        self.running = True
        for cycle in range(execution_cycles):
            if not self.running:
                logger.info("AI Core stopping.")
                break
            logger.info(f"--- Cycle {cycle + 1}/{execution_cycles} ---")
            start_time = time.time()

            # 1. Simulate receiving data (replace with real data source)
            input_data = self._get_simulated_input()

            # 2. Process data through perspectives
            perspective_results = self.perspective_manager.process_perspectives(input_data.get("text", "")) # Example: text only

            # 3. Distribute tasks to nodes
            node_results = self._distribute_and_process(input_data)

            # 4. Apply Growth Laws (Energy update, Replication check)
            self.growth_laws.apply(self.node_manager.nodes.values()) # Pass node objects

            # 5. Replication based on Growth Laws decision
            self._handle_replication()

            # 6. Mirrored Network Sync
            self.mirrored_network.synchronize(self.node_manager.nodes) # Needs node data

            # 7. Update Memory Graph
            self._update_memory_graph(node_results)

            # 8. Pattern Recognition
            combined_data = self._combine_results(node_results, perspective_results)
            recognized_patterns = self.pattern_recognizer.recognize_patterns(combined_data)

            # 9. Logging and Status Update
            cycle_duration = time.time() - start_time
            self._log_cycle_status(cycle, cycle_duration, recognized_patterns)

            time.sleep(interval)

        self.running = False
        self.memory_graph.save_memory() # Final save
        logger.info("AI Core execution loop finished.")

    def stop(self):
        """Stops the execution loop."""
        logger.info("Received stop signal.")
        self.running = False

    def _get_simulated_input(self):
        """Generates sample input data for simulation."""
        # In a real system, this would come from data ingestion pipelines
        text_input = random.choice([
            "The quick brown fox jumps over the lazy dog.",
            "Photosynthesis is the process used by plants to convert light energy.",
            "Quantum computing utilizes qubits for complex calculations.",
            "Machine learning models improve with more data."
        ])
        # Add image data simulation if VisualNode is implemented
        # image_data = np.random.rand(100, 100, 3) # Example image data
        return {"text": text_input} # , "image": image_data}

    def _distribute_and_process(self, input_data):
        """Distributes data to relevant nodes and collects results."""
        results = {}
        for node_id, node in self.node_manager.nodes.items():
            # Basic routing based on node type (can be more sophisticated)
            if isinstance(node, TextNode) and "text" in input_data:
                result = node.process_text(input_data["text"])
                results[node_id] = result
            # Add logic for VisualNode if implemented
            # elif isinstance(node, VisualNode) and "image" in input_data:
            #     result = node.process_image(input_data["image"])
            #     results[node_id] = result
            # Add processing for other node types if needed
        return results

    def _handle_replication(self):
        """Checks nodes for replication conditions and performs replication."""
        nodes_to_add = []
        nodes_to_update_energy = {} # Store energy changes

        for node_id, node in list(self.node_manager.nodes.items()): # Iterate over copy
            if hasattr(node, 'can_replicate') and node.can_replicate(): # Check if method exists
                new_node_dna = node.replicate_dna() # Assuming DNA replication logic
                if new_node_dna:
                    # Use SeedManager to initialize the new node based on DNA
                    new_node_id = self.seed_manager.initialize_from_seed(
                        seed_id = f"seed_{node_id}_{time.time()}", # Generate a unique seed ID
                        parent_dna=new_node_dna,
                        parent_id=node_id
                    )
                    # Create the actual node instance (assuming SeedManager returns ID/config)
                    # This part needs refinement based on how SeedManager provides node info
                    # For now, let's assume it returns an ID and we create a default node type
                    if new_node_id:
                         # Determine node type based on parent or DNA, simplified here
                        NewNodeClass = type(node) # Replicate the same type
                        new_node_instance = NewNodeClass(
                            core_laws=node.core_laws.mutate(), # Mutate laws
                            gpt_processor=self.gpt_processor if isinstance(node, TextNode) else None, # Pass GPT if TextNode
                            node_id=new_node_id
                        )
                        nodes_to_add.append(new_node_instance)
                        # Manage energy transfer
                        transfer_energy = node.state.energy * 0.4 # Example energy split
                        nodes_to_update_energy[node_id] = -transfer_energy
                        nodes_to_update_energy[new_node_id] = transfer_energy # Initial energy for child
                        logger.info(f"Node {node_id} replicated to {new_node_id}")

        # Add new nodes and update energy outside the loop to avoid modifying dict during iteration
        for new_node in nodes_to_add:
            self.node_manager.register_node(new_node)
            if new_node.id in nodes_to_update_energy: # Assign initial energy
                 new_node.state.energy = nodes_to_update_energy[new_node.id]

        for node_id, energy_change in nodes_to_update_energy.items():
             if node_id in self.node_manager.nodes and node_id not in [n.id for n in nodes_to_add]: # Update parent energy
                  self.node_manager.nodes[node_id].state.energy += energy_change


    def _update_memory_graph(self, node_results):
        """Updates the central memory graph with relationships."""
        # Example: Add relationship based on shared entities or high confidence
        for node_id, result in node_results.items():
            if result and result.get("status") == "success" and "entities" in result:
                for entity, label in result["entities"]:
                    # Add node for entity if not exists
                    if not self.memory_graph.graph.has_node(entity):
                        self.memory_graph.add_node(entity, type=label)
                    # Add relationship between processing node and entity
                    self.memory_graph.add_relationship(node_id, entity, "processed")

    def _combine_results(self, node_results, perspective_results):
         """Combines results from nodes and perspectives for pattern recognition."""
         # Simple combination for now, can be more complex
         combined = {"nodes": node_results, "perspectives": perspective_results}
         # Flatten or structure data appropriately for PatternRecognition module
         # Example: extract all text summaries into one list
         all_summaries = []
         for res in node_results.values():
              if res and res.get("summary"):
                   all_summaries.append(res["summary"])
         combined["flat_text"] = " ".join(all_summaries)
         return combined


    def _log_cycle_status(self, cycle, duration, patterns):
        """Logs the status of the system after each cycle."""
        num_nodes = self.node_manager.get_node_count()
        status = self.node_manager.get_all_statuses()
        avg_energy = np.mean([s.get('energy', 0) for s in status.values()]) if status else 0
        logger.info(f"Cycle {cycle + 1} completed in {duration:.4f}s.")
        logger.info(f"Nodes: {num_nodes}, Avg Energy: {avg_energy:.2f}")
        if patterns:
             logger.info(f"Patterns Recognized: {len(patterns)}")
        # Optionally log memory graph status
        # logger.info(f"Memory Graph: {self.memory_graph.get_status()}")

# Main execution block
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    try:
        core = AI_Core(initial_nodes=5) # Start with 5 nodes
        core.start(execution_cycles=20, interval=0.5) # Run 20 cycles with 0.5s interval
    except RuntimeError as e:
         print(f"AI Core failed to start: {e}")
    except Exception as e:
         print(f"An unexpected error occurred: {e}")
