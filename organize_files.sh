#!/bin/bash
# Script to organize Kaleidoscope AI project files

# Stop on first error
set -e

echo "Creating organizational directories..."
mkdir -p scripts
mkdir -p docs
mkdir -p archive
mkdir -p frontend
mkdir -p quantum_sim

echo "Moving scripts (.sh, deployment)..."
# Move shell scripts and deployment instructions
mv -f start-quantum-bridge.sh setup_kaleidoscope.sh aws_setup.sh scripts/ 2>/dev/null || echo "No .sh scripts to move to scripts/"
mv -f deployment-fix.txt deployment-script.txt scripts/ 2>/dev/null || echo "No deployment text files to move to scripts/"
# Assuming simulation_script.txt is a script, rename to .py
mv -f simulation_script.txt scripts/simulation_script.py 2>/dev/null || echo "simulation_script.txt not found."

echo "Moving documentation and notes (.txt, .md)..."
# Move conceptual documents, readmes, notes
mv -f cubeforollama.txt readme-quantum-bridge.md hyperdimensional-processing.txt docs/ 2>/dev/null || echo "No core docs found."
mv -f integrate_kaleidoscope.txt visualization_integration.txt kaleidoscope_ai_platform.txt docs/ 2>/dev/null || echo "No integration docs found."
mv -f llm_integration.txt system.txt "thoughts EditI understand.txt" docs/ 2>/dev/null || echo "No misc notes found."
mv -f hypercube-viz.txt qsin-network-visualizer.txt visualization.txt visualizer.txt docs/ 2>/dev/null || echo "No visualization docs found."
mv -f example_cube.txt advanced-cube.txt advanced-quantum-cube.txt docs/ 2>/dev/null || echo "No cube example docs found."
mv -f execution-sandbox-system.txt quantum-sync-protocol.txt quantum-kaleidoscope-middleware.txt docs/ 2>/dev/null || echo "No system concept docs found."
mv -f quantum_integration_manager.txt quantum-integrator.txt docs/ 2>/dev/null || echo "No quantum integration docs found."

echo "Moving frontend files (.html, .js)..."
# Move HTML files, rename index.txt
mv -f advanced-quantum-cube.html quantum-kaleidoscope.html frontend/ 2>/dev/null || echo "No primary HTML files found."
mv -f index.txt frontend/index.html 2>/dev/null || echo "index.txt not found."
# Handle JS file and potential duplicate .txt version
mv -f quantum-kaleidoscope-frontend.js frontend/ 2>/dev/null || echo "quantum-kaleidoscope-frontend.js not found."
mv -f quantum-kaleidoscope-frontendjs.txt archive/ 2>/dev/null || echo "No frontendjs.txt to archive." # Archive the .txt version

echo "Moving Quantum Simulation files..."
# Move core quantum simulation Python files if they exist in root
mv -f quantum_kaleidoscope_core.py quantum-server.py quantum-client.py quantum_kaleidoscope_launcher.py quantum_sim/ 2>/dev/null || echo "No quantum .py files found in root to move."

echo "Archiving source .txt files, duplicates, old versions, and unclear files..."
# Archive .txt versions of core AI code (assuming they are sources for files inside kaleidoscope_ai/)
mv -f AI_Core.txt NodeManager.txt GPTProcessor.txt PerspectiveManager.txt SeedManager.txt archive/ 2>/dev/null || echo "No core module sources (1) to archive."
mv -f PatternRecognition.txt BaseNode.txt TextNode.txt VisualNode.txt CapabilityNode.txt archive/ 2>/dev/null || echo "No core module sources (2) to archive."
mv -f error_definitions.txt GrowthLaws.txt llm_client.txt resource_monitor.txt archive/ 2>/dev/null || echo "No core module sources (3) to archive."
mv -f logging_config.txt run.txt archive/ 2>/dev/null || echo "No core module sources (4) to archive."
# Archive .txt versions of quantum sim code
mv -f quantum-client.txt quantum-server.txt quantum_kaleidoscope_core.txt quantum_kaleidoscope_launcher.txt archive/ 2>/dev/null || echo "No quantum sim sources to archive."
# Archive duplicates/old versions
mv -f base_node.txt Basenode.txt "laws (1).txt" laws.txt archive/ 2>/dev/null || echo "No duplicate node/laws files to archive."
# Archive combined/merged scripts
mv -f merged_script.txt unified_quantum_kaleidoscope.txt kaleidoscope-complete.txt archive/ 2>/dev/null || echo "No merged scripts to archive."
# Archive other .txt files that seem like notes, code snippets, or unclear purpose
mv -f core.txt app.txt engine.txt controller.txt error_handler.txt systemadditions.txt archive/ 2>/dev/null || echo "No misc component txt (1) to archive."
mv -f systemfixesandadditions3.txt "more_error_definitions .txt" kaleidoscope-api-server.txt archive/ 2>/dev/null || echo "No misc component txt (2) to archive."
mv -f MemoryGraph.py MirroredNetwork.txt supernode_core.txt supernode-manager.txt supernode-processor.txt archive/ 2>/dev/null || echo "No unplaced module/supernode files to archive."
mv -f "class EmergentPatternDetector.txt" quantum-swarm-network.txt core-reconstruction.txt archive/ 2>/dev/null || echo "No misc class/network files to archive."
# Archive misc quantum files
mv -f "quantum-kaleidoscope(1).txt" quantum-kaleidoscope.txt quantum_kaleidoscope.txt enhanced-quantum-kaleidoscope.txt archive/ 2>/dev/null || echo "No misc quantum files to archive."
# Archive misc system files
mv -f system3.txt.txt archive/ 2>/dev/null || echo "No system3.txt.txt to archive."
# Archive build/cache files
mv -f AI_Core.cpython-312.txt PYZ-00.toc.txt archive/ 2>/dev/null || echo "No cache/build files to archive."
# Archive Untitled files
mv -f Untitled-10.txt "Untitled-10(1).txt" Untitled-11.txt archive/ 2>/dev/null || echo "No Untitled files to archive."

# Cleanup empty directories potentially left by 'mv' if source was empty
find . -maxdepth 1 -type d -empty -delete

echo "--------------------------------------"
echo "File organization complete."
echo "Please review the contents of scripts/, docs/, archive/, frontend/, and quantum_sim/ directories."
echo "Files intended for the core 'kaleidoscope_ai' library were not moved by this script."
echo "Remember to update paths in scripts (like startup scripts) if necessary."
echo "--------------------------------------"
