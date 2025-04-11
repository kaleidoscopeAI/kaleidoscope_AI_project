#!/bin/bash
# Script to consolidate the Kaleidoscope AI project structure v2.
# Run from the project root directory containing kaleidoscope_ai/.

# Stop on first error
set -e

echo "Starting project consolidation (v2)..."

# --- Define Target Locations within kaleidoscope_ai ---
CORE_DIR="kaleidoscope_ai/core"
MODULES_DIR="kaleidoscope_ai/modules"
NODES_DIR="kaleidoscope_ai/nodes"
UTILS_DIR="kaleidoscope_ai/utils"
LLM_DIR="kaleidoscope_ai/llm"
CLIENT_DIR="kaleidoscope_ai/client"
SERVER_DIR="kaleidoscope_ai/server"
SW_ANALYSIS_DIR="kaleidoscope_ai/software_analysis"
MANAGERS_DIR="kaleidoscope_ai/managers"
LOGS_DIR="kaleidoscope_ai/logs"
ROOT_LIB_DIR="kaleidoscope_ai"
PROJECT_ROOT="."
ARCHIVE_DIR="archive" # Archive in project root

# Ensure target directories exist (including potential new ones)
mkdir -p "$CORE_DIR" "$MODULES_DIR" "$NODES_DIR" "$UTILS_DIR" "$LLM_DIR" "$CLIENT_DIR" "$SERVER_DIR" "$ARCHIVE_DIR"

# --- Step 1: Move files OUT of kaleidoscope_ai that belong in root ---
echo "Moving bridge and requirements to project root..."
if [ -f "$ROOT_LIB_DIR/quantum-bridge.py" ]; then
    mv -f "$ROOT_LIB_DIR/quantum-bridge.py" "$PROJECT_ROOT/"
    echo "Moved quantum-bridge.py to root."
else
    echo "INFO: quantum-bridge.py not found in $ROOT_LIB_DIR."
fi

# Decide which requirements.txt to keep (usually the one in root)
if [ -f "$PROJECT_ROOT/requirements.txt" ] && [ -f "$ROOT_LIB_DIR/requirements.txt" ]; then
    echo "INFO: requirements.txt found in root and $ROOT_LIB_DIR. Archiving the one from $ROOT_LIB_DIR."
    mv -f "$ROOT_LIB_DIR/requirements.txt" "$ARCHIVE_DIR/requirements.kaleidoscope_ai.txt.archived"
elif [ -f "$ROOT_LIB_DIR/requirements.txt" ]; then
    echo "INFO: Moving requirements.txt from $ROOT_LIB_DIR to root."
    mv -f "$ROOT_LIB_DIR/requirements.txt" "$PROJECT_ROOT/"
fi


# --- Step 2: Consolidate Duplicates and Place Core Files ---
echo "Consolidating core components into kaleidoscope_ai/..."

# NodeManager: Target core/
# Check if the primary source exists in core, if not, move from nodes if possible
if [ ! -f "$CORE_DIR/NodeManager.py" ] && [ -f "$NODES_DIR/NodeManager.py" ]; then
    echo "Moving NodeManager.py from nodes to core."
    mv -f "$NODES_DIR/NodeManager.py" "$CORE_DIR/"
elif [ -f "$NODES_DIR/NodeManager.py" ]; then
    echo "Archiving duplicate NodeManager.py from nodes."
    mv -f "$NODES_DIR/NodeManager.py" "$ARCHIVE_DIR/NodeManager.nodes.py.archived"
fi

# PerspectiveManager: Target core/
if [ ! -f "$CORE_DIR/PerspectiveManager.py" ] && [ -f "$MODULES_DIR/PerspectiveManager.py" ]; then
    echo "Moving PerspectiveManager.py from modules to core."
    mv -f "$MODULES_DIR/PerspectiveManager.py" "$CORE_DIR/"
elif [ -f "$MODULES_DIR/PerspectiveManager.py" ]; then
     echo "Archiving duplicate PerspectiveManager.py from modules."
     mv -f "$MODULES_DIR/PerspectiveManager.py" "$ARCHIVE_DIR/PerspectiveManager.modules.py.archived"
fi

# SeedManager: Target core/ (Assuming core is better than nodes/modules)
if [ ! -f "$CORE_DIR/SeedManager.py" ] && [ -f "$MODULES_DIR/SeedManager.py" ]; then
     echo "Moving SeedManager.py from modules to core."
     mv -f "$MODULES_DIR/SeedManager.py" "$CORE_DIR/"
elif [ -f "$MODULES_DIR/SeedManager.py" ]; then
     echo "Archiving SeedManager.py from modules."
     mv -f "$MODULES_DIR/SeedManager.py" "$ARCHIVE_DIR/SeedManager.modules.py.archived"
fi
if [ -f "$NODES_DIR/SeedManager.py" ]; then # Also check nodes
    echo "Archiving SeedManager.py from nodes."
    mv -f "$NODES_DIR/SeedManager.py" "$ARCHIVE_DIR/SeedManager.nodes.py.archived"
fi

# laws.py: Target core/
if [ ! -f "$CORE_DIR/laws.py" ] && [ -f "$NODES_DIR/laws.py" ]; then
     echo "Moving laws.py from nodes to core."
     mv -f "$NODES_DIR/laws.py" "$CORE_DIR/"
elif [ -f "$NODES_DIR/laws.py" ]; then
     echo "Archiving duplicate laws.py from nodes."
     mv -f "$NODES_DIR/laws.py" "$ARCHIVE_DIR/laws.nodes.py.archived"
fi
# Archive other law files from nodes
if [ -f "$NODES_DIR/laws (1).py" ]; then mv -f "$NODES_DIR/laws (1).py" "$ARCHIVE_DIR/laws_1.nodes.py.archived"; fi
if [ -f "$NODES_DIR/GrowthLaws.py" ]; then mv -f "$NODES_DIR/GrowthLaws.py" "$ARCHIVE_DIR/GrowthLaws.nodes.py.archived"; fi


# PatternRecognition: Target modules/
if [ ! -f "$MODULES_DIR/PatternRecognition.py" ] && [ -f "$CORE_DIR/PatternRecognition.py" ]; then
     echo "Moving PatternRecognition.py from core to modules."
     mv -f "$CORE_DIR/PatternRecognition.py" "$MODULES_DIR/"
elif [ -f "$CORE_DIR/PatternRecognition.py" ]; then
     echo "Archiving duplicate PatternRecognition.py from core."
     mv -f "$CORE_DIR/PatternRecognition.py" "$ARCHIVE_DIR/PatternRecognition.core.py.archived"
fi

# GPTProcessor: Target llm/
if [ ! -f "$LLM_DIR/GPTProcessor.py" ] && [ -f "$MODULES_DIR/GPTProcessor.py" ]; then
     echo "Moving GPTProcessor.py from modules to llm."
     mv -f "$MODULES_DIR/GPTProcessor.py" "$LLM_DIR/"
elif [ -f "$MODULES_DIR/GPTProcessor.py" ]; then
     echo "Archiving duplicate GPTProcessor.py from modules."
     mv -f "$MODULES_DIR/GPTProcessor.py" "$ARCHIVE_DIR/GPTProcessor.modules.py.archived"
fi

# llm_client.py: Target llm/
if [ ! -f "$LLM_DIR/llm_client.py" ] && [ -f "$CLIENT_DIR/llm_client.py" ]; then
     echo "Moving llm_client.py from client to llm."
     mv -f "$CLIENT_DIR/llm_client.py" "$LLM_DIR/"
elif [ -f "$CLIENT_DIR/llm_client.py" ]; then
     echo "Archiving duplicate llm_client.py from client."
     mv -f "$CLIENT_DIR/llm_client.py" "$ARCHIVE_DIR/llm_client.client.py.archived"
fi

# BaseNode: Target nodes/
if [ ! -f "$NODES_DIR/BaseNode.py" ] && [ -f "$CORE_DIR/BaseNode.py" ]; then
     echo "Moving BaseNode.py from core to nodes."
     mv -f "$CORE_DIR/BaseNode.py" "$NODES_DIR/"
elif [ -f "$CORE_DIR/BaseNode.py" ]; then
     echo "Archiving duplicate BaseNode.py from core."
     mv -f "$CORE_DIR/BaseNode.py" "$ARCHIVE_DIR/BaseNode.core.py.archived"
fi
# Archive other BaseNode versions
if [ -f "$NODES_DIR/base_node.py" ]; then mv -f "$NODES_DIR/base_node.py" "$ARCHIVE_DIR/base_node.nodes.py.archived"; fi
if [ -f "$NODES_DIR/Basenode.py" ]; then mv -f "$NODES_DIR/Basenode.py" "$ARCHIVE_DIR/Basenode.nodes.py.archived"; fi

# CapabilityNode: Target nodes/ (Assuming it exists there)
# TextNode: Target nodes/ (Assuming it exists there)
# VisualNode: Target nodes/ (Assuming it exists there)

# logging_config.py: Target utils/
if [ ! -f "$UTILS_DIR/logging_config.py" ] && [ -f "$LOGS_DIR/logging_config.py" ]; then
     echo "Moving logging_config.py from logs to utils."
     mv -f "$LOGS_DIR/logging_config.py" "$UTILS_DIR/"
elif [ -f "$LOGS_DIR/logging_config.py" ]; then
     echo "Archiving duplicate logging_config.py from logs."
     mv -f "$LOGS_DIR/logging_config.py" "$ARCHIVE_DIR/logging_config.logs.py.archived"
fi

# error_definitions.py: Target kaleidoscope_ai/
if [ ! -f "$ROOT_LIB_DIR/error_definitions.py" ] && [ -f "$SW_ANALYSIS_DIR/error_definitions.py" ]; then
     echo "Moving error_definitions.py from software_analysis to $ROOT_LIB_DIR."
     mv -f "$SW_ANALYSIS_DIR/error_definitions.py" "$ROOT_LIB_DIR/"
elif [ -f "$SW_ANALYSIS_DIR/error_definitions.py" ]; then
     echo "Archiving duplicate error_definitions.py from software_analysis."
     mv -f "$SW_ANALYSIS_DIR/error_definitions.py" "$ARCHIVE_DIR/error_definitions.sw_analysis.py.archived"
fi

# resource_monitor.py: Target utils/
if [ -f "$ROOT_LIB_DIR/resource_monitor.py" ]; then
    echo "Moving resource_monitor.py from $ROOT_LIB_DIR to utils."
    mv -f "$ROOT_LIB_DIR/resource_monitor.py" "$UTILS_DIR/"
fi


# --- Step 3: Clean up miscellaneous/archived files within kaleidoscope_ai ---
echo "Cleaning up other misplaced files within kaleidoscope_ai..."
# Archive specific files listed under core that seem out of place or old
if [ -f "$CORE_DIR/AI_Core.cpython-312.py" ]; then mv -f "$CORE_DIR/AI_Core.cpython-312.py" "$ARCHIVE_DIR/"; fi
if [ -f "$CORE_DIR/app.py" ]; then mv -f "$CORE_DIR/app.py" "$ARCHIVE_DIR/app.core.py.archived"; fi
if [ -f "$CORE_DIR/core.py" ]; then mv -f "$CORE_DIR/core.py" "$ARCHIVE_DIR/core.core.py.archived"; fi
if [ -f "$CORE_DIR/controller.py" ]; then mv -f "$CORE_DIR/controller.py" "$ARCHIVE_DIR/controller.core.py.archived"; fi
if [ -f "$CORE_DIR/kaleidoscope-complete.py" ]; then mv -f "$CORE_DIR/kaleidoscope-complete.py" "$ARCHIVE_DIR/"; fi
if [ -f "$CORE_DIR/enhanced-quantum-kaleidoscope.py" ]; then mv -f "$CORE_DIR/enhanced-quantum-kaleidoscope.py" "$ARCHIVE_DIR/"; fi
if [ -f "$CORE_DIR/MirroredNetwork.py" ]; then mv -f "$CORE_DIR/MirroredNetwork.py" "$ARCHIVE_DIR/"; fi
if [ -f "$CORE_DIR/quantum_kaleidoscope_core.py" ]; then mv -f "$CORE_DIR/quantum_kaleidoscope_core.py" "$ARCHIVE_DIR/"; fi
if [ -f "$CORE_DIR/quantum_kaleidoscope_launcher.py" ]; then mv -f "$CORE_DIR/quantum_kaleidoscope_launcher.py" "$ARCHIVE_DIR/"; fi
if [ -f "$CORE_DIR/quantum-kaleidoscope.py" ]; then mv -f "$CORE_DIR/quantum-kaleidoscope.py" "$ARCHIVE_DIR/quantum-kaleidoscope.core.py.archived"; fi
if [ -f "$CORE_DIR/quantum_kaleidoscope.py" ]; then mv -f "$CORE_DIR/quantum_kaleidoscope.py" "$ARCHIVE_DIR/quantum_kaleidoscope_2.core.py.archived"; fi
if [ -f "$CORE_DIR/unified_quantum_kaleidoscope.py" ]; then mv -f "$CORE_DIR/unified_quantum_kaleidoscope.py" "$ARCHIVE_DIR/"; fi

# Clean modules
if [ -f "$MODULES_DIR/class EmergentPatternDetector.py" ]; then mv -f "$MODULES_DIR/class EmergentPatternDetector.py" "$ARCHIVE_DIR/"; fi

# Clean server dir (archive contents for now)
if [ -f "$SERVER_DIR/kaleidoscope-api-server.py" ]; then mv -f "$SERVER_DIR/kaleidoscope-api-server.py" "$ARCHIVE_DIR/"; fi
if [ -f "$SERVER_DIR/quantum-server.py" ]; then mv -f "$SERVER_DIR/quantum-server.py" "$ARCHIVE_DIR/"; fi

# Clean client dir (archive contents for now)
if [ -f "$CLIENT_DIR/quantum-client.py" ]; then mv -f "$CLIENT_DIR/quantum-client.py" "$ARCHIVE_DIR/"; fi

# Clean software_analysis dir (archive contents)
if [ -f "$SW_ANALYSIS_DIR/error_handler.py" ]; then mv -f "$SW_ANALYSIS_DIR/error_handler.py" "$ARCHIVE_DIR/"; fi
if [ -f "$SW_ANALYSIS_DIR/engine.py" ]; then mv -f "$SW_ANALYSIS_DIR/engine.py" "$ARCHIVE_DIR/"; fi
if [ -f "$SW_ANALYSIS_DIR/more_error_definitions .py" ]; then mv -f "$SW_ANALYSIS_DIR/more_error_definitions .py" "$ARCHIVE_DIR/"; fi
if [ -f "$SW_ANALYSIS_DIR/core-reconstruction.py" ]; then mv -f "$SW_ANALYSIS_DIR/core-reconstruction.py" "$ARCHIVE_DIR/"; fi

# Clean managers dir
if [ -f "$MANAGERS_DIR/quantum-swarm-network.py" ]; then mv -f "$MANAGERS_DIR/quantum-swarm-network.py" "$ARCHIVE_DIR/"; fi

# Clean misplaced middleware
if [ -f "$ROOT_LIB_DIR/quantum-kaleidoscope-middleware.py" ]; then mv -f "$ROOT_LIB_DIR/quantum-kaleidoscope-middleware.py" "$ARCHIVE_DIR/"; fi

# Remove nano directory if it exists
if [ -d "$ROOT_LIB_DIR/nano" ]; then
    rm -rf "$ROOT_LIB_DIR/nano"
    echo "Removed $ROOT_LIB_DIR/nano directory."
fi

# --- Step 4: Clean up loose files in Project Root ---
echo "Cleaning up loose files in project root..."
# Move remaining python files from root to archive
if [ -f "$PROJECT_ROOT/text_node.py" ]; then mv -f "$PROJECT_ROOT/text_node.py" "$ARCHIVE_DIR/"; fi
if [ -f "$PROJECT_ROOT/visualization-integration.py" ]; then mv -f "$PROJECT_ROOT/visualization-integration.py" "$ARCHIVE_DIR/"; fi
if [ -f "$PROJECT_ROOT/visualization-renderer.py" ]; then mv -f "$PROJECT_ROOT/visualization-renderer.py" "$ARCHIVE_DIR/"; fi
if [ -f "$PROJECT_ROOT/quantum_visualization.py" ]; then mv -f "$PROJECT_ROOT/quantum_visualization.py" "$ARCHIVE_DIR/"; fi
if [ -f "$PROJECT_ROOT/quantum_visualizer_new.py" ]; then mv -f "$PROJECT_ROOT/quantum_visualizer_new.py" "$ARCHIVE_DIR/"; fi


# --- Step 5: Ensure __init__.py files exist ---
echo "Ensuring __init__.py files exist in package directories..."
touch "$ROOT_LIB_DIR/__init__.py" # Make kaleidoscope_ai itself a package
touch "$CORE_DIR/__init__.py"
touch "$MODULES_DIR/__init__.py"
touch "$NODES_DIR/__init__.py"
touch "$UTILS_DIR/__init__.py"
touch "$LLM_DIR/__init__.py"
touch "$CLIENT_DIR/__init__.py"
touch "$SERVER_DIR/__init__.py"
touch "$SW_ANALYSIS_DIR/__init__.py"
touch "$MANAGERS_DIR/__init__.py"

echo "--------------------------------------"
echo "Consolidation attempt complete."
echo "Structure:"
echo " kaleidoscope_ai/ (core library)"
echo "  ├── core/"
echo "  ├── modules/"
echo "  ├── nodes/"
echo "  ├── utils/"
echo "  ├── llm/"
echo "  ├── client/"
echo "  ├── server/"
echo "  ├── ... (other library components)"
echo " archive/ (old/duplicate files)"
echo " docs/ (documentation)"
echo " frontend/ (UI files)"
echo " scripts/ (utility scripts)"
echo " quantum_bridge.py (runnable server)"
echo " requirements.txt"
echo ""
echo "IMPORTANT: Review the directories and update Python 'import' statements in your code!"
echo "WARNING: Do NOT use 'sudo' to run this script unless absolutely necessary due to permissions!"
echo "--------------------------------------"
