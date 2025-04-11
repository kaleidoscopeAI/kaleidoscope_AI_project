#!/bin/bash

# Exit script on error
set -e

# List of updates
declare -A files=(
  ["kaleidoscope_ai/core/AI_Core.py"]="from .GrowthLaws import GrowthLaws -> from .laws import GrowthLaws
from modules.PerspectiveManager import PerspectiveManager -> from .PerspectiveManager import PerspectiveManager
from modules.SeedManager import SeedManager -> from .SeedManager import SeedManager
Remove: from modules.MirroredNetwork import MirroredNetwork
Remove: from modules.MemoryGraph import MemoryGraph
from modules.PatternRecognition import PatternRecognition -> from kaleidoscope_ai.modules import PatternRecognition
from modules.GPTProcessor import GPTProcessor -> from kaleidoscope_ai.llm import GPTProcessor
from nodes.TextNode import TextNode -> from kaleidoscope_ai.nodes import TextNode
from .BaseNode import BaseNode -> from kaleidoscope_ai.nodes import BaseNode"
  ["kaleidoscope_ai/core/NodeManager.py"]="from .BaseNode import BaseNode -> from kaleidoscope_ai.nodes import BaseNode"
  ["kaleidoscope_ai/llm/GPTProcessor.py"]="Add: import numpy as np
Add: from typing import Tuple"
  ["kaleidoscope_ai/llm/llm_client.py"]="from src.utils.logging_config import -> from kaleidoscope_ai.utils.logging_config import"
  ["kaleidoscope_ai/nodes/CapabilityNode.py"]="Add: import logging
Add: from typing import Optional, Any
Add: from collections import deque
from core import CoreLaws -> from kaleidoscope_ai.core.laws import GrowthLaws"
  ["kaleidoscope_ai/nodes/TextNode.py"]="Add: from collections import deque
from core import CoreLaws -> from kaleidoscope_ai.core.laws import GrowthLaws
from CapabilityNode import CapabilityNode -> from .CapabilityNode import CapabilityNode
from modules.GPTProcessor import GPTProcessor -> from kaleidoscope_ai.llm import GPTProcessor"
  ["kaleidoscope_ai/nodes/VisualNode.py"]="Add: from collections import deque
from core import CoreLaws -> from kaleidoscope_ai.core.laws import GrowthLaws
from CapabilityNode import CapabilityNode -> from .CapabilityNode import CapabilityNode"
  ["kaleidoscope_ai/error_definitions.py"]="from utils.logging_config import -> from kaleidoscope_ai.utils.logging_config import"
)

# Function to update a file
update_file() {
  local file=$1
  local changes=$2

  echo "Updating $file..."
  while IFS= read -r change; do
    if [[ $change == Add* ]]; then
      local content=${change#Add: }
      sed -i "1s/^/$content\n/" "$file"
    elif [[ $change == Remove* ]]; then
      local content=${change#Remove: }
      sed -i "/$content/d" "$file"
    else
      local old=${change%% -> *}
      local new=${change#* -> }
      sed -i "s|$old|$new|g" "$file"
    fi
  done <<< "$changes"
}

# Iterate through all files to make updates
for file in "${!files[@]}"; do
  update_file "$file" "${files[$file]}"
done

# Stage all changes
git add .

# Commit changes
git commit -m "Update imports and remove deprecated references for better modularity"

# Push changes
echo "Please provide the branch name to push changes:"
read branch
git push origin "$branch"

echo "Changes committed and pushed successfully!"
