# src/reconstruction/engine.py
import os
import logging
import asyncio
import json
import re
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import ast  # For Python code analysis
import aiofiles  # For async file operations

# Core imports with fallback for robustness
try:
    from src.llm.client import LLMClient
    from src.error.handler import ErrorManager
    from .config import ReconstructionConfig
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.error(f"Import failed: {e}. Using advanced dummies.")
    class LLMClient:
        async def complete(self, prompt, system_message=None, **kwargs): 
            return f"Improved: {prompt.split('```')[1]}"
    class ErrorManager:
        def handle_exception(self, e, **kwargs): logger.error(f"Error: {e}", extra=kwargs)
        def error_context(self, *args, **kwargs): return self
        def __enter__(self): pass
        def __exit__(self, exc_type, exc_val, exc_tb): logger.error(f"Context error: {exc_val}") if exc_val else None
    class ReconstructionConfig:
        def __init__(self): 
            self.target_language = None
            self.optimize_performance = True 
            self.add_comments = True

logger = logging.getLogger(__name__)

class ReconstructionEngine:
    """Advanced engine for reconstructing, optimizing, and translating code with AI-driven insights."""

    def __init__(self, output_dir: Optional[str] = None, llm_client: Optional[LLMClient] = None, 
                 max_workers: int = 4):
        """Initialize with advanced features."""
        self.base_output_dir = output_dir or os.path.join(os.environ.get("KALEIDOSCOPE_DATA_DIR", "data"), "reconstructed")
        os.makedirs(self.base_output_dir, exist_ok=True)
        self.llm_client = llm_client or LLMClient()
        self.error_manager = ErrorManager()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)  # Parallel processing
        self.performance_cache = {}  # Cache for profiling results
        self.language_handlers = {
            "python": self._transform_python,
            "javascript": self._transform_generic,  # Add more as needed
            # Extend with other languages (e.g., "cpp", "java") with specific handlers
        }
        self.supported_languages = {
            "python": [".py"], "javascript": [".js"],  # Expand as needed
        }
        self.language_matchers = {
            "python": re.compile(r'^\s*(import\s+|def\s+\w+\s*\(|class\s+\w+\s*[:\(])', re.MULTILINE),
            "javascript": re.compile(r'^\s*(function\s+\w+\s*\(|const\s+\w+\s*=)', re.MULTILINE),
        }
        logger.info(f"ReconstructionEngine initialized with {max_workers} workers.")

    @lru_cache(maxsize=128)
    def detect_language(self, file_path: str, content: str) -> str:
        """Detect programming language with semantic analysis."""
        ext = os.path.splitext(file_path)[1].lower()
        for lang, extensions in self.supported_languages.items():
            if ext in extensions:
                return lang
        for lang, pattern in self.language_matchers.items():
            if pattern.search(content[:5000]):  # Check first 5KB
                return lang
        logger.warning(f"Unknown language for {file_path}. Defaulting to 'generic'.")
        return "generic"

    async def _apply_llm_transformation(self, content: str, config: ReconstructionConfig, 
                                      language: str, system_message: str) -> Tuple[str, Dict[str, Any]]:
        """Advanced LLM transformation with optimization suggestions."""
        prompt = f"""Transform this {language} code with these goals:
- Optimize performance: {config.optimize_performance}
- Add explanatory comments: {config.add_comments}
- Preserve semantics
Return JSON: {{'code': '...', 'suggestions': ['...', ...], 'performance_gain': 0.0-1.0}}
``` {language}
{content[:20000]}  # Limit input size
```"""
        try:
            with self.error_manager.error_context("LLM Transform", language=language, reraise=True):
                result = await self.llm_client.complete(prompt, system_message=system_message)
                data = json.loads(result.strip())
                return data["code"], {
                    "suggestions": data.get("suggestions", []),
                    "performance_gain": data.get("performance_gain", 0.0)
                }
        except Exception as e:
            logger.error(f"LLM transformation failed: {e}")
            return content, {"suggestions": [f"Error: {e}"], "performance_gain": 0.0}

    async def _transform_python(self, content: str, config: ReconstructionConfig, language: str) -> Tuple[str, Dict[str, Any]]:
        """Python-specific transformation with static analysis."""
        system_message = "You are a Python optimization expert adhering to PEP 8 and modern practices."
        try:
            # Static analysis for complexity
            tree = ast.parse(content)
            complexity = sum(1 for _ in ast.walk(tree)
