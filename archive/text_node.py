#!/us# src/nodes/text_node.py
import logging
import time
import json
import re
import asyncio
from typing import List, Dict, Any, Optional, Callable, Tuple
from functools import lru_cache
from collections import deque
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Core system imports with fallback for robustness
try:
    from src.core.base_node import CapabilityNode
    from src.core.laws import CoreLaws
    from src.error.handler import ErrorManager
    from src.core.llm_service import get_llm_service, LLMMessage, LLMOptions, LLMResponse
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.error(f"Import failed: {e}. Using advanced dummies.")
    class CapabilityNode:
        def __init__(self, capability_name, core_laws, node_id): 
            self.node_id, self.core_laws = node_id, core_laws
    class CoreLaws:
        def apply_energy_dynamics(self, energy, task_success, task_complexity): 
            return min(1.0, max(0.0, energy + (0.1 if task_success else -0.1) * task_complexity))
    class ErrorManager:
        def handle_exception(self, e, **kwargs): logger.error(f"Error: {e}", extra=kwargs)
        def error_context(self, *args, **kwargs): return self
        def __enter__(self): pass
        def __exit__(self, exc_type, exc_val, exc_tb): logger.error(f"Context error: {exc_val}") if exc_val else None
    class LLMMessage:
        def __init__(self, role, content): self.role, self.content = role, content
    class LLMOptions:
        def __init__(self, max_tokens=500, temperature=0.7): 
            self.max_tokens, self.temperature = max_tokens, temperature
    class LLMResponse:
        def __init__(self, content): self.content = content
    def get_llm_service():
        class DummyLLM:
            async def generate(self, messages, options): 
                return LLMResponse(f"Dummy: {messages[-1].content}")
        return DummyLLM()

logger = logging.getLogger(__name__)

class TextNode(CapabilityNode):
    """Advanced text processing node with adaptive learning and multi-task capabilities."""
    
    def __init__(self, core_laws: CoreLaws, node_id: Optional[str] = None, 
                 llm_service: Optional[Any] = None, capability_name: str = "text_processing",
                 max_workers: int = 4):
        super().__init__(capability_name=capability_name, core_laws=core_laws, node_id=node_id)
        self.llm_service = llm_service or get_llm_service()
        self.error_manager = ErrorManager()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)  # Parallel task processing
        self.task_history = deque(maxlen=100)  # Adaptive learning from past tasks
        self.performance_metrics = {"success_rate": 0.0, "avg_time": 0.0}  # Self-optimization
        if not self.llm_service:
            logger.warning(f"Node {self.node_id}: LLMService unavailable. Using fallback.")
        logger.info(f"TextNode '{self.node_id}' initialized with {max_workers} workers.")

    @lru_cache(maxsize=256)
    def _estimate_cost_factor(self, data: str) -> float:
        """Advanced cost estimation using text complexity and historical data."""
        try:
            length = len(data)
            complexity = len(re.findall(r'[.!?]', data)) / max(1, length / 1000)  # Sentence density
            avg_time = self.performance_metrics.get("avg_time", 1.0)
            return max(0.5, 1.0 + length * 0.001 + complexity * 0.5 + avg_time * 0.1)
        except Exception as e:
            logger.error(f"Cost estimation failed: {e}")
            return 1.0

    async def _call_llm(self, action: str, prompt: str, llm_options: Optional[LLMOptions] = None) -> LLMResponse:
        """Optimized LLM call with retry logic and context awareness."""
        if not self.llm_service:
            raise ConnectionError("LLM Service unavailable")
        messages = [LLMMessage(role="system", content="You are an advanced NLP assistant."),
                    LLMMessage(role="user", content=prompt)]
        options = llm_options or LLMOptions(temperature=0.7 if "analyze" in action else 0.9)
        for attempt in range(3):
            try:
                with self.error_manager.error_context("LLM Call", action=action, reraise=True):
                    start = time.time()
                    response = await self.llm_service.generate(messages, options)
                    elapsed = time.time() - start
                    self._update_metrics(True, elapsed)
                    return response
            except Exception as e:
                logger.warning(f"LLM attempt {attempt + 1} failed: {e}")
                if attempt == 2:
                    self._update_metrics(False, 0)
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

    def _update_metrics(self, success: bool, elapsed: float):
        """Update performance metrics for self-optimization."""
        sr = self.performance_metrics["success_rate"]
        at = self.performance_metrics["avg_time"]
        n = len(self.task_history) + 1
        self.performance_metrics["success_rate"] = (sr * (n - 1) + (1 if success else 0)) / n
        self.performance_metrics["avg_time"] = (at * (n - 1) + elapsed) / n if elapsed else at

    async def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Advanced sentiment analysis with confidence scores."""
        prompt = f"Analyze sentiment of '{text}' and return JSON: {{'sentiment': 'POSITIVE/NEGATIVE/NEUTRAL', 'confidence': 0.0-1.0}}"
        response = await self._call_llm("analyze_sentiment", prompt)
        try:
            result = json.loads(response.content)
            return {"sentiment": result["sentiment"], "confidence": result["confidence"]}
        except (json.JSONDecodeError, KeyError):
            return {"sentiment": "NEUTRAL", "confidence": 0.5}

    async def _summarize_text(self, text: str, max_length: int = 150, min_length: int = 30) -> Dict[str, Any]:
        """Context-aware summarization with key points extraction."""
        prompt = f"Summarize '{text}' in {min_length}-{max_length} words, including 3 key points in JSON: {{'summary': '...', 'key_points': ['...', '...', '...']}}"
        options = LLMOptions(max_tokens=max_length + 100)
        response = await self._call_llm("summarize", prompt, options)
        try:
            result = json.loads(response.content)
            return result
        except (json.JSONDecodeError, KeyError):
            return {"summary": text[:max_length] + "...", "key_points": []}

    async def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Entity extraction with relationship mapping."""
        prompt = f"Extract entities from '{text}' and their relationships in JSON: {{'entities': {{'PERSON': [...], ...}}, 'relationships': [['entity1', 'entity2', 'relation']]}}"
        response = await self._call_llm("extract_entities", prompt)
        try:
            return json.loads(response.content)
        except (json.JSONDecodeError, KeyError):
            return {"entities": {}, "relationships": []}

    async def execute_capability(self, data: Any, **kwargs) -> Dict[str, Any]:
        """Parallel execution of multiple text processing tasks."""
        action = kwargs.get("action", "full_analysis")
        text = data if isinstance(data, str) else data.get("text") if isinstance(data, dict) else None
        if not text:
            raise ValueError("No text provided")

        logger.info(f"Executing {action} on text (length={len(text)})")
        start_time = time.time()
        try:
            if action == "full_analysis":
                # Parallel execution of multiple analyses
                tasks = [
                    self._analyze_sentiment(text),
                    self._summarize_text(text, **kwargs),
                    self._extract_entities(text)
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                output = {"sentiment": {}, "summary": {}, "entities": {}}
                for i, res in enumerate(results):
                    if isinstance(res, Exception):
                        logger.error(f"Task {i} failed: {res}")
                        continue
                    if i == 0: output["sentiment"] = res
                    elif i == 1: output["summary"] = res
                    elif i == 2: output["entities"] = res
                self.task_history.append((action, True, time.time() - start_time))
                return output
            elif action in {"analyze_sentiment", "summarize", "extract_entities"}:
                method = getattr(self, f"_{action}")
                result = await method(text)
                self.task_history.append((action, True, time.time() - start_time))
                return result
            else:
                raise ValueError(f"Unsupported action: {action}")
        except Exception as e:
            self.error_manager.handle_exception(e, operation=action, component="TextNode")
            self.task_history.append((action, False, time.time() - start_time))
            return {"error": str(e), "status": "failed"}

    def __del__(self):
        """Clean up resources."""
        self.executor.shutdown(wait=False)

# Standalone execution with advanced example
if __name__ == "__main__":
    async def main():
        node = TextNode(core_laws=CoreLaws())
        text = "Elon Musk launched SpaceX in 2002, aiming to revolutionize space travel."
        result = await node.execute_capability({"text": text, "action": "full_analysis"})
        print(json.dumps(result, indent=2))
    asyncio.run(main())
