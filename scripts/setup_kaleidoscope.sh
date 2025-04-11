#!/bin/bash

# setup_kaleidoscope.sh - Sets up the kaleidoscope_ai project with error handling

# Exit on any error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Check prerequisites
check_prerequisites() {
    echo "Checking prerequisites..."
    for cmd in python3 nano chmod; do
        if ! command -v "$cmd" &> /dev/null; then
            echo -e "${RED}Error: '$cmd' is required but not installed. Please install it.${NC}"
            exit 1
        fi
    done
    echo -e "${GREEN}Prerequisites met.${NC}"
}

# Create project directory
create_project_dir() {
    echo "Creating project directory..."
    if [ -d "kaleidoscope_ai" ]; then
        echo "Directory 'kaleidoscope_ai' already exists. Overwriting files as needed."
    else
        mkdir -p kaleidoscope_ai || {
            echo -e "${RED}Error: Failed to create 'kaleidoscope_ai' directory.${NC}"
            exit 1
        }
    fi
    cd kaleidoscope_ai || {
        echo -e "${RED}Error: Failed to enter 'kaleidoscope_ai' directory.${NC}"
        exit 1
    }
    echo -e "${GREEN}Project directory created and entered.${NC}"
}

# Create .env file
create_env_file() {
    echo "Creating .env file..."
    cat << 'EOF' > .env || {
        echo -e "${RED}Error: Failed to create '.env' file.${NC}"
        exit 1
    }
# LLM Configuration for Ollama
LLM_PROVIDER="ollama"
LLM_MODEL="mistral:latest"  # Or specify another model served by your Ollama instance
LLM_API_KEY=""              # API key is not needed for Ollama

# Set other provider keys to empty just in case parts of the code check them directly
OPENAI_API_KEY=""
ANTHROPIC_API_KEY=""
EOF
    echo -e "${GREEN}.env file created.${NC}"
}

# Create requirements.txt
create_requirements_file() {
    echo "Creating requirements.txt..."
    cat << 'EOF' > requirements.txt || {
        echo -e "${RED}Error: Failed to create 'requirements.txt' file.${NC}"
        exit 1
    }
numpy
scipy
networkx
matplotlib
plotly # For visualization
transformers[torch] # Or transformers[tensorflow]
torch # Or tensorflow
spacy
websockets
aiohttp
requests
openai
anthropic
python-dotenv
psutil
qiskit # Added based on quantum file imports
EOF
    echo -e "${GREEN}requirements.txt created.${NC}"
}

# Create run.py
create_run_file() {
    echo "Creating run.py..."
    cat << 'EOF' > run.py || {
        echo -e "${RED}Error: Failed to create 'run.py' file.${NC}"
        exit 1
    }
# run.py
import sys
import os
import logging

# Dynamically set PYTHONPATH to include the project root
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the main AI Core class and logging config AFTER setting the path
try:
    from core.AI_Core import AI_Core
except ImportError as e:
    print(f"Error: Could not import AI_Core from core: {e}")
    print("Ensure 'core/AI_Core.py' exists in the project root.")
    sys.exit(1)

try:
    from utils.logging_config import configure_logging, get_logger
except ImportError as e:
    print(f"Error importing logging config: {e}. Ensure utils/logging_config.py exists and utils/__init__.py is present.")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    get_logger = logging.getLogger

if __name__ == "__main__":
    log_dir = os.path.join(project_root, "logs")
    try:
        configure_logging(log_dir=log_dir)
    except Exception as e:
        print(f"Error configuring logging: {e}. Using fallback configuration.")
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    logger = get_logger(__name__)

    try:
        logger.info("Starting Unified AI Core via run.py...")
        ai_system = AI_Core(initial_nodes=5)
        ai_system.start(execution_cycles=15, interval=1)
        logger.info("Unified AI Core run finished.")
    except Exception as e:
        logger.error(f"Failed to run AI Core: {e}", exc_info=True)
        print(f"An error occurred. Check logs/ai_core_run.log for details.")
        sys.exit(1)
EOF
    chmod +x run.py || {
        echo -e "${RED}Error: Failed to make 'run.py' executable.${NC}"
        exit 1
    }
    echo -e "${GREEN}run.py created and made executable.${NC}"
}

# Create llm_client.py
create_llm_client_file() {
    echo "Creating llm_client.py..."
    cat << 'EOF' > llm_client.py || {
        echo -e "${RED}Error: Failed to create 'llm_client.py' file.${NC}"
        exit 1
    }
#!/usr/bin/env python3
import os
import json
import logging
import asyncio
import subprocess
import requests
import time
import hashlib
import aiohttp
import sys
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
from pathlib import Path

try:
    from utils.logging_config import configure_logging, get_logger
except ImportError:
    print("Warning: Could not import logging_config from utils. Using basic logging.")
    logging.basicConfig(level=logging.INFO)
    get_logger = logging.getLogger

if __name__ == "__main__":
    try:
        configure_logging()
    except Exception as e:
        print(f"Warning: Could not configure logging: {e}")

logger = get_logger(__name__)

class LLMClient:
    def __init__(self, config_file: str = None, api_key: str = None, model: str = None, provider: str = None, endpoint: str = None, cache_dir: str = ".llm_cache", max_tokens: int = 2048, temperature: float = 0.7, request_timeout: int = 60):
        default_provider = os.environ.get("LLM_PROVIDER", "ollama")
        default_model = os.environ.get("LLM_MODEL", "mistral:latest")
        default_api_key = os.environ.get("LLM_API_KEY")
        self.config = {
            "api_key": api_key if api_key is not None else default_api_key,
            "model": model if model is not None else default_model,
            "provider": provider if provider is not None else default_provider,
            "endpoint": endpoint,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "request_timeout": request_timeout
        }
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                self.config.update({k: v for k, v in file_config.items() if self.config.get(k) is None})
                logger.info(f"Loaded LLM configuration from {config_file}")
            except Exception as e:
                logger.error(f"Error loading config from {config_file}: {e}")
        if not self.config.get("api_key"):
            if self.config["provider"] == "openai":
                self.config["api_key"] = os.environ.get("OPENAI_API_KEY", "")
            elif self.config["provider"] == "anthropic":
                self.config["api_key"] = os.environ.get("ANTHROPIC_API_KEY", "")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.cache_hits = 0
        self._setup_provider()
        self.request_queue = asyncio.Queue()
        self.batch_size = 5
        self.active_requests = 0
        self.is_processing = False
        self.request_semaphore = asyncio.Semaphore(10)
        self.perf_metrics = {"avg_latency": 0, "success_rate": 1.0, "total_requests": 0, "failed_requests": 0, "total_tokens": 0}
        logger.info(f"LLMClient initialized with provider: {self.config['provider']}, model: {self.config['model']}")

    def _setup_provider(self):
        provider = self.config["provider"].lower()
        if provider == "ollama":
            endpoint = self.config.get("endpoint", "http://localhost:11434")
            self.config["endpoint"] = endpoint
            try:
                response = requests.get(f"{endpoint}/api/tags", timeout=5)
                if response.status_code != 200:
                    logger.warning(f"Ollama API not responding at {endpoint}.")
                else:
                    models = response.json().get("models", [])
                    if self.config["model"] not in [m.get("name") for m in models]:
                        logger.warning(f"Model {self.config['model']} not found in Ollama.")
            except Exception as e:
                logger.warning(f"Cannot connect to Ollama: {e}.")
        elif provider in ["openai", "anthropic"] and not self.config.get("api_key"):
            logger.warning(f"No API key for {provider}. API calls may fail.")
        elif provider not in ["ollama", "openai", "anthropic", "llama.cpp", "custom"]:
            logger.error(f"Unknown provider: {provider}.")
            sys.exit(1)

    def _get_cache_key(self, prompt: str, system_message: str = None, **kwargs):
        cache_config = self.config.copy()
        cache_config.update(kwargs)
        cache_dict = {"prompt": prompt, "system_message": system_message, "model": cache_config.get("model"), "temperature": cache_config.get("temperature"), "max_tokens": cache_config.get("max_tokens"), "provider": cache_config.get("provider"), "stop_sequences": str(cache_config.get("stop_sequences", [])), "top_p": cache_config.get("top_p")}
        return hashlib.md5(json.dumps(cache_dict, sort_keys=True, default=str).encode()).hexdigest()

    def _check_cache(self, cache_key: str) -> Optional[str]:
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                if cached_data.get("expiry", float('inf')) > time.time():
                    self.cache_hits += 1
                    logger.debug(f"Cache hit for key {cache_key}")
                    return cached_data.get("response")
            except Exception as e:
                logger.error(f"Error reading cache file {cache_file}: {e}")
        return None

    def _save_to_cache(self, cache_key: str, response: str, ttl_seconds: int = 86400):
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump({"response": response, "created": time.time(), "expiry": time.time() + ttl_seconds}, f)
            logger.debug(f"Saved response to cache: {cache_key}")
        except Exception as e:
            logger.error(f"Error saving to cache file {cache_file}: {e}")

    async def complete(self, prompt: str, system_message: str = None, use_cache: bool = True, **kwargs):
        request_config = self.config.copy()
        request_config.update(kwargs)
        provider = request_config["provider"].lower()
        cache_key = self._get_cache_key(prompt, system_message, **request_config) if use_cache else None
        if use_cache and cache_key:
            cached_response = self._check_cache(cache_key)
            if cached_response:
                return cached_response
        self.perf_metrics["total_requests"] += 1
        start_time = time.time()
        try:
            if provider == "ollama":
                response = await self._complete_ollama(prompt, system_message, **request_config)
            else:
                raise ValueError(f"Provider {provider} not fully implemented in this setup.")
            if use_cache and cache_key:
                self._save_to_cache(cache_key, response)
            latency = time.time() - start_time
            self.perf_metrics["avg_latency"] = ((self.perf_metrics["avg_latency"] * (self.perf_metrics["total_requests"] - 1)) + latency) / self.perf_metrics["total_requests"]
            self.perf_metrics["total_tokens"] += len(prompt.split()) + len(response.split())
            return response
        except Exception as e:
            self.perf_metrics["failed_requests"] += 1
            self.perf_metrics["success_rate"] = 1.0 - (self.perf_metrics["failed_requests"] / self.perf_metrics["total_requests"])
            logger.error(f"Error completing prompt: {e}")
            raise

    async def _complete_ollama(self, prompt: str, system_message: str = None, **kwargs):
        endpoint = kwargs.get("endpoint", "http://localhost:11434")
        model = kwargs.get("model")
        max_tokens = kwargs.get("max_tokens", 2048)
        temperature = kwargs.get("temperature", 0.7)
        timeout = kwargs.get("request_timeout", 60)
        request_body = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": temperature, "num_predict": max_tokens}}
        if system_message:
            request_body["system"] = system_message
        request_body["options"] = {k: v for k, v in request_body["options"].items() if v is not None}
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(f"{endpoint}/api/generate", json=request_body, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                    response_data = await response.json()
                    if response.status == 200:
                        return response_data.get("response", "").strip()
                    else:
                        raise RuntimeError(f"Ollama API error: {response_data.get('error', 'Unknown error')}")
            except Exception as e:
                raise RuntimeError(f"Ollama request failed: {e}")

if __name__ == "__main__":
    async def test_client():
        client = LLMClient()
        prompt = "Test prompt"
        try:
            response = await client.complete(prompt)
            print(f"Response: {response}")
        except Exception as e:
            print(f"Test failed: {e}")
    asyncio.run(test_client())
EOF
    echo -e "${GREEN}llm_client.py created.${NC}"
}

# Create error_definitions.py
create_error_definitions_file() {
    echo "Creating error_definitions.py..."
    cat << 'EOF' > error_definitions.py || {
        echo -e "${RED}Error: Failed to create 'error_definitions.py' file.${NC}"
        exit 1
    }
#!/usr/bin/env python3
import time
import traceback
import hashlib
import logging
import os
import sys
from enum import Enum, auto
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional

try:
    from utils.logging_config import configure_logging, get_logger
except ImportError:
    print("Warning: Could not import logging_config from utils. Using basic logging.")
    logging.basicConfig(level=logging.INFO)
    get_logger = logging.getLogger

if __name__ == "__main__":
    try:
        configure_logging()
    except Exception as e:
        print(f"Warning: Could not configure logging: {e}")

logger = get_logger(__name__)

class ErrorSeverity(Enum):
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()
    FATAL = auto()

class ErrorCategory(Enum):
    SYSTEM = auto()
    NETWORK = auto()
    API = auto()
    LLM = auto()
    CONFIGURATION = auto()
    UNKNOWN = auto()

@dataclass
class ErrorContext:
    operation: str
    input_data: Optional[Any] = None
    file_path: Optional[str] = None
    component: Optional[str] = None
    additional_info: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        if self.input_data and not isinstance(self.input_data, (str, int, float, bool, type(None))):
            self.input_data = str(self.input_data)[:500] + "...(truncated)"

@dataclass
class EnhancedError:
    message: str
    category: ErrorCategory = ErrorCategory.UNKNOWN
    severity: ErrorSeverity = ErrorSeverity.ERROR
    exception: Optional[Exception] = None
    traceback_str: Optional[str] = field(default=None)
    context: Optional[ErrorContext] = None
    error_id: str = field(default_factory=lambda: hashlib.md5(str(time.time_ns()).encode()).hexdigest())

    def __post_init__(self):
        if self.exception and not self.traceback_str:
            self.traceback_str = ''.join(traceback.format_exception(type(self.exception), self.exception, self.exception.__traceback__))

    def log(self, logger_instance=None):
        logger_instance = logger_instance or logger
        log_message = f"[{self.error_id}] {self.category.name} [{self.severity.name}] - {self.message}"
        level_map = {ErrorSeverity.DEBUG: logging.DEBUG, ErrorSeverity.INFO: logging.INFO, ErrorSeverity.WARNING: logging.WARNING, ErrorSeverity.ERROR: logging.ERROR, ErrorSeverity.CRITICAL: logging.CRITICAL, ErrorSeverity.FATAL: logging.CRITICAL}
        logger_instance.log(level_map.get(self.severity, logging.ERROR), log_message)
        if self.traceback_str and self.severity in [ErrorSeverity.WARNING, ErrorSeverity.ERROR, ErrorSeverity.CRITICAL, ErrorSeverity.FATAL]:
            logger_instance.log(level_map.get(self.severity), f"Traceback:\n{self.traceback_str}")

if __name__ == "__main__":
    try:
        1 / 0
    except Exception as e:
        context = ErrorContext(operation="Test division", input_data="1/0")
        err = EnhancedError(message="Division by zero", category=ErrorCategory.SYSTEM, severity=ErrorSeverity.ERROR, exception=e, context=context)
        err.log()
EOF
    echo -e "${GREEN}error_definitions.py created.${NC}"
}

# Main execution with error trapping
main() {
    trap 'echo -e "${RED}Script interrupted or failed at line $LINENO${NC}"; exit 1' ERR INT TERM
    echo "Starting Kaleidoscope AI project setup..."
    check_prerequisites
    create_project_dir
    create_env_file
    create_requirements_file
    create_run_file
    create_llm_client_file
    create_error_definitions_file
    echo -e "${GREEN}Kaleidoscope AI project setup completed successfully!${NC}"
    echo "Next steps:"
    echo "  cd kaleidoscope_ai"
    echo "  python3 -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  pip install -r requirements.txt"
    echo "  python run.py"
}

# Run main
main
