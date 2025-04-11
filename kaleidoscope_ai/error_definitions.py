#!/usr/bin/env python3
import time
import traceback
import hashlib
import logging
import os
import sys
import json
import uuid
from enum import Enum, auto
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Union, Tuple

# Adjust logging import assuming logging_config.py is in utils/
try:
    from kaleidoscope_ai.utils.logging_config import configure_logging, get_logger
except ImportError:
    # Fallback if run directly from root perhaps?
    print("Warning: Could not import logging_config from utils. Using basic logging.")
    logging.basicConfig(level=logging.INFO)
    get_logger = logging.getLogger

# Check if running directly to configure logging for tests
if __name__ == "__main__":
    try:
        configure_logging()
    except NameError:
        pass

logger = get_logger(__name__)


class ErrorSeverity(Enum):
    """Severity levels for errors"""
    DEBUG = auto()      # Minor issues that don't affect operation
    INFO = auto()       # Informational errors, minimal impact
    WARNING = auto()    # Potential problems that should be addressed
    ERROR = auto()      # Serious problems affecting functionality
    CRITICAL = auto()   # Severe errors that prevent operation
    FATAL = auto()      # Catastrophic errors requiring immediate attention


class ErrorCategory(Enum):
    """Categories for different types of errors"""
    SYSTEM = auto()           # Operating system or environment errors
    NETWORK = auto()          # Network connectivity issues
    API = auto()              # API interaction errors (LLM, external)
    PARSING = auto()          # Errors parsing files or data
    NODE_OPERATION = auto()   # Errors within a Node's processing
    CORE_LOGIC = auto()       # Errors in AI_Core or NodeManager
    CONFIGURATION = auto()    # Errors related to configuration loading/values
    RESOURCE = auto()         # Resource availability errors (Memory, CPU, Disk)
    VALIDATION = auto()       # Input validation errors
    INITIALIZATION = auto()   # Errors during system or component startup
    REPLICATION = auto()      # Errors during node replication
    QUANTUM_SIM = auto()      # Errors related to the quantum components
    VISUALIZATION = auto()    # Errors related to visualization components
    UNKNOWN = auto()          # Unclassified errors


@dataclass
class ErrorContext:
    """Contextual information about an error"""
    operation: str                          # Operation being performed when error occurred
    component: Optional[str] = None         # Component where error occurred (e.g., 'AI_Core', 'TextNode')
    node_id: Optional[str] = None           # ID of the specific node involved, if applicable
    input_data_summary: Optional[str] = None # Summary/type of input data related to the error
    file_path: Optional[str] = None         # Path to relevant file, if applicable
    task_id: Optional[str] = None           # If error occurred within a specific task/cycle
    additional_info: Dict[str, Any] = field(default_factory=dict)  # Other relevant context
    timestamp: float = field(default_factory=time.time)  # Error timestamp

    # Ensure input_data_summary is generated safely
    def set_input_summary(self, input_data: Any):
         if input_data is None:
              self.input_data_summary = "None"
              return
         try:
             if isinstance(input_data, (str, bytes)):
                  self.input_data_summary = f"{type(input_data).__name__} (len={len(input_data)})"
             elif isinstance(input_data, (list, tuple, set)):
                  self.input_data_summary = f"{type(input_data).__name__} (len={len(input_data)}, first_item_type={type(input_data[0]).__name__ if len(input_data)>0 else 'N/A'})"
             elif isinstance(input_data, dict):
                  self.input_data_summary = f"dict (keys={list(input_data.keys())[:5]})" # Show first 5 keys
             elif hasattr(input_data, 'shape'): # Handle numpy arrays etc.
                  self.input_data_summary = f"{type(input_data).__name__} (shape={input_data.shape})"
             else:
                  self.input_data_summary = f"{type(input_data).__name__}"
         except Exception:
             self.input_data_summary = f"Data of type {type(input_data).__name__} (summary failed)"


@dataclass
class EnhancedError:
    """Enhanced error object with detailed information"""
    message: str                            # Clear error message
    category: ErrorCategory = ErrorCategory.UNKNOWN  # Default category
    severity: ErrorSeverity = ErrorSeverity.ERROR    # Default severity
    exception: Optional[Exception] = None   # Original exception, if any
    traceback_str: Optional[str] = field(init=False, default=None) # Store traceback as string
    context: Optional[ErrorContext] = None  # Error context
    error_id: str = field(default_factory=lambda: f"ERR-{uuid.uuid4().hex[:8]}")  # Unique, readable ID
    timestamp: float = field(default_factory=time.time)  # Error timestamp

    def __post_init__(self):
        """Initialize traceback if exception is provided"""
        if self.exception:
            try:
                 self.traceback_str = ''.join(traceback.format_exception(
                     type(self.exception),
                     self.exception,
                     self.exception.__traceback__
                 ))
            except Exception:
                 self.traceback_str = "Could not format traceback."

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = {
            "error_id": self.error_id,
            "message": self.message,
            "category": self.category.name, # Use name for consistency
            "severity": self.severity.name, # Use name for consistency
            "timestamp": self.timestamp,
            "exception_type": type(self.exception).__name__ if self.exception else None,
            "exception_message": str(self.exception) if self.exception else None,
        }

        if self.traceback_str:
            # Limit traceback length for storage if necessary
            limit = 5000
            truncated = len(self.traceback_str) > limit
            result["traceback"] = self.traceback_str[:limit] + ('... (truncated)' if truncated else '')

        if self.context:
            # Convert context dataclass to dict
            result["context"] = asdict(self.context)

        return result

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    def log(self, logger_instance=None):
        """Log the error using a provided logger instance based on severity"""
        if logger_instance is None:
            logger_instance = logger # Use the module logger by default

        log_message = f"[{self.error_id}] [{self.category.name}] {self.message}"

        context_details = []
        if self.context:
            if self.context.component: context_details.append(f"Component={self.context.component}")
            if self.context.operation: context_details.append(f"Op={self.context.operation}")
            if self.context.node_id: context_details.append(f"Node={self.context.node_id}")
            if self.context.task_id: context_details.append(f"Task={self.context.task_id}")
            if self.context.file_path: context_details.append(f"File={os.path.basename(self.context.file_path)}") # Show only filename
            if context_details:
                log_message += f" (Context: {', '.join(context_details)})"
            if self.context.additional_info:
                 log_message += f" (Info: {self.context.additional_info})" # Log additional info

        # Map severity to logging level
        level_map = {
            ErrorSeverity.DEBUG: logging.DEBUG,
            ErrorSeverity.INFO: logging.INFO,
            ErrorSeverity.WARNING: logging.WARNING,
            ErrorSeverity.ERROR: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL,
            ErrorSeverity.FATAL: logging.CRITICAL # FATAL logs as CRITICAL
        }
        log_level = level_map.get(self.severity, logging.ERROR) # Default to ERROR

        # Log main message
        logger_instance.log(log_level, log_message)

        # Log exception message and traceback for WARNING and above
        if log_level >= logging.WARNING:
            if self.exception:
                 # Log exception message distinctly if traceback is long
                 logger_instance.log(log_level, f"[{self.error_id}] Exception: {type(self.exception).__name__}: {str(self.exception)}")
            if self.traceback_str:
                 # Log traceback at a potentially lower level (e.g., DEBUG) or same level
                 # For simplicity, logging at the same level but indicating it's a traceback
                 logger_instance.log(log_level, f"[{self.error_id}] Traceback follows:\n{self.traceback_str}")


# --- Define Specific Error Classes (Optional but Recommended) ---

class AIError(Exception):
    """Base class for custom AI errors."""
    def __init__(self, message: str,
                 category: ErrorCategory = ErrorCategory.UNKNOWN,
                 severity: ErrorSeverity = ErrorSeverity.ERROR,
                 context: Optional[ErrorContext] = None,
                 original_exception: Optional[Exception] = None):
        super().__init__(message)
        self.enhanced_error = EnhancedError(
            message=message,
            category=category,
            severity=severity,
            exception=original_exception,
            context=context
        )

    def log(self, logger_instance=None):
        self.enhanced_error.log(logger_instance)

    def to_dict(self):
        return self.enhanced_error.to_dict()

    def to_json(self):
        return self.enhanced_error.to_json()


class NodeError(AIError):
    """Error related to Node operations."""
    def __init__(self, message: str, node_id: Optional[str] = None, **kwargs):
        # Ensure context exists or create it
        context = kwargs.pop('context', ErrorContext(operation="Node Operation"))
        context.node_id = node_id
        context.component = kwargs.pop('component', 'Node') # Set component if not provided
        super().__init__(message, category=ErrorCategory.NODE_OPERATION, context=context, **kwargs)


class LLMError(AIError):
     """Error related to LLM interactions."""
     def __init__(self, message: str, provider: Optional[str] = None, **kwargs):
        context = kwargs.pop('context', ErrorContext(operation="LLM Interaction"))
        context.component = kwargs.pop('component', 'LLMClient')
        if provider: context.additional_info['provider'] = provider
        super().__init__(message, category=ErrorCategory.API, context=context, **kwargs)

class ConfigError(AIError):
     """Error related to configuration."""
     def __init__(self, message: str, key: Optional[str] = None, **kwargs):
        context = kwargs.pop('context', ErrorContext(operation="Configuration"))
        context.component = kwargs.pop('component', 'Configuration')
        if key: context.additional_info['config_key'] = key
        super().__init__(message, category=ErrorCategory.CONFIGURATION, severity=ErrorSeverity.CRITICAL, context=context, **kwargs)


# Example Usage
if __name__ == "__main__":
    logger = get_logger("ErrorTest") # Get named logger

    try:
        # Simulate an error during node processing
        node_id_example = "Node_XYZ"
        input_summary = "dict (keys=['text', 'image'])"
        context1 = ErrorContext(operation="process_text", component="TextNode", node_id=node_id_example)
        context1.input_data_summary = input_summary # Set summary separately
        raise ValueError("Invalid text encoding detected")
    except Exception as e:
        node_err = NodeError("Failed to process text data", node_id=node_id_example, context=context1, original_exception=e)
        node_err.log(logger)
        # print("\nNode Error JSON:")
        # print(node_err.to_json())

    try:
        # Simulate an LLM API error
        context2 = ErrorContext(operation="complete", component="LLMClient")
        context2.set_input_summary("This is a test prompt.") # Use helper method
        raise ConnectionError("API connection timeout")
    except Exception as e:
        llm_err = LLMError("LLM request failed", provider="openai", context=context2, original_exception=e, severity=ErrorSeverity.WARNING)
        llm_err.log(logger)
        # print("\nLLM Error JSON:")
        # print(llm_err.to_json())

    try:
        # Simulate a configuration error
        raise KeyError("Missing 'api_key' in configuration")
    except Exception as e:
        config_err = ConfigError("Essential configuration parameter missing", key="api_key", original_exception=e)
        config_err.log(logger)
        # print("\nConfig Error JSON:")
        # print(config_err.to_json())
