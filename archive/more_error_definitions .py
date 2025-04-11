#!/usr/bin/env python3
import time
import traceback
import hashlib
import logging
import os
import sys
from enum import Enum, auto
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Union, Tuple

# Add project root to path if running this file directly
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.insert(0, project_root)
    from src.utils.logging_config import configure_logging, get_logger
    configure_logging()
else:
    from src.utils.logging_config import get_logger

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
    API = auto()              # API interaction errors
    PARSING = auto()          # Errors parsing files or data
    ANALYSIS = auto()         # Errors during code analysis
    DECOMPILATION = auto()    # Errors in decompilation process (if applicable)
    SPECIFICATION = auto()    # Errors generating specifications (if applicable)
    RECONSTRUCTION = auto()   # Errors during code reconstruction (if applicable)
    MIMICRY = auto()          # Errors during code mimicry (if applicable)
    LLM = auto()              # LLM API or integration errors
    SECURITY = auto()         # Security-related errors
    RESOURCE = auto()         # Resource availability errors
    VALIDATION = auto()       # Input validation errors
    RECOVERY = auto()         # Recovery process errors
    CONFIGURATION = auto()    # Errors related to configuration
    TASK_EXECUTION = auto()   # Errors during task execution in scheduler
    DATABASE = auto()         # Database errors
    SANDBOX = auto()          # Errors in the sandbox environment
    UNKNOWN = auto()          # Unclassified errors

@dataclass
class ErrorContext:
    """Contextual information about an error"""
    operation: str                          # Operation being performed when error occurred
    input_data: Optional[Any] = None        # Input data related to the error (can be any type)
    file_path: Optional[str] = None         # Path to relevant file
    component: Optional[str] = None         # Component where error occurred (e.g., 'LLMProcessor', 'TaskScheduler')
    task_id: Optional[str] = None           # If error occurred within a scheduled task
    additional_info: Dict[str, Any] = field(default_factory=dict)  # Additional context information
    timestamp: float = field(default_factory=time.time)  # Error timestamp

    # Make input_data serializable for asdict
    def __post_init__(self):
         if self.input_data is not None:
              try:
                  # Attempt a simple representation, fallback to string
                  if isinstance(self.input_data, (str, int, float, bool, list, dict, tuple)):
                      pass # Already serializable
                  elif hasattr(self.input_data, '__dict__'):
                       # Basic representation for objects
                       self.input_data = str(self.input_data.__dict__)[:500] # Limit length
                  else:
                      self.input_data = str(self.input_data)[:500] # Limit length
              except Exception:
                  self.input_data = f"Non-serializable data of type {type(self.input_data).__name__}"


@dataclass
class EnhancedError:
    """Enhanced error object with detailed information"""
    message: str                            # Error message
    category: ErrorCategory = ErrorCategory.UNKNOWN  # Default category
    severity: ErrorSeverity = ErrorSeverity.ERROR    # Default severity
    exception: Optional[Exception] = None   # Original exception
    traceback: Optional[str] = None         # Exception traceback
    context: Optional[ErrorContext] = None  # Error context
    error_id: str = field(default_factory=lambda: hashlib.md5(str(time.time()).encode()).hexdigest())  # Unique ID
    timestamp: float = field(default_factory=time.time)  # Error timestamp

    def __post_init__(self):
        """Initialize traceback if exception is provided"""
        if self.exception and not self.traceback:
            try:
                 # Limit traceback depth if needed? For now, full traceback.
                 self.traceback = ''.join(traceback.format_exception(
                     type(self.exception),
                     self.exception,
                     self.exception.__traceback__
                 ))
            except Exception:
                 self.traceback = "Could not format traceback." # Fallback
        
        # Ensure error_id has a value even if default_factory fails
        if not self.error_id:
            self.error_id = hashlib.md5(str(time.time()).encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = {
            "error_id": self.error_id,
            "message": self.message,
            "category": self.category.name if isinstance(self.category, Enum) else str(self.category),
            "severity": self.severity.name if isinstance(self.severity, Enum) else str(self.severity),
            "timestamp": self.timestamp,
            "exception_type": type(self.exception).__name__ if self.exception else None
        }

        if self.traceback:
            # Limit traceback length for storage if necessary
            result["traceback"] = self.traceback[:5000] + '... (truncated)' if len(self.traceback or '') > 5000 else self.traceback

        if self.context:
            result["context"] = asdict(self.context)

        return result

    def to_json(self) -> str:
        """Convert to JSON string"""
        import json
        return json.dumps(self.to_dict(), indent=2)

    def log(self, logger_instance=None):
        """Log the error using a provided logger instance based on severity"""
        if logger_instance is None:
            logger_instance = logger
            
        log_message = f"[{self.error_id}] Category: {self.category.name}, Severity: {self.severity.name} - {self.message}"

        if self.context:
            comp = self.context.component or "unknown"
            op = self.context.operation or "unknown"
            task = self.context.task_id or "none"
            log_message += f" (Context: Component='{comp}', Operation='{op}', TaskID='{task}')"

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

        # Log message
        logger_instance.log(log_level, log_message)

        # Log traceback for ERROR and above
        if log_level >= logging.ERROR and self.traceback:
             # Log traceback separately for readability, maybe at DEBUG level or same level
             logger_instance.log(log_level, f"[{self.error_id}] Traceback:\n{self.traceback}")


if __name__ == "__main__":
    # Test error definitions
    try:
        # Create a deliberately failing function
        def test_failing_function(x):
            return x / 0
            
        # Create a test error with context
        test_failing_function(10)  # This will raise a ZeroDivisionError
        
    except Exception as e:
        # Create enhanced error
        context = ErrorContext(
            operation="Testing error definitions",
            input_data={"test_value": 10},
            component="ErrorDefinitionTest",
            additional_info={"test_phase": "initial"}
        )
        
        error = EnhancedError(
            message="Division by zero error in test function",
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.ERROR,
            exception=e,
            context=context
        )
        
        # Log the error
        error.log()
        
        # Print error details
        print("\nError Details:")
        print(f"Error ID: {error.error_id}")
        print(f"Message: {error.message}")
        print(f"Category: {error.category.name}")
        print(f"Severity: {error.severity.name}")
        print(f"Exception Type: {type(e).__name__}")
        print(f"Has Traceback: {'Yes' if error.traceback else 'No'}")
        
        # Convert to dictionary
        error_dict = error.to_dict()
        print("\nError Dictionary:")
        print(f"Keys: {', '.join(error_dict.keys())}")
        
        # Convert to JSON
        error_json = error.to_json()
        print(f"\nJSON Length: {len(error_json)} characters")
        
        print("\nTest completed successfully!")
