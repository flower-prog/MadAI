import logging
import os
import sys

def setup_logging():
    """
    Configures the logging system based on environment variables.
    
    Env vars:
      LOG_LEVEL: DEBUG, INFO, WARNING, ERROR, CRITICAL (default: INFO)
      LOG_FORMAT: "json" or "text" (default: text)
    """
    level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    
    # Define a custom format that highlights key info if running in a terminal
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Basic configuration
    logging.basicConfig(
        level=level,
        format=fmt,
        stream=sys.stdout
    )
    
    # Adjust specific noisy loggers if needed
    if level == logging.INFO:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("watchfiles").setLevel(logging.WARNING)

    logging.info(f"Logging configured with level: {level_str}")

def get_token_usage_enabled() -> bool:
    """Returns True if token usage logging is enabled via environment variable."""
    val = os.getenv("BIOINFO_LOG_LLM_USAGE", "1").strip().lower()
    return val in ("1", "true", "yes", "on")

def get_agent_logger():
    """
    Returns a dedicated logger for agent interactions (trace).
    This logger writes to a separate file.
    Default: 'logs/agent_trace.log' (relative to project root, not cwd)
    Env Var Override: 'AGENT_TRACE_LOG_FILE'
    
    IMPORTANT: For multi-process environments (like BixBench), each process should
    set AGENT_TRACE_LOG_FILE before calling this function to ensure separate log files.
    """
    logger = logging.getLogger("agent_trace")
    logger.setLevel(logging.INFO)
    
    # Check for environment variable override (supports per-process log files)
    log_file_path = os.getenv("AGENT_TRACE_LOG_FILE")
    
    if not log_file_path:
        # Use project root directory (where this file is located) instead of cwd
        # This ensures logs go to the same place regardless of where the script is run from
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        log_dir = os.path.join(project_root, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, "agent_trace.log")
    else:
        # Ensure the directory exists for the custom path
        os.makedirs(os.path.dirname(os.path.abspath(log_file_path)), exist_ok=True)
    
    # For multi-process support: check if handler already exists for this specific file
    # If log file path changed (e.g., different process), remove old handlers and add new one
    existing_handler = None
    for handler in logger.handlers[:]:  # Copy list to avoid modification during iteration
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == os.path.abspath(log_file_path):
            existing_handler = handler
            break
        elif isinstance(handler, logging.FileHandler):
            # Remove handler for different file (process-specific log file changed)
            logger.removeHandler(handler)
            handler.close()
    
    # Only add handler if it doesn't exist for this specific file
    if existing_handler is None:
        handler = logging.FileHandler(log_file_path, encoding='utf-8')
        formatter = logging.Formatter("%(asctime)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    # Prevent propagation to root logger to avoid double logging in backend.log
    logger.propagate = False
        
    return logger
