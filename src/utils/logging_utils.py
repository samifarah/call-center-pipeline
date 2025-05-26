"""
Logging utilities for the call center pipeline.
Provides consistent logging configuration across all scripts.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logging(
    script_name: str,
    log_dir: str = "logs",
    log_level: int = logging.INFO,
    console: bool = True
) -> Path:
    """
    Configure logging for a script with both file and console handlers.
    
    Args:
        script_name: Name of the script (e.g., 'clean_data', 'ingest')
        log_dir: Directory to store log files
        log_level: Logging level (default: INFO)
        console: Whether to also log to console (default: True)
    
    Returns:
        Path to the log file
    """
    # Create logs directory if it doesn't exist
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    # Create a timestamp for the log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{script_name}_{timestamp}.log"
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # Create handlers
    handlers = []
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(file_formatter)
    handlers.append(file_handler)
    
    # Console handler (if requested)
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        handlers.append(console_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=handlers
    )
    
    # Create logger for this script
    logger = logging.getLogger(script_name)
    logger.info(f"Logging initialized for {script_name}. Log file: {log_file}")
    
    return log_file


def get_script_logger(script_name: str) -> logging.Logger:
    """
    Get a logger instance for a specific script.
    Use this after setup_logging() has been called.
    
    Args:
        script_name: Name of the script
        
    Returns:
        Logger instance
    """
    return logging.getLogger(script_name) 