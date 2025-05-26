"""
Call Center Pipeline - Module Initialization

This module ensures the required directory structure exists and initializes the package.
"""

import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create required directories
def ensure_directories():
    """Create all required directories for the pipeline if they don't exist."""
    
    # Get the project root directory
    root_dir = Path(__file__).parent.parent.absolute()
    
    # Define directories to create
    directories = [
        root_dir / "data" / "raw" / "audio_mp3",
        root_dir / "data" / "processed" / "audio_wav",
        root_dir / "data" / "processed" / "transcripts",
        root_dir / "models",
        root_dir / "notebooks",
        root_dir / "logs"  # Add logs directory
    ]
    
    # Create directories
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"Directory created or verified: {directory}")
    
    logger.info("Directory structure verified")

# Create directories when package is imported
ensure_directories() 