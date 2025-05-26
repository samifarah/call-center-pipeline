"""
Clean metadata CSV to match available audio files.

This script:
1. Reads the metadata CSV file
2. Identifies which audio files exist in the raw data directory
3. Creates a new CSV with only the rows corresponding to existing audio files
4. Generates a report of the cleaning process
"""

import sys
import logging
from pathlib import Path

# Import from utils directory using relative path
sys.path.append(str(Path(__file__).parent))
from utils.logging_utils import setup_logging, get_script_logger

import pandas as pd
from typing import Set, Tuple


def get_existing_audio_files(data_dir: Path) -> Set[str]:
    """
    Get set of audio file IDs that exist in the data directory.
    
    Args:
        data_dir: Path to directory containing audio files
        
    Returns:
        Set of audio file IDs (without extension)
    """
    # Get all MP3 files
    mp3_files = {f.stem for f in data_dir.glob("*.mp3")}
    
    # Get all WAV files
    wav_files = {f.stem for f in data_dir.parent.joinpath("processed/audio_wav").glob("*.wav")}
    
    # Return union of both sets
    return mp3_files | wav_files


def clean_metadata(
    metadata_path: Path,
    output_path: Path,
    existing_files: Set[str],
    logger: logging.Logger
) -> Tuple[int, int]:
    """
    Clean metadata CSV to keep only rows for existing audio files.
    
    Args:
        metadata_path: Path to input metadata CSV
        output_path: Path to save cleaned CSV
        existing_files: Set of existing audio file IDs
        logger: Logger instance
        
    Returns:
        Tuple of (total_rows, kept_rows)
    """
    # Read metadata
    logger.info(f"Reading metadata from {metadata_path}")
    df = pd.read_csv(metadata_path)
    total_rows = len(df)
    
    # Assuming the audio file ID is in a column named 'audio_id' or similar
    # You may need to adjust this based on your actual CSV structure
    id_column = 'Call ID'
    
    if id_column not in df.columns:
        raise ValueError(f"Column '{id_column}' not found in metadata CSV")
    
    # Filter rows
    df_cleaned = df[df[id_column].isin(existing_files)]
    kept_rows = len(df_cleaned)
    
    # Save cleaned data
    logger.info(f"Saving cleaned metadata to {output_path}")
    df_cleaned.to_csv(output_path, index=False)
    
    return total_rows, kept_rows


def main():
    """Main entry point."""
    # Setup logging
    script_name = "clean_data"
    log_file = setup_logging(script_name)
    logger = get_script_logger(script_name)
    
    try:
        # Define paths
        base_dir = Path("data")
        metadata_path = base_dir / "raw" / "calls.csv"
        output_path = base_dir / "processed" / "cleaned_metadata.csv"
        audio_dir = base_dir / "raw" / "audio_mp3"
        
        # Validate inputs
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        if not audio_dir.exists():
            raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
        
        # Get existing audio files
        logger.info("Scanning for existing audio files...")
        existing_files = get_existing_audio_files(audio_dir)
        logger.info(f"Found {len(existing_files)} existing audio files")
        
        # Clean metadata
        total_rows, kept_rows = clean_metadata(
            metadata_path,
            output_path,
            existing_files,
            logger
        )
        
        # Log summary
        logger.info("=== Cleaning Summary ===")
        logger.info(f"Total rows in metadata: {total_rows}")
        logger.info(f"Rows kept: {kept_rows}")
        logger.info(f"Rows removed: {total_rows - kept_rows}")
        logger.info(f"Removal rate: {((total_rows - kept_rows) / total_rows * 100):.1f}%")
        logger.info(f"Detailed log available at: {log_file}")
        logger.info("======================")
        
    except Exception as e:
        logger.error(f"Error during cleaning: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main() 