"""
CHANGELOG
---------
v0.1 – 2025‑05‑13
  • auto‑generated skeleton by Cursor
  • Added multiprocessing for faster conversion
  • Added file logging for better tracking
  • Updated to use shared logging utility
Known issues:
  • none yet
Fix log:
  • —
"""

import os
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import soundfile as sf
from pydub import AudioSegment
from tqdm import tqdm

import logging

# Import from utils directory using relative path
import sys
sys.path.append(str(Path(__file__).parent))
from utils.logging_utils import setup_logging, get_script_logger

def load_config() -> Dict:
    """Load config from YAML or use defaults."""
    default_config = {
        "paths": {
            "raw_audio_mp3": "data/raw/audio_mp3",
            "processed_audio_wav": "data/processed/audio_wav",
        },
        "audio": {"target_sample_rate": 16000},
        "processing": {"max_workers": os.cpu_count() or 4},  # Default to CPU count
    }

    config_path = Path("config.yaml")
    if not config_path.exists():
        logging.warning("config.yaml not found, using defaults")
        return default_config

    with open(config_path) as f:
        user_config = yaml.safe_load(f)

    # Merge with defaults, allowing partial overrides
    if "paths" in user_config:
        default_config["paths"].update(user_config["paths"])
    if "audio" in user_config:
        default_config["audio"].update(user_config["audio"])
    if "processing" in user_config:
        default_config["processing"].update(user_config["processing"])

    return default_config


def collect_mp3s(input_dir: Path) -> List[Path]:
    """Find all MP3 files in the input directory."""
    return list(input_dir.glob("*.mp3"))


def convert_audio(
    input_path: Path, output_path: Path, target_sr: int
) -> Tuple[bool, Optional[float], Optional[str]]:
    """
    Convert MP3 to WAV using pydub + FFmpeg.
    Returns (success, duration_sec, error_msg).
    """
    try:
        # Load and convert
        audio = AudioSegment.from_mp3(input_path)
        audio = audio.set_channels(1).set_frame_rate(target_sr)
        audio.export(output_path, format="wav", parameters=["-acodec", "pcm_s16le"])

        # Get duration using soundfile (more accurate than pydub)
        with sf.SoundFile(output_path) as f:
            duration = len(f) / f.samplerate

        return True, duration, None

    except Exception as e:
        if "ffmpeg" in str(e).lower():
            raise RuntimeError("FFmpeg not found on PATH. Please install FFmpeg.")
        return False, None, str(e)


def process_file(args: Tuple[Path, Path, int]) -> Tuple[Path, bool, Optional[float], Optional[str]]:
    """Process a single file (worker function for multiprocessing)."""
    input_path, output_path, target_sr = args
    success, duration, error = convert_audio(input_path, output_path, target_sr)
    return input_path, success, duration, error


def generate_report(
    results: List[Tuple[Path, bool, Optional[float], Optional[str]]], output_path: Path
) -> None:
    """Write conversion report to CSV."""
    import csv

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "status", "duration_sec"])

        for input_path, success, duration, error in results:
            status = "converted" if success else f"error:{error}" if error else "skipped"
            writer.writerow([input_path.name, status, duration if duration else ""])


def main() -> None:
    """Main entry point."""
    # Setup logging first
    script_name = "ingest"
    log_file = setup_logging(script_name)
    logger = get_script_logger(script_name)
    
    config = load_config()
    input_dir = Path(config["paths"]["raw_audio_mp3"])
    output_dir = Path(config["paths"]["processed_audio_wav"])
    target_sr = config["audio"]["target_sample_rate"]
    max_workers = config["processing"]["max_workers"]

    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect MP3s
    mp3_files = collect_mp3s(input_dir)
    if not mp3_files:
        logger.warning(f"No MP3 files found in {input_dir}")
        return

    logger.info(f"Found {len(mp3_files)} MP3 files to process")

    # Prepare conversion tasks
    tasks = []
    results = []
    for mp3_path in mp3_files:
        wav_path = output_dir / f"{mp3_path.stem}.wav"
        
        # Skip if output exists
        if wav_path.exists():
            results.append((mp3_path, False, None, None))
            continue
            
        tasks.append((mp3_path, wav_path, target_sr))

    # Process files in parallel
    logger.info(f"Starting parallel conversion with {max_workers} workers")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file, task) for task in tasks]
        
        # Process results as they complete
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Converting MP3s"):
            try:
                result = future.result()
                results.append(result)
                
                # Log status
                input_path, success, duration, error = result
                if success:
                    logger.info(f"Converted {input_path.name} ({duration:.1f}s)")
                else:
                    logger.error(f"Failed {input_path.name}: {error}")
            except Exception as e:
                logger.error(f"Worker error: {str(e)}")

    # Generate report
    report_path = Path("data/processed/conversion_report.csv")
    generate_report(results, report_path)
    logger.info(f"Report written to {report_path}")

    # Summary
    converted = sum(1 for _, success, _, _ in results if success)
    skipped   = sum(1 for _, success, _, _ in results if not success and not _)
    failed    = sum(1 for _, success, _, _ in results if not success and _)

    # Enhanced summary logging
    logger.info("=== Conversion Summary ===")
    logger.info(f"Total files processed: {len(mp3_files)}")
    logger.info(f"Successfully converted: {converted}")
    logger.info(f"Skipped (already exist): {skipped}")
    logger.info(f"Failed: {failed}")
    if failed > 0:
        logger.warning("Some files failed conversion. Check the log file for details.")
    logger.info(f"Detailed log available at: {log_file}")
    logger.info("========================")

if __name__ == "__main__":
    main()

# Quick test:
#   python -m src.00_ingest
# Watch nvidia‑smi – conversion runs on CPU but verifies env isolation 