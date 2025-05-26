"""
CHANGELOG
---------
v0.1 – 2025‑05‑18
  • Initial release
  • GPU-first diarization with CPU fallback
  • Stable speaker mapping per call
  • Segment merging for same speaker (post-processing)
  • Multiple output formats (RTTM, JSON, Audacity labels)
  • Added .env support for HF_TOKEN
  • Improved model access error handling
  • Fixed pipeline parameters (v7) - proper post-processing
Known issues:
  • none yet
Fix log:
  • —
"""

import os
import json
import yaml
import torch
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from pyannote.core import Segment, Timeline, Annotation

# Import from utils directory using relative path
import sys
sys.path.append(str(Path(__file__).parent))
from utils.logging_utils import setup_logging, get_script_logger

# Load environment variables from .env file
load_dotenv()


def load_config() -> Dict:
    """Load config from YAML or use defaults."""
    default_config = {
        "paths": {
            "wavs": "data/processed/audio_wav",
            "diarization": "data/processed/diarization",
        },
        "diarization": {
            "model_name": "pyannote/speaker-diarization-3.1",
            "max_num_speakers": 3,
            "min_gap_sec": 0.25,
            "max_files": 10  # Process only first N files
        }
    }

    config_path = Path("config.yaml")
    if not config_path.exists():
        logger.warning("config.yaml not found, using defaults")
        return default_config

    with open(config_path) as f:
        user_config = yaml.safe_load(f)

    # Merge with defaults
    if "paths" in user_config:
        default_config["paths"].update(user_config["paths"])
    if "diarization" in user_config:
        default_config["diarization"].update(user_config["diarization"])

    return default_config


def merge_segments(annotation: Annotation, min_gap: float) -> Annotation:
    """Merge segments from same speaker if gap between consecutive segments
    of the same speaker is smaller than ``min_gap`` seconds.

    Args:
        annotation: Original diarization result.
        min_gap: Maximum gap (in seconds) between two consecutive segments of the
            same speaker for them to be merged.

    Returns
    -------
        Annotation with small intra-speaker gaps removed (segments merged).
    """
    merged = Annotation(uri=annotation.uri)

    # Collect segments per speaker.
    segments_by_speaker = {}
    for segment, _, label in annotation.itertracks(yield_label=True):
        segments_by_speaker.setdefault(label, []).append(segment)

    for speaker, segments in segments_by_speaker.items():
        # Sort segments chronologically
        segments.sort(key=lambda s: s.start)
        if not segments:
            continue

        current_seg = segments[0]
        for seg in segments[1:]:
            # If the gap is smaller than min_gap, extend current segment
            if seg.start - current_seg.end <= min_gap:
                current_seg = Segment(current_seg.start, max(current_seg.end, seg.end))
            else:
                # Add finished segment and start new one
                merged[current_seg] = speaker
                current_seg = seg
        # Add last segment for this speaker
        merged[current_seg] = speaker

    return merged


def write_rttm(annotation: Annotation, output_path: Path, wav_path: Path) -> None:
    """Write diarization results in RTTM format."""
    with open(output_path, "w") as f:
        for segment, track, speaker in annotation.itertracks(yield_label=True):
            # RTTM format: SPEAKER file 1 start duration <NA> <NA> speaker <NA> <NA>
            f.write(f"SPEAKER {wav_path.stem} 1 {segment.start:.3f} {segment.duration:.3f} "
                   f"<NA> <NA> {speaker} <NA> <NA>\n")


def write_json(annotation: Annotation, output_path: Path) -> None:
    """Write diarization results in JSON format."""
    result = {
        "speakers": list(annotation.labels()),
        "segments": []
    }
    
    for segment, track, speaker in annotation.itertracks(yield_label=True):
        result["segments"].append({
            "speaker": speaker,
            "start": segment.start,
            "end": segment.end,
            "duration": segment.duration
        })
    
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)


def write_audacity_labels(annotation: Annotation, output_path: Path) -> None:
    """Write diarization results in Audacity label format."""
    with open(output_path, "w") as f:
        for segment, track, speaker in annotation.itertracks(yield_label=True):
            # Audacity format: start end label
            f.write(f"{segment.start:.3f}\t{segment.end:.3f}\t{speaker}\n")


def process_file(
    wav_path: Path,
    output_dir: Path,
    pipeline: Pipeline,
    max_speakers: int,
    min_gap: float,
    force: bool,
    logger: logging.Logger
) -> bool:
    """Process a single WAV file through the diarization pipeline."""
    # Setup output paths
    base_name = wav_path.stem
    rttm_path = output_dir / f"{base_name}.rttm"
    json_path = output_dir / f"{base_name}.json"
    labels_path = output_dir / f"{base_name}_labels.txt"
    
    # Skip if all outputs exist and not forcing
    if not force and all(p.exists() for p in [rttm_path, json_path, labels_path]):
        logger.info(f"Skipping {base_name} - outputs exist")
        return True
    
    try:
        # Run diarization with speaker count parameter
        logger.info(f"Processing {base_name} with {max_speakers} speakers")
        diarization = pipeline(
            wav_path,
            num_speakers=max_speakers  # number of speakers to detect
        )
        
        # Post-process: merge small gaps between same speaker
        logger.info(f"Merging gaps < {min_gap}s for same speaker")
        diarization = merge_segments(diarization, min_gap)
        
        # Write outputs
        write_rttm(diarization, rttm_path, wav_path)
        write_json(diarization, json_path)
        write_audacity_labels(diarization, labels_path)
        
        logger.info(f"Completed {base_name}")
        return True
        
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            logger.warning(f"GPU OOM for {base_name}, falling back to CPU")
            try:
                # Move pipeline to CPU
                pipeline = pipeline.to(torch.device("cpu"))
                # Retry processing with CPU
                return process_file(wav_path, output_dir, pipeline, max_speakers, min_gap, force, logger)
            except Exception as cpu_e:
                logger.error(f"CPU processing failed for {base_name}: {str(cpu_e)}")
                return False
        else:
            logger.error(f"Error processing {base_name}: {str(e)}")
            return False


def main():
    """Main entry point."""
    # Setup logging
    script_name = "diarize"
    log_file = setup_logging(script_name)
    logger = get_script_logger(script_name)
    
    try:
        # Verify HF_TOKEN is set
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            logger.error("HF_TOKEN not found in environment variables. Please set it in .env file")
            logger.error("Get your token from https://huggingface.co/settings/tokens")
            return

        # Load config
        config = load_config()
        wav_dir = Path(config["paths"]["wavs"])
        output_dir = Path(config["paths"]["diarization"])
        model_name = config["diarization"]["model_name"]
        max_speakers = config["diarization"]["max_num_speakers"]
        min_gap = config["diarization"]["min_gap_sec"]
        max_files = config["diarization"]["max_files"]
        
        # Check force flag
        force = bool(os.environ.get("DIARIZE_FORCE", False))
        if force:
            logger.info("Force flag set - will overwrite existing outputs")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize pipeline
        logger.info(f"Loading diarization model: {model_name}")
        try:
            pipeline = Pipeline.from_pretrained(
                model_name,
                use_auth_token=hf_token
            )
            if pipeline is None:
                raise ValueError("Failed to load pipeline - model access may be restricted")
                
        except Exception as e:
            if "gated" in str(e).lower() or "access" in str(e).lower():
                logger.error("Model access is restricted. Please:")
                logger.error("1. Visit https://huggingface.co/pyannote/speaker-diarization-3.1")
                logger.error("2. Click 'Access repository' and accept the terms")
                logger.error("3. Make sure your HF_TOKEN is correct in .env file")
            else:
                logger.error(f"Error loading model: {str(e)}")
            return
        
        # Move to GPU if available
        if torch.cuda.is_available():
            logger.info("Using GPU for diarization")
            pipeline = pipeline.to(torch.device("cuda"))
        else:
            logger.info("GPU not available, using CPU")
            pipeline = pipeline.to(torch.device("cpu"))
        
        # Process WAV files
        wav_files = list(wav_dir.glob("*.wav"))
        if not wav_files:
            logger.warning(f"No WAV files found in {wav_dir}")
            return
        
        # Limit to first N files
        total_files = len(wav_files)
        wav_files = wav_files[:max_files]
        
        logger.info(f"Found {total_files} WAV files, processing first {len(wav_files)}")
        if total_files > max_files:
            logger.info(f"Skipping {total_files - max_files} remaining files")
        
        # Process files
        success_count = 0
        for wav_path in wav_files:
            if process_file(wav_path, output_dir, pipeline, max_speakers, min_gap, force, logger):
                success_count += 1
        
        # Log summary
        logger.info("=== Diarization Summary ===")
        logger.info(f"Total files available: {total_files}")
        logger.info(f"Files processed: {len(wav_files)}")
        logger.info(f"Successfully processed: {success_count}")
        logger.info(f"Failed: {len(wav_files) - success_count}")
        if total_files > max_files:
            logger.info(f"Skipped: {total_files - max_files} (due to max_files limit)")
        if force:
            logger.info("Note: Force flag was set - existing files were overwritten")
        logger.info(f"Detailed log available at: {log_file}")
        logger.info("=========================")
        
    except Exception as e:
        logger.error(f"Error during diarization: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main() 