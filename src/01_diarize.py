"""
# Note: Before running, ensure environment.yml includes:
#   - pip:
#       - nemo_toolkit[asr]==2.3.1

CHANGELOG
---------
v0.5 – 2025-05-30
  • Added GPU memory logging
  • Increased num_workers to 4 and batch_size to 4 for parallel preprocessing and inference
Known issues:
  • TBD – evaluate DER on female-female calls
Fix log:
  • —
"""

import os
import json
import csv
import torch
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import NeuralDiarizer

from utils.logging_utils import setup_logging
import logging

# ─── Logging setup ──────────────────────────────────────────────────────────────
setup_logging("diarize", log_dir="logs")
logger = logging.getLogger("diarize")

# ─── Default config ─────────────────────────────────────────────────────────────
DEFAULT_CFG = {
    "paths": {
        "wavs": "data/processed/audio_wav",
        "diarization": "data/processed/diarization"
    },
    "diarization": {
        "msdd_model": "diar_msdd_telephonic",
        "min_gap_sec": 0.25
    },
    "performance": {
        "num_workers": 4,
        "batch_size": 4
    }
}

def load_config(defaults: Dict) -> Dict:
    import yaml
    cfg = defaults.copy()
    path = Path("config.yaml")
    if not path.exists():
        logger.warning("config.yaml not found, using defaults")
        return cfg
    with open(path) as f:
        user_cfg = yaml.safe_load(f) or {}
    cfg["paths"].update(user_cfg.get("paths", {}))
    cfg["diarization"].update(user_cfg.get("diarization", {}))
    cfg.setdefault("performance", {}).update(user_cfg.get("performance", {}))
    return cfg

def merge_segments(segments: List[Dict], min_gap: float) -> List[Dict]:
    if not segments:
        return []
    by_spk = {}
    for seg in segments:
        by_spk.setdefault(seg["label"], []).append(seg)
    merged = []
    for spk, segs in by_spk.items():
        segs.sort(key=lambda x: x["start"])
        curr = segs[0]
        for nxt in segs[1:]:
            if nxt["start"] - curr["end"] <= min_gap:
                curr["end"] = max(curr["end"], nxt["end"])
            else:
                merged.append(curr)
                curr = nxt
        merged.append(curr)
    merged.sort(key=lambda x: x["start"])
    return merged

def write_outputs(wav_path: Path, segments: List[Dict], cfg: Dict) -> None:
    base = wav_path.stem
    out = Path(cfg["paths"]["diarization"])
    out.mkdir(parents=True, exist_ok=True)

    # RTTM
    with open(out / f"{base}.rttm", "w") as f:
        for seg in segments:
            dur = seg["end"] - seg["start"]
            f.write(
                f"SPEAKER {base} 1 {seg['start']:.3f} {dur:.3f} "
                f"<NA> <NA> {seg['label']} <NA> <NA>\n"
            )

    # JSON
    info = {
        "speakers": sorted({s["label"] for s in segments}),
        "segments": segments
    }
    with open(out / f"{base}.json", "w") as f:
        json.dump(info, f, indent=2)

    # Audacity labels
    with open(out / f"{base}_labels.txt", "w") as f:
        for seg in segments:
            f.write(f"{seg['start']:.3f}\t{seg['end']:.3f}\t{seg['label']}\n")

def diarize_file(wav_path: Path, cfg: Dict, model: NeuralDiarizer, device: str) -> Tuple[bool, str]:
    """Run NeuralDiarizer on one file; handle GPU/CPU and OOM fallback."""
    try:
        num_workers = cfg["performance"]["num_workers"]
        batch_size  = cfg["performance"]["batch_size"]
        logger.info(f"Diarizing {wav_path.name} on device: {device}")
        if device == "cuda":
            mem_alloc = torch.cuda.memory_allocated() >> 20
            mem_total = torch.cuda.get_device_properties(0).total_memory >> 20
            logger.info(f"  → CUDA device #0, allocated {mem_alloc}MB / {mem_total}MB")
        ann = model(
            wav_path.as_posix(),
            num_workers=num_workers,
            batch_size=batch_size
        )
        raw: List[Dict] = []
        for segment, _, speaker in ann.itertracks(yield_label=True):
            raw.append({
                "start": segment.start,
                "end":   segment.end,
                "label": speaker
            })
        merged = merge_segments(raw, cfg["diarization"]["min_gap_sec"])
        write_outputs(wav_path, merged, cfg)
        return True, f"{len(merged)} segments"
    except RuntimeError as e:
        msg = str(e)
        if "CUDA out of memory" in msg and device == "cuda":
            logger.warning(f"OOM on GPU for {wav_path.name}, retrying on CPU")
            torch.cuda.empty_cache()
            return diarize_file(wav_path, cfg, model.cpu(), "cpu")
        return False, msg
    except Exception as e:
        return False, str(e)

def main() -> None:
    cfg     = load_config(DEFAULT_CFG)
    wav_dir = Path(cfg["paths"]["wavs"])
    out_dir = Path(cfg["paths"]["diarization"])
    out_dir.mkdir(parents=True, exist_ok=True)
    force   = os.getenv("DIARIZE_FORCE", "0") == "1"

    # ─── Determine device and load model once ─────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Initializing NeuralDiarizer on {device}")
    model = NeuralDiarizer.from_pretrained(
        model_name=cfg["diarization"]["msdd_model"]
    ).to(device)

    # ─── Process all WAV files ────────────────────────────────────────────────
    files = sorted(wav_dir.glob("*.wav"))

    report = []
    for wav in tqdm(files, desc="Diarizing"):
        target = out_dir / f"{wav.stem}.rttm"
        if target.exists() and not force:
            report.append((wav.name, "skipped", "exists"))
            continue
        ok, msg = diarize_file(wav, cfg, model, device)
        report.append((wav.name, "ok" if ok else "error", msg))

    # write summary CSV
    with open(out_dir / "diarization_report.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "status", "detail"])
        writer.writerows(report)

    logger.info(f"Finished diarization for {len(files)} files")

if __name__ == "__main__":
    main()
