# smoke_test_pipeline.py
import ffmpeg           # from FFmpeg‑python wrapper
from pydub import AudioSegment
import whisper
from pyannote.audio import Pipeline
import faiss
import presidiopy

# Simple usage of ffmpeg to verify it's working
ffmpeg_version = ffmpeg.probe(None, show_format=None, show_streams=None, count_frames=None, count_packets=None, show_entries=None, show_error=True)

print("All imports succeeded 🎉")
