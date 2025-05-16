# Call Center Pipeline

An end-to-end audio processing pipeline for call center recordings, including speaker diarization, transcription, analysis, and visualization.

## Project Structure

```
📁 call‑center‑pipeline
├─ 📁 data
│  ├─ 📁 raw
│  │  ├─ 📁 audio_mp3          # original .mp3 calls
│  │  └─ calls.csv             # metadata sheet
│  └─ 📁 processed
│     ├─ 📁 audio_wav          # 16 kHz .wav output
│     └─ transcripts           # .json or .txt after ASR
├─ 📁 src
│  ├─ 00_ingest.py             # mp3→wav converter
│  ├─ 01_diarize.py            # speaker diarization
│  ├─ 02_transcribe.py         # ASR
│  ├─ 03_analyze.py            # NLP / sentiment / redaction
│  ├─ 04_store.py              # DB + vector storage
│  └─ 05_dashboard.py          # Streamlit UI
├─ 📁 models                   # downloaded checkpoints
├─ 📁 notebooks                # scratch Jupyter/Colab
├─ config.yaml                 # shared parameters
├─ environment.yml             # conda spec (GPU ready)
└─ .gitignore                  # exclude large files
```

## Setup

1. Install [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

2. Create the conda environment:
   ```bash
   conda env create -f environment.yml
   ```

3. Activate the environment:
   ```bash
   conda activate cc_pipeline
   ```

4. Place your MP3 audio files in the `data/raw/audio_mp3` directory

5. Update the metadata in `data/raw/calls.csv` (if applicable)

## Pipeline Execution

Run each stage of the pipeline in order:

1. Convert MP3 files to WAV format:
   ```bash
   python -m src.00_ingest
   ```

2. Perform speaker diarization:
   ```bash
   python -m src.01_diarize
   ```

3. Transcribe audio:
   ```bash
   python -m src.02_transcribe
   ```

4. Analyze transcriptions (sentiment, PII redaction):
   ```bash
   python -m src.03_analyze
   ```

5. Store results in database and vector index:
   ```bash
   python -m src.04_store
   ```

6. Launch the dashboard:
   ```bash
   python -m src.05_dashboard
   ```

## Configuration

All pipeline settings are stored in `config.yaml`. Update this file to customize:
- Audio processing parameters
- Model selections and parameters
- Database settings
- Vector storage configurations
- Dashboard preferences

## Requirements

- Python 3.10
- CUDA-compatible GPU (recommended for faster processing)
- FFmpeg (automatically installed via conda)

## License

[MIT License](LICENSE) 