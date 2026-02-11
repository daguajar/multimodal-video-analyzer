# 🎥 Multimodal Video Analyzer

A powerful multimodal analysis tool that processes **local videos or
YouTube content** using state-of-the-art Hugging Face models and other
AI frameworks.

This project extracts insights from **three modalities --- text, audio,
and video ---** to deliver a comprehensive understanding of sentiment,
emotions, topics, speaker characteristics, and more.

The analyzer works by splitting media into configurable chunks and
generating a structured JSON output with both per-segment and global
summaries.

------------------------------------------------------------------------

## 🚀 Features

-   ✅ Automatic Speech Recognition (ASR) with Whisper
-   ✅ Audio emotion recognition
-   ✅ Text emotion analysis
-   ✅ Zero-shot topic classification
-   ✅ Age and gender detection from audio
-   ✅ Facial analysis (age & gender) from video frames
-   ✅ Speaker diarization (who spoke and when)
-   ✅ Unified Sentiment Score (USS) combining:
    -   VADER
    -   pysentimiento
    -   Audio prosody
-   ✅ Parallel processing for faster execution
-   ✅ Chunk-based analysis for long media files
-   ✅ JSON output ready for downstream analytics

------------------------------------------------------------------------

## 🧠 Architecture Overview

    Video / YouTube
          ↓
     Chunk Segmentation
          ↓
     ┌─────────────┬─────────────┬─────────────┐
     │    Audio    │     Text    │    Video    │
     └─────────────┴─────────────┴─────────────┘
          ↓
     Multimodal Fusion → Unified Sentiment Score
          ↓
     Structured JSON Output

------------------------------------------------------------------------

## 📦 Installation

> Python **3.13+ recommended**

Clone the repository:

``` bash
git clone https://github.com/daguajar/multimodal-video-analyzer.git
```

Install dependencies:

- Using GPU

``` bash
pip install -r requirements_gpu.txt
```

- Using CPU

``` bash
pip install -r requirements_cpu.txt
```

### System Requirements

Some libraries may require additional setup:

-   **FFmpeg** (required by moviepy, see [Getting started with MoviePy](https://zulko.github.io/moviepy/getting_started/install.html))
-   **PyTorch** with GPU support recommended for faster inference
-   **Hugging Face token** (needed for speaker diarization)

------------------------------------------------------------------------

## ▶️ Usage

### Analyze a Local Video

``` bash
python multimodal_chunked_analyzer.py   --input path/to/video.mp4   --hf_token <YOUR_HF_TOKEN>
```

------------------------------------------------------------------------

### Analyze a YouTube Video

``` bash
python multimodal_chunked_analyzer.py   --url "https://youtube.com/video_tag"   --hf_token <YOUR_HF_TOKEN>
```

------------------------------------------------------------------------

## ⚙️ Parameters

  Argument            Description                          Default
  ------------------- ------------------------------------ -------------------
  `--input`           Path to local media file             ---
  `--url`             YouTube video URL                    ---
  `--output_folder`   Base output directory                `output_analysis`
  `--segment`         Chunk duration in seconds            `30`
  `--whisper_model`   Whisper model size                   `large-v3`
  `--language`        Force transcription language         Auto
  `--max_workers`     Parallel threads                     `6`
  `--hf_token`        Hugging Face token for diarization   ---

> `--input` and `--url` are mutually exclusive.

------------------------------------------------------------------------

## 📊 Output

The analyzer generates:

    outputs/
     └── YOUR_OUTPUT_FOLDER
       ├── tmp_chunks/
       └── output.json

### Example JSON Structure

``` json
{
  "source": "video.mp4",
  "segments": 12,
  "duration_seconds": 360,
  "total_processing_time_seconds": 145.2,
  "chunks": [
    {
      "segment": "0_30",
      "time": "0:00:00-0:00:30",
      "transcription": "...",
      "audio_emotions": {},
      "text_emotions": {},
      "topics": {},
      "speaker_segments": [],
      "age_gender_audio": {},
      "video_detections": [],
      "USS_details": {},
      "USS_final": 7.8
    }
  ]
}
```

------------------------------------------------------------------------

## 🧩 Models & Technologies

-   **Whisper** --- Speech recognition
-   **Hugging Face Transformers** --- Emotion, topic, and audio models
-   **pyannote.audio** --- Speaker diarization
-   **DeepFace** --- Facial analysis
-   **VADER + pysentimiento** --- Sentiment scoring
-   **librosa** --- Prosodic feature extraction
-   **moviepy** --- Media handling

------------------------------------------------------------------------

## ⚡ Performance Tips

-   Use a **GPU** whenever possible.
-   Reduce Whisper model size (`medium` or `small`) for faster runs.
-   Increase chunk size to minimize overhead.
-   Adjust `max_workers` based on CPU capacity.

------------------------------------------------------------------------

## ⚠️ Limitations

-   Facial analysis depends on frame quality and lighting.
-   Age/gender predictions are probabilistic and may be inaccurate.
-   Speaker diarization requires a valid Hugging Face token.
-   Large models demand significant RAM/VRAM.
