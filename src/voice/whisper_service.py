
"""
src/voice/whisper_service.py

Speech-to-text using OpenAI Whisper. Converts a voice recording into text
+ timestamped segments, which then feeds into the same pipeline as typed
tickets (src/graph/workflow.py).

Usage:
    from src.voice.whisper_service import transcribe
    result = transcribe("customer_complaint.mp3")
    print(result["text"])
"""

import os
import sys

sys.path.append(os.getcwd())
from src.utils.config import config

_model = None


def _load_model():
    global _model
    if _model is not None:
        return
    import whisper
    print(f"Loading Whisper ({config.WHISPER_MODEL_SIZE})...")
    _model = whisper.load_model(config.WHISPER_MODEL_SIZE)
    print("Whisper loaded.")


def transcribe(audio_path: str, language: str = None) -> dict:
    """
    Returns:
        {
            "text": "<full transcript>",
            "segments": [{"start": ..., "end": ..., "text": ...}, ...],
            "language": "<detected language code>"
        }
    """
    _load_model()

    result = _model.transcribe(audio_path, language=language)

    segments = [
        {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
        for s in result.get("segments", [])
    ]

    return {
        "text": result["text"].strip(),
        "segments": segments,
        "language": result.get("language"),
    }


if __name__ == "__main__":
    # Point this at a real audio file to test
    test_audio = "data/raw/audio/sample.mp3"
    if os.path.exists(test_audio):
        result = transcribe(test_audio)
        print(f"Detected language: {result['language']}")
        print(f"Transcript: {result['text']}\n")
        for seg in result["segments"]:
            print(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}")
    else:
        print(f"No test file found at {test_audio}. "
              f"Upload a short audio clip there to test transcription.")