"""
src/voice/tts_service.py

Text-to-speech using gTTS (Google Text-to-Speech). Converts the AI's
drafted answer into a playable audio file for the "voice in -> voice out"
demo loop.

Usage:
    from src.voice.tts_service import synthesize_speech
    path = synthesize_speech("Your refund will be processed in 5-7 days.")
"""

import os
import sys
import uuid

sys.path.append(os.getcwd())

OUTPUT_DIR = "data/processed/tts_output"


def synthesize_speech(text: str, lang: str = "en", out_dir: str = None) -> str:
    """
    Converts text to speech and saves it as an mp3 file.
    Returns the file path. Returns None if text is empty/None (e.g. an
    escalated ticket with no draft answer to speak).
    """
    if not text:
        return None

    from gtts import gTTS

    out_dir = out_dir or OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    filename = f"response_{uuid.uuid4().hex[:8]}.mp3"
    filepath = os.path.join(out_dir, filename)

    try:
        tts = gTTS(text=text, lang=lang)
        tts.save(filepath)
        return filepath
    except Exception as e:
        # gTTS needs internet access; fail gracefully so the dashboard
        # still shows the text answer even if TTS is unavailable
        print(f"TTS generation failed: {e}")
        return None


if __name__ == "__main__":
    test_text = "Your refund will be processed within 5 to 7 business days."
    path = synthesize_speech(test_text)
    print(f"Audio saved to: {path}" if path else "TTS failed.")
