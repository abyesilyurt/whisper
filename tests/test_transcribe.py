import io
import os

import pytest
import torch

import whisper
from whisper.utils import write_vtt


def test_transcribe():
    model_name = "base"
    model = whisper.load_model(model_name)
    audio_path = os.path.join(os.path.dirname(__file__), "jfk.flac")

    language = "en" if model_name.endswith(".en") else None
    result = model.transcribe(
        audio_path, language=language, temperature=0.0, word_timestamps=True
    )
    assert result["language"] == "en"
    assert result["text"] == "".join([s["text"] for s in result["segments"]])

    transcription = result["text"].lower()
    assert "my fellow americans" in transcription
    assert "your country" in transcription
    assert "do for you" in transcription


def segment_callback(segments):
    with io.StringIO() as file:
        write_vtt(segments, file)
        vtt = file.getvalue()
        assert vtt


def test_transcribe_callback():
    model = whisper.load_model("tiny.en")
    audio_path = os.path.join(os.path.dirname(__file__), "jfk.flac")

    result = model.transcribe(
        audio_path, language="en", temperature=0.0, segment_callback=segment_callback
    )
    assert result["language"] == "en"

    transcription = result["text"].lower()
    assert "my fellow americans" in transcription
    assert "your country" in transcription
    assert "do for you" in transcription
    timing_checked = False
    for segment in result["segments"]:
        for timing in segment["words"]:
            assert timing["start"] < timing["end"]
            if timing["word"].strip(" ,") == "Americans":
                assert timing["start"] <= 1.8
                assert timing["end"] >= 1.8
                print(timing)
                timing_checked = True

    assert timing_checked
