import io
import os

import pytest
import torch

import whisper
from whisper.tokenizer import get_tokenizer
from whisper.utils import WriteVTT, get_writer


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

def write_vtt(segments, file):
    for segment in segments:
        start = segment["start"]
        end = segment["end"]
        text = segment["text"]
        print(f"{start} --> {end}\n{text}\n", file=file, flush=True)

def segment_callback(segments):
    with io.StringIO() as file:
        # writer: WriteVTT = get_writer("vtt", file.name)
        # writer.write_result(segments, file)
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
