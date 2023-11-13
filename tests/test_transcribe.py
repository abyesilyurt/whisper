import io
import os

import pytest
import torch

import whisper
from whisper.utils import write_vtt
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
    tokenizer = get_tokenizer(model.is_multilingual)
    tokenizer = get_tokenizer(model.is_multilingual, num_languages=model.num_languages)
    all_tokens = [t for s in result["segments"] for t in s["tokens"]]
    assert tokenizer.decode(all_tokens) == result["text"]
    assert tokenizer.decode_with_timestamps(all_tokens).startswith("<|0.00|>")

    timing_checked = False
    for segment in result["segments"]:
        for timing in segment["words"]:
            assert timing["start"] < timing["end"]
            if timing["word"].strip(" ,") == "Americans":
                assert timing["start"] <= 1.8
                assert timing["end"] >= 1.8
                timing_checked = True

    assert timing_checked


def segment_callback(segments):
    with io.StringIO() as file:
        writer: WriteVTT = get_writer("vtt", file.name)
        writer.write_result(segments, file)
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
