from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from voiceprint.audio import (
    decode_audio,
    detect_audio_format,
    inspect_audio,
    split_windows,
    trim_silence,
)


def _audio_file(tmp_path: Path, suffix: str = ".mp3") -> Path:
    source = tmp_path / f"sample{suffix}"
    source.touch()
    return source


@pytest.mark.parametrize(
    ("suffix", "expected_format"),
    [
        (".mp3", "mp3"),
        (".wav", "wav"),
        (".flac", "flac"),
        (".aac", "aac"),
        (".MP3", "mp3"),
        (".AAC", "aac"),
    ],
)
def test_detect_audio_format_is_case_insensitive(
    suffix: str,
    expected_format: str,
) -> None:
    assert detect_audio_format(Path(f"recording{suffix}")) == expected_format


@pytest.mark.parametrize(
    "name",
    ["recording.ogg", "recording.m4a", "recording", "recording.mp3.tmp"],
)
def test_detect_audio_format_rejects_unsupported_suffix(name: str) -> None:
    assert detect_audio_format(Path(name)) is None


@pytest.mark.parametrize(
    ("suffix", "expected_parser"),
    [
        (".mp3", "mpegaudioparse"),
        (".wav", "wavparse"),
        (".flac", "flacparse"),
        (".aac", "aacparse"),
    ],
)
def test_decode_audio_selects_parser_from_suffix(
    tmp_path: Path,
    suffix: str,
    expected_parser: str,
) -> None:
    source = _audio_file(tmp_path, suffix)
    expected = np.array([-0.5, 0.0, 0.75], dtype="<f4")
    completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout=expected.tobytes(), stderr=b""
    )

    with patch("voiceprint.audio.subprocess.run", return_value=completed) as run:
        samples = decode_audio(source, sample_rate=16000)

    assert samples is not None
    np.testing.assert_array_equal(samples, expected.astype(np.float32))
    command = run.call_args.args[0]
    source_index = command.index(f"location={source.resolve()}")
    assert command[source_index + 1 : source_index + 5] == [
        "!",
        expected_parser,
        "!",
        "decodebin",
    ]


def test_decode_audio_returns_mono_float32_pcm(tmp_path: Path) -> None:
    source = _audio_file(tmp_path)
    expected = np.array([-0.5, 0.0, 0.75], dtype="<f4")
    completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout=expected.tobytes(), stderr=b""
    )

    with patch("voiceprint.audio.subprocess.run", return_value=completed) as run:
        samples = decode_audio(source, sample_rate=16000)

    assert samples is not None
    np.testing.assert_array_equal(samples, expected.astype(np.float32))
    assert samples.dtype == np.float32
    command = run.call_args.args[0]
    assert command[0] == "gst-launch-1.0"
    assert f"location={source.resolve()}" in command
    assert "audio/x-raw,format=F32LE,layout=interleaved,channels=1,rate=16000" in command
    assert run.call_args.kwargs["shell"] is False


@pytest.mark.parametrize(
    "name",
    ["sample.ogg", "sample.m4a", "sample", "sample.mp3.tmp"],
)
def test_decode_audio_rejects_unsupported_suffix_without_running_gstreamer(
    tmp_path: Path,
    name: str,
) -> None:
    source = tmp_path / name
    source.touch()

    with patch("voiceprint.audio.subprocess.run") as run:
        assert decode_audio(source) is None

    run.assert_not_called()


def test_decode_audio_loads_workspace_gstreamer_runtime(tmp_path: Path) -> None:
    source = _audio_file(tmp_path, ".aac")
    completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout=np.zeros(1, dtype="<f4").tobytes(), stderr=b""
    )
    existing_plugins = "/existing/gstreamer/plugins"
    existing_libraries = "/existing/libraries"

    with (
        patch.dict(
            "voiceprint.audio.os.environ",
            {
                "GST_PLUGIN_PATH_1_0": existing_plugins,
                "LD_LIBRARY_PATH": existing_libraries,
            },
        ),
        patch("voiceprint.audio.subprocess.run", return_value=completed) as run,
    ):
        assert decode_audio(source) is not None

    environment = run.call_args.kwargs["env"]
    project_root = Path(__file__).resolve().parents[1]
    assert environment["GST_PLUGIN_PATH_1_0"].split(os.pathsep) == [
        str(project_root / ".runtime" / "gstreamer" / "plugins"),
        existing_plugins,
    ]
    assert environment["LD_LIBRARY_PATH"].split(os.pathsep) == [
        str(project_root / ".runtime" / "gstreamer" / "lib"),
        existing_libraries,
    ]


def test_decode_audio_returns_none_on_gstreamer_error(tmp_path: Path) -> None:
    source = _audio_file(tmp_path)
    completed = subprocess.CompletedProcess(
        args=[], returncode=1, stdout=b"", stderr=b"decode failed"
    )

    with patch("voiceprint.audio.subprocess.run", return_value=completed):
        assert decode_audio(source) is None


def test_decode_audio_rejects_misaligned_pcm(tmp_path: Path) -> None:
    source = _audio_file(tmp_path)
    completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout=b"abc", stderr=b""
    )

    with patch("voiceprint.audio.subprocess.run", return_value=completed):
        assert decode_audio(source) is None


def test_inspect_audio_parses_discoverer_output(tmp_path: Path) -> None:
    source = _audio_file(tmp_path, ".MP3")
    output = """
Properties:
  Duration: 0:00:03.500000000
  Tags:
    container format: MP3
    audio codec: MPEG-1 Layer 3 (MP3)
Topology:
  container: MPEG-1 System Stream
    audio #0: audio/mpeg
      Channels: 2 (front-left, front-right)
      Sample rate: 44100
      Bitrate: 128000
"""
    completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout=output, stderr=""
    )

    with patch("voiceprint.audio.subprocess.run", return_value=completed) as run:
        info = inspect_audio(source)

    assert info == {
        "path": str(source.resolve()),
        "format": "mp3",
        "duration_seconds": 3.5,
        "codec": "MPEG-1 Layer 3 (MP3)",
        "sample_rate": 44100,
        "channels": 2,
        "container": "MP3",
        "bitrate": 128000,
    }
    assert run.call_args.args[0][-1] == str(source.resolve())
    assert run.call_args.kwargs["shell"] is False


def test_inspect_audio_keeps_unavailable_fields_as_none(tmp_path: Path) -> None:
    source = _audio_file(tmp_path, ".aac")
    completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="Duration: 0:01:02.25\n", stderr=""
    )

    with patch("voiceprint.audio.subprocess.run", return_value=completed):
        info = inspect_audio(source)

    assert info is not None
    assert info["format"] == "aac"
    assert info["duration_seconds"] == 62.25
    assert info["codec"] is None
    assert info["sample_rate"] is None


def test_inspect_audio_rejects_unsupported_suffix_without_gstreamer(
    tmp_path: Path,
) -> None:
    source = _audio_file(tmp_path, ".m4a")

    with patch("voiceprint.audio.subprocess.run") as run:
        assert inspect_audio(source) is None

    run.assert_not_called()


def test_trim_silence_removes_quiet_edges_and_keeps_padding() -> None:
    samples = np.concatenate(
        [
            np.zeros(400, dtype=np.float32),
            np.full(400, 0.5, dtype=np.float32),
            np.zeros(400, dtype=np.float32),
        ]
    )

    trimmed = trim_silence(
        samples,
        sample_rate=1000,
        threshold_db=-20.0,
        frame_ms=100.0,
        padding_ms=100.0,
    )

    assert trimmed.size == 600
    np.testing.assert_array_equal(trimmed, samples[300:900])


def test_trim_silence_returns_empty_for_silence() -> None:
    trimmed = trim_silence(np.zeros(1000, dtype=np.float32), sample_rate=1000)

    assert trimmed.dtype == np.float32
    assert trimmed.size == 0


def test_split_windows_keeps_short_last_window() -> None:
    samples = np.arange(10, dtype=np.float32)

    windows = split_windows(samples, sample_rate=2, window_seconds=2.0)

    assert [window.tolist() for window in windows] == [
        [0.0, 1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0, 7.0],
        [8.0, 9.0],
    ]


def test_split_windows_can_pad_and_overlap() -> None:
    samples = np.arange(9, dtype=np.float32)

    windows = split_windows(
        samples,
        sample_rate=2,
        window_seconds=2.0,
        hop_seconds=1.0,
        pad_last=True,
    )

    assert [window.tolist() for window in windows] == [
        [0.0, 1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0, 5.0],
        [4.0, 5.0, 6.0, 7.0],
        [6.0, 7.0, 8.0, 0.0],
    ]
