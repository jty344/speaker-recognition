"""Audio decoding and lightweight waveform utilities."""

from __future__ import annotations

import logging
import os
import re
import subprocess
from pathlib import Path

import numpy as np


LOGGER = logging.getLogger(__name__)

GST_LAUNCH = "gst-launch-1.0"
GST_DISCOVERER = "gst-discoverer-1.0"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_XDG_CACHE_HOME = _PROJECT_ROOT / ".cache" / "xdg"
_GST_REGISTRY = _PROJECT_ROOT / ".cache" / "gstreamer" / "registry.bin"
_GST_PLUGIN_DIR = _PROJECT_ROOT / ".runtime" / "gstreamer" / "plugins"
_GST_LIBRARY_DIR = _PROJECT_ROOT / ".runtime" / "gstreamer" / "lib"
_AUDIO_FORMATS = {
    ".aac": "aac",
    ".flac": "flac",
    ".mp3": "mp3",
    ".wav": "wav",
}
_AUDIO_PARSERS = {
    "aac": "aacparse",
    "flac": "flacparse",
    "mp3": "mpegaudioparse",
    "wav": "wavparse",
}


def _prepend_env_path(env: dict[str, str], name: str, path: Path) -> None:
    current = env.get(name)
    env[name] = str(path) if not current else f"{path}{os.pathsep}{current}"


def _gstreamer_env() -> dict[str, str]:
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    env["GST_DEBUG_NO_COLOR"] = "1"
    env["XDG_CACHE_HOME"] = str(_XDG_CACHE_HOME)
    env["GST_REGISTRY"] = str(_GST_REGISTRY)
    _prepend_env_path(env, "GST_PLUGIN_PATH_1_0", _GST_PLUGIN_DIR)
    _prepend_env_path(env, "LD_LIBRARY_PATH", _GST_LIBRARY_DIR)
    return env


def _source_path(path: str | Path) -> Path | None:
    source = Path(path).expanduser().resolve()
    if source.is_file():
        return source

    LOGGER.error("音频文件不存在或不是普通文件: %s", source)
    return None


def detect_audio_format(path: str | Path) -> str | None:
    """Return the normalized format selected from the final filename suffix."""
    suffix = Path(path).suffix.lower()
    audio_format = _AUDIO_FORMATS.get(suffix)
    if audio_format is not None:
        return audio_format

    supported = ", ".join(sorted(_AUDIO_FORMATS))
    LOGGER.error("不支持的音频文件后缀 %r；支持: %s", suffix or "<无后缀>", supported)
    return None


def decode_audio(path: str | Path, sample_rate: int = 16000) -> np.ndarray | None:
    """Decode an audio file into mono float32 PCM at ``sample_rate``."""
    source = _source_path(path)
    if source is None:
        return None
    audio_format = detect_audio_format(path)
    if audio_format is None:
        return None
    if sample_rate <= 0:
        LOGGER.error("采样率必须大于 0: %s", sample_rate)
        return None

    parser = _AUDIO_PARSERS[audio_format]

    command = [
        GST_LAUNCH,
        "-q",
        "filesrc",
        f"location={source}",
        "!",
        parser,
        "!",
        "decodebin",
        "!",
        "audioconvert",
        "!",
        "audioresample",
        "!",
        f"audio/x-raw,format=F32LE,layout=interleaved,channels=1,rate={sample_rate}",
        "!",
        "fdsink",
        "fd=1",
        "sync=false",
    ]

    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
            env=_gstreamer_env(),
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        LOGGER.error("GStreamer 解码执行失败: %s", error)
        return None

    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        LOGGER.error("GStreamer 解码失败: %s", detail or f"退出码 {result.returncode}")
        return None
    if not result.stdout:
        LOGGER.error("GStreamer 解码未产生 PCM 数据: %s", source)
        return None
    if len(result.stdout) % np.dtype("<f4").itemsize != 0:
        LOGGER.error("GStreamer 返回的 PCM 数据长度无效: %s bytes", len(result.stdout))
        return None

    return np.frombuffer(result.stdout, dtype="<f4").astype(np.float32, copy=True)


def _match_text(pattern: str, output: str) -> str | None:
    match = re.search(pattern, output, flags=re.IGNORECASE | re.MULTILINE)
    if match is None:
        return None
    value = match.group(1).strip()
    return value or None


def _match_int(pattern: str, output: str) -> int | None:
    value = _match_text(pattern, output)
    if value is None:
        return None
    return int(value)


def _duration_seconds(value: str | None) -> float | None:
    if value is None:
        return None

    fields = value.split(":")
    if len(fields) != 3:
        return None
    try:
        hours, minutes, seconds = fields
        return int(hours) * 3600.0 + int(minutes) * 60.0 + float(seconds)
    except ValueError:
        return None


def inspect_audio(path: str | Path) -> dict[str, object] | None:
    """Return stable metadata fields reported by ``gst-discoverer-1.0``."""
    source = _source_path(path)
    if source is None:
        return None
    audio_format = detect_audio_format(path)
    if audio_format is None:
        return None

    command = [GST_DISCOVERER, "--timeout=10", "--verbose", str(source)]
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20,
            env=_gstreamer_env(),
            text=True,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        LOGGER.error("GStreamer 音频探测执行失败: %s", error)
        return None

    if result.returncode != 0:
        detail = result.stderr.strip()
        LOGGER.error("GStreamer 音频探测失败: %s", detail or f"退出码 {result.returncode}")
        return None

    duration = _match_text(r"^\s*Duration:\s*([^\r\n]+)", result.stdout)
    codec = _match_text(r"^\s*audio codec:\s*([^\r\n]+)", result.stdout)
    if codec is None:
        codec = _match_text(r"^\s*audio(?:\s+#\d+)?:\s*([^\r\n]+)", result.stdout)
    container = _match_text(r"^\s*container format:\s*([^\r\n]+)", result.stdout)
    if container is None:
        container = _match_text(
            r"^\s*container(?:\s+#\d+)?:\s*([^\r\n]+)", result.stdout
        )
    return {
        "path": str(source),
        "format": audio_format,
        "duration_seconds": _duration_seconds(duration),
        "codec": codec,
        "sample_rate": _match_int(r"^\s*Sample rate:\s*(\d+)", result.stdout),
        "channels": _match_int(r"^\s*Channels:\s*(\d+)", result.stdout),
        "container": container,
        "bitrate": _match_int(r"^\s*Bitrate:\s*(\d+)", result.stdout),
    }


def trim_silence(
    samples: np.ndarray,
    sample_rate: int = 16000,
    threshold_db: float = -40.0,
    frame_ms: float = 20.0,
    padding_ms: float = 100.0,
) -> np.ndarray:
    """Trim leading and trailing low-RMS frames, retaining optional padding."""
    signal = np.asarray(samples, dtype=np.float32).reshape(-1)
    if signal.size == 0:
        return signal.copy()

    frame_size = int(round(sample_rate * frame_ms / 1000.0))
    padding_size = int(round(sample_rate * padding_ms / 1000.0))
    if sample_rate <= 0 or frame_size <= 0 or padding_size < 0:
        LOGGER.error("静音裁剪参数无效")
        return signal.copy()

    threshold = 10.0 ** (threshold_db / 20.0)
    active_frames: list[int] = []
    frame_count = (signal.size + frame_size - 1) // frame_size
    for index in range(frame_count):
        frame = signal[index * frame_size : (index + 1) * frame_size]
        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))
        if rms >= threshold:
            active_frames.append(index)

    if not active_frames:
        return np.empty(0, dtype=np.float32)

    start = max(0, active_frames[0] * frame_size - padding_size)
    end = min(signal.size, (active_frames[-1] + 1) * frame_size + padding_size)
    return signal[start:end].copy()


def split_windows(
    samples: np.ndarray,
    sample_rate: int = 16000,
    window_seconds: float = 3.0,
    hop_seconds: float | None = None,
    pad_last: bool = False,
) -> list[np.ndarray]:
    """Split PCM into windows; the final window is short or zero-padded."""
    signal = np.asarray(samples, dtype=np.float32).reshape(-1)
    if signal.size == 0:
        return []

    hop_seconds = window_seconds if hop_seconds is None else hop_seconds
    window_size = int(round(sample_rate * window_seconds))
    hop_size = int(round(sample_rate * hop_seconds))
    if sample_rate <= 0 or window_size <= 0 or hop_size <= 0:
        LOGGER.error("音频切片参数无效")
        return []

    windows: list[np.ndarray] = []
    for start in range(0, signal.size, hop_size):
        window = signal[start : start + window_size].copy()
        if window.size < window_size and pad_last:
            window = np.pad(window, (0, window_size - window.size))
        windows.append(window.astype(np.float32, copy=False))
        if start + window_size >= signal.size:
            break
    return windows
