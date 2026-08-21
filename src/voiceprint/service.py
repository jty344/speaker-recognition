import logging
from pathlib import Path

import numpy as np

from voiceprint.audio import decode_audio, split_windows, trim_silence
from voiceprint.backend import OnnxEmbeddingBackend
from voiceprint.config import MIN_SPEECH_SECONDS, SAMPLE_RATE, WINDOW_SECONDS
from voiceprint.features import compute_fbank, l2_normalize


LOGGER = logging.getLogger(__name__)


class VoiceprintService:
    def __init__(self, backend: OnnxEmbeddingBackend):
        self.backend = backend

    def extract_embedding(self, audio_path: str | Path) -> np.ndarray | None:
        samples = decode_audio(audio_path, sample_rate=SAMPLE_RATE)
        if samples is None:
            return None
        speech = trim_silence(samples, sample_rate=SAMPLE_RATE)
        duration = speech.size / float(SAMPLE_RATE)
        if duration < MIN_SPEECH_SECONDS:
            LOGGER.error(
                "有效语音过短: %.2f 秒，至少需要 %.1f 秒",
                duration,
                MIN_SPEECH_SECONDS,
            )
            return None

        segments = split_windows(
            speech,
            sample_rate=SAMPLE_RATE,
            window_seconds=WINDOW_SECONDS,
            hop_seconds=WINDOW_SECONDS,
            pad_last=False,
        )
        minimum_samples = int(MIN_SPEECH_SECONDS * SAMPLE_RATE)
        segments = [segment for segment in segments if segment.size >= minimum_samples]
        if not segments:
            segments = [speech]

        embeddings = []
        for segment in segments:
            features = compute_fbank(segment, sample_rate=SAMPLE_RATE)
            if features is None:
                continue
            embedding = self.backend.embed(features)
            if embedding is not None:
                embeddings.append(embedding)

        if not embeddings:
            LOGGER.error("没有得到有效声纹特征: %s", audio_path)
            return None
        return l2_normalize(np.mean(np.stack(embeddings), axis=0))
