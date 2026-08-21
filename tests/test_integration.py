from pathlib import Path

import numpy as np
import pytest

from voiceprint.backend import OnnxEmbeddingBackend
from voiceprint.config import DEFAULT_THRESHOLD, MODEL_PATH, PREPROCESSING_ID
from voiceprint.registry import UNKNOWN_SPEAKER, VoiceprintRegistry
from voiceprint.service import VoiceprintService


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_AUDIO_DIR = PROJECT_ROOT / "data" / "test_assets" / "generated"
ALICE_ENROLL = TEST_AUDIO_DIR / "spk1_snt1.mp3"
ALICE_QUERY = TEST_AUDIO_DIR / "spk1_snt2.mp3"
UNKNOWN_QUERY = TEST_AUDIO_DIR / "spk2_snt1_twice.mp3"
ASSETS_READY = MODEL_PATH.is_file() and all(
    path.is_file() for path in [ALICE_ENROLL, ALICE_QUERY, UNKNOWN_QUERY]
)


@pytest.mark.skipif(not ASSETS_READY, reason="模型或测试 MP3 尚未准备")
def test_real_mp3_open_set_pipeline(tmp_path):
    backend = OnnxEmbeddingBackend(MODEL_PATH, num_threads=1)
    assert backend.load()
    service = VoiceprintService(backend)

    enroll_embedding = service.extract_embedding(ALICE_ENROLL)
    same_embedding = service.extract_embedding(ALICE_QUERY)
    unknown_embedding = service.extract_embedding(UNKNOWN_QUERY)
    assert enroll_embedding is not None
    assert same_embedding is not None
    assert unknown_embedding is not None
    assert enroll_embedding.shape == (256,)

    same_score = float(enroll_embedding @ same_embedding)
    unknown_score = float(enroll_embedding @ unknown_embedding)
    assert same_score > unknown_score
    threshold = (same_score + unknown_score) / 2.0

    registry = VoiceprintRegistry(
        tmp_path / "voiceprints.npz",
        tmp_path / "voiceprints.json",
        model_sha256="integration-model",
        preprocessing_id=PREPROCESSING_ID,
    )
    assert registry.enroll("alice", [enroll_embedding])

    same_result = registry.identify(same_embedding, threshold=threshold)
    unknown_result = registry.identify(unknown_embedding, threshold=threshold)
    same_default_result = registry.identify(
        same_embedding,
        threshold=DEFAULT_THRESHOLD,
    )
    unknown_default_result = registry.identify(
        unknown_embedding,
        threshold=DEFAULT_THRESHOLD,
    )
    assert same_result is not None
    assert unknown_result is not None
    assert same_default_result is not None
    assert unknown_default_result is not None
    assert same_result["speaker"] == "alice"
    assert unknown_result["speaker"] == UNKNOWN_SPEAKER
    assert same_default_result["speaker"] == "alice"
    assert unknown_default_result["speaker"] == UNKNOWN_SPEAKER
    assert np.isclose(np.linalg.norm(enroll_embedding), 1.0)
