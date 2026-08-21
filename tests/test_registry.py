import json
from pathlib import Path

import numpy as np

from voiceprint.registry import UNKNOWN_SPEAKER, VoiceprintRegistry


def create_registry(tmp_path: Path) -> VoiceprintRegistry:
    return VoiceprintRegistry(
        tmp_path / "voiceprints.npz",
        tmp_path / "voiceprints.json",
        model_sha256="test-model",
        preprocessing_id="test-preprocessing",
    )


def test_enroll_identify_and_reload(tmp_path):
    registry = create_registry(tmp_path)
    assert registry.enroll(
        "alice",
        [
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.9, 0.1, 0.0], dtype=np.float32),
        ],
    )
    assert registry.enroll("bob", [np.array([0.0, 1.0, 0.0], dtype=np.float32)])

    result = registry.identify(
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        threshold=0.8,
    )
    assert result is not None
    assert result["speaker"] == "alice"
    assert result["accepted"] is True

    reloaded = create_registry(tmp_path)
    assert [item["name"] for item in reloaded.list_speakers()] == ["alice", "bob"]


def test_identify_rejects_unknown():
    registry = VoiceprintRegistry(
        "unused.npz",
        "unused.json",
        "test-model",
        "test-preprocessing",
    )
    registry.speakers = ["alice"]
    registry.sample_counts = [1]
    registry.centroids = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    result = registry.identify(
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        threshold=0.5,
    )
    assert result is not None
    assert result["speaker"] == UNKNOWN_SPEAKER
    assert result["accepted"] is False


def test_enroll_rejects_reserved_unknown_name(tmp_path):
    registry = create_registry(tmp_path)
    assert not registry.enroll(
        "unknown",
        [np.array([1.0, 0.0, 0.0], dtype=np.float32)],
    )


def test_load_rejects_invalid_metadata_structure(tmp_path):
    data_path = tmp_path / "voiceprints.npz"
    metadata_path = tmp_path / "voiceprints.json"
    np.savez_compressed(
        data_path,
        centroids=np.array([[1.0, 0.0]], dtype=np.float32),
    )
    metadata_path.write_text(json.dumps([]), encoding="utf-8")

    registry = VoiceprintRegistry(
        data_path,
        metadata_path,
        model_sha256="test-model",
        preprocessing_id="test-preprocessing",
    )

    assert registry.loaded is False


def test_load_rejects_different_preprocessing(tmp_path):
    registry = create_registry(tmp_path)
    assert registry.enroll(
        "alice",
        [np.array([1.0, 0.0], dtype=np.float32)],
    )

    mismatched = VoiceprintRegistry(
        tmp_path / "voiceprints.npz",
        tmp_path / "voiceprints.json",
        model_sha256="test-model",
        preprocessing_id="different-preprocessing",
    )

    assert mismatched.loaded is False
