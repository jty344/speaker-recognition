from unittest.mock import patch

import numpy as np
import torch

from voiceprint.features import compute_fbank, l2_normalize


def test_l2_normalize_returns_unit_vector():
    result = l2_normalize(np.array([3.0, 4.0], dtype=np.float32))
    assert result is not None
    assert np.allclose(result, [0.6, 0.8])


def test_compute_fbank_shape_and_cmn():
    sample_rate = 16000
    timeline = np.arange(sample_rate * 3, dtype=np.float32) / sample_rate
    samples = 0.1 * np.sin(2.0 * np.pi * 220.0 * timeline)
    features = compute_fbank(samples, sample_rate=sample_rate)
    assert features is not None
    assert features.ndim == 2
    assert features.shape[1] == 80
    assert np.allclose(features.mean(axis=0), 0.0, atol=1e-4)


def test_compute_fbank_matches_wespeaker_frontend_parameters():
    samples = np.linspace(-0.5, 0.5, 400, dtype=np.float32)
    fake_features = torch.arange(160, dtype=torch.float32).reshape(2, 80)

    with patch("voiceprint.features.kaldi.fbank", return_value=fake_features) as fbank:
        result = compute_fbank(samples, sample_rate=16000)

    assert result is not None
    waveform = fbank.call_args.args[0]
    expected = torch.from_numpy(samples).unsqueeze(0) * float(1 << 15)
    assert torch.equal(waveform, expected)
    assert fbank.call_args.kwargs == {
        "num_mel_bins": 80,
        "frame_length": 25,
        "frame_shift": 10,
        "dither": 0.0,
        "sample_frequency": 16000,
        "window_type": "hamming",
        "use_energy": False,
    }
