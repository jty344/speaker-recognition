import logging

import numpy as np
import torch
import torchaudio.compliance.kaldi as kaldi


LOGGER = logging.getLogger(__name__)


def l2_normalize(vector: np.ndarray) -> np.ndarray | None:
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(values))
    if norm <= 0.0 or not np.isfinite(norm):
        LOGGER.error("无法归一化空向量或无效向量")
        return None
    return values / norm


def compute_fbank(
    samples: np.ndarray,
    sample_rate: int = 16000,
    num_mel_bins: int = 80,
) -> np.ndarray | None:
    waveform_values = np.asarray(samples, dtype=np.float32).reshape(-1)
    if waveform_values.size < int(sample_rate * 0.025):
        LOGGER.error("有效音频不足一个 25 ms 分帧窗口")
        return None

    waveform = torch.from_numpy(np.ascontiguousarray(waveform_values))
    waveform = waveform.unsqueeze(0) * float(1 << 15)

    with torch.inference_mode():
        features = kaldi.fbank(
            waveform,
            num_mel_bins=num_mel_bins,
            frame_length=25,
            frame_shift=10,
            dither=0.0,
            sample_frequency=sample_rate,
            window_type="hamming",
            use_energy=False,
        )

    if features.shape[0] == 0:
        LOGGER.error("FBank 特征为空")
        return None

    features = features - torch.mean(features, dim=0, keepdim=True)
    return features.cpu().numpy().astype(np.float32, copy=False)

