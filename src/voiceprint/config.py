from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "wespeaker-cnceleb-resnet34-lm"
MODEL_PATH = MODEL_DIR / "cnceleb_resnet34_LM.onnx"
MODEL_CONFIG_PATH = MODEL_DIR / "config.yaml"
REGISTRY_DIR = PROJECT_ROOT / "artifacts" / "registry"
REGISTRY_DATA_PATH = REGISTRY_DIR / "voiceprints.npz"
REGISTRY_META_PATH = REGISTRY_DIR / "voiceprints.json"

SAMPLE_RATE = 16000
MIN_SPEECH_SECONDS = 2.0
WINDOW_SECONDS = 3.0
DEFAULT_THRESHOLD = 0.5
PREPROCESSING_ID = (
    "gstreamer-f32le-mono16k|rms-trim--40db|segments-3s-min2s|"
    "kaldi-fbank80-25ms-10ms-hamming-dither0-cmn"
)
