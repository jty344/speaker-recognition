import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort

from voiceprint.features import l2_normalize


LOGGER = logging.getLogger(__name__)


class OnnxEmbeddingBackend:
    def __init__(self, model_path: str | Path, num_threads: int = 1):
        self.model_path = Path(model_path)
        self.num_threads = max(1, int(num_threads))
        self.session: ort.InferenceSession | None = None
        self.input_name = "feats"
        self.output_name = "embs"

    def load(self) -> bool:
        if not self.model_path.is_file():
            LOGGER.error("ONNX 模型不存在: %s", self.model_path)
            return False

        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = self.num_threads
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        try:
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
        except Exception as error:
            LOGGER.error("加载 ONNX 模型失败: %s", error)
            return False

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if not inputs or not outputs:
            LOGGER.error("ONNX 模型缺少输入或输出")
            self.session = None
            return False

        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        return True

    def embed(self, features: np.ndarray) -> np.ndarray | None:
        if self.session is None and not self.load():
            return None

        feature_values = np.asarray(features, dtype=np.float32)
        if feature_values.ndim != 2 or feature_values.shape[1] != 80:
            LOGGER.error("模型输入必须是 [T, 80] FBank，当前为 %s", feature_values.shape)
            return None

        model_input = np.ascontiguousarray(feature_values[None, :, :])
        try:
            outputs = self.session.run(
                [self.output_name],
                {self.input_name: model_input},
            )
        except Exception as error:
            LOGGER.error("ONNX 推理失败: %s", error)
            return None

        embedding = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        return l2_normalize(embedding)

    def runtime_info(self) -> dict[str, object]:
        return {
            "backend": "onnxruntime",
            "providers": self.session.get_providers() if self.session else [],
            "threads": self.num_threads,
            "model": str(self.model_path),
        }

