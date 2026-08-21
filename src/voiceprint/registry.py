import json
import logging
import os
from pathlib import Path

import numpy as np

from voiceprint.features import l2_normalize


LOGGER = logging.getLogger(__name__)
UNKNOWN_SPEAKER = "UNKNOWN"


class VoiceprintRegistry:
    def __init__(
        self,
        data_path: str | Path,
        metadata_path: str | Path,
        model_sha256: str,
        preprocessing_id: str,
    ):
        self.data_path = Path(data_path)
        self.metadata_path = Path(metadata_path)
        self.model_sha256 = model_sha256
        self.preprocessing_id = preprocessing_id
        self.speakers: list[str] = []
        self.sample_counts: list[int] = []
        self.centroids = np.empty((0, 0), dtype=np.float32)
        self.loaded = self._load()

    def _load(self) -> bool:
        data_exists = self.data_path.exists()
        metadata_exists = self.metadata_path.exists()
        if not data_exists and not metadata_exists:
            return True
        if data_exists != metadata_exists:
            LOGGER.error("声纹库数据与元数据不完整")
            return False

        try:
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            with np.load(self.data_path, allow_pickle=False) as data:
                centroids = np.asarray(data["centroids"], dtype=np.float32)
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            json.JSONDecodeError,
        ) as error:
            LOGGER.error("读取声纹库失败: %s", error)
            return False

        if not isinstance(metadata, dict):
            LOGGER.error("声纹库元数据必须是 JSON 对象")
            return False
        if metadata.get("model_sha256") != self.model_sha256:
            LOGGER.error("声纹库与当前 ONNX 模型不匹配")
            return False
        if metadata.get("preprocessing_id") != self.preprocessing_id:
            LOGGER.error("声纹库与当前音频前处理配置不匹配")
            return False

        speakers = metadata.get("speakers")
        if not isinstance(speakers, list):
            LOGGER.error("声纹库说话人元数据格式无效")
            return False
        if centroids.ndim != 2 or len(speakers) != centroids.shape[0]:
            LOGGER.error("声纹库维度与元数据不一致")
            return False

        try:
            names = [str(item["name"]).strip() for item in speakers]
            sample_counts = [int(item["sample_count"]) for item in speakers]
            embedding_dim = int(metadata["embedding_dim"])
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            LOGGER.error("声纹库说话人元数据格式无效: %s", error)
            return False

        actual_dim = int(centroids.shape[1]) if centroids.size else 0
        if embedding_dim != actual_dim:
            LOGGER.error("声纹库 embedding 维度元数据不一致")
            return False
        if (
            any(not name or name.upper() == UNKNOWN_SPEAKER for name in names)
            or len(set(names)) != len(names)
            or any(count <= 0 for count in sample_counts)
            or not np.all(np.isfinite(centroids))
        ):
            LOGGER.error("声纹库包含无效的说话人、样本数或向量")
            return False

        self.speakers = names
        self.sample_counts = sample_counts
        self.centroids = centroids
        return True

    def _save(self) -> bool:
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        data_tmp = self.data_path.with_suffix(self.data_path.suffix + ".tmp")
        metadata_tmp = self.metadata_path.with_suffix(self.metadata_path.suffix + ".tmp")
        metadata = {
            "version": 1,
            "model_sha256": self.model_sha256,
            "preprocessing_id": self.preprocessing_id,
            "embedding_dim": int(self.centroids.shape[1]) if self.centroids.size else 0,
            "speakers": [
                {"name": name, "sample_count": count}
                for name, count in zip(self.speakers, self.sample_counts)
            ],
        }

        try:
            with data_tmp.open("wb") as output:
                np.savez_compressed(output, centroids=self.centroids)
            metadata_tmp.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(data_tmp, self.data_path)
            os.replace(metadata_tmp, self.metadata_path)
        except OSError as error:
            LOGGER.error("保存声纹库失败: %s", error)
            return False
        return True

    def list_speakers(self) -> list[dict[str, object]]:
        return [
            {"name": name, "sample_count": count}
            for name, count in zip(self.speakers, self.sample_counts)
        ]

    def enroll(
        self,
        speaker: str,
        embeddings: list[np.ndarray],
        replace: bool = False,
    ) -> bool:
        name = speaker.strip()
        if name.upper() == UNKNOWN_SPEAKER:
            LOGGER.error("%s 是系统保留名称，不能用于注册", UNKNOWN_SPEAKER)
            return False
        if not self.loaded or not name or not embeddings:
            LOGGER.error("注册参数无效或声纹库未成功加载")
            return False

        normalized = [l2_normalize(item) for item in embeddings]
        if any(item is None for item in normalized):
            return False
        matrix = np.stack(normalized).astype(np.float32)
        centroid = l2_normalize(np.mean(matrix, axis=0))
        if centroid is None:
            return False

        if self.centroids.size and centroid.shape[0] != self.centroids.shape[1]:
            LOGGER.error("新声纹维度与现有声纹库不一致")
            return False

        if name in self.speakers:
            if not replace:
                LOGGER.error("说话人 %s 已存在；如需覆盖请使用 --replace", name)
                return False
            index = self.speakers.index(name)
            self.centroids[index] = centroid
            self.sample_counts[index] = len(embeddings)
        else:
            self.speakers.append(name)
            self.sample_counts.append(len(embeddings))
            if self.centroids.size:
                self.centroids = np.vstack([self.centroids, centroid])
            else:
                self.centroids = centroid.reshape(1, -1)
        return self._save()

    def remove(self, speaker: str) -> bool:
        name = speaker.strip()
        if not self.loaded or name not in self.speakers:
            LOGGER.error("说话人不存在: %s", name)
            return False
        index = self.speakers.index(name)
        self.speakers.pop(index)
        self.sample_counts.pop(index)
        self.centroids = np.delete(self.centroids, index, axis=0)
        if self.centroids.shape[0] == 0:
            self.centroids = np.empty((0, 0), dtype=np.float32)
        return self._save()

    def identify(
        self,
        embedding: np.ndarray,
        threshold: float,
        top_k: int = 3,
    ) -> dict[str, object] | None:
        if not self.loaded or not self.speakers:
            LOGGER.error("声纹库为空或未成功加载")
            return None
        query = l2_normalize(embedding)
        if query is None or query.shape[0] != self.centroids.shape[1]:
            LOGGER.error("查询声纹维度与声纹库不一致")
            return None

        scores = self.centroids @ query
        order = np.argsort(scores)[::-1][:max(1, min(top_k, len(self.speakers)))]
        matches = [
            {"speaker": self.speakers[int(index)], "score": float(scores[index])}
            for index in order
        ]
        best = matches[0]
        accepted = best["score"] >= threshold
        return {
            "speaker": best["speaker"] if accepted else UNKNOWN_SPEAKER,
            "score": best["score"],
            "accepted": accepted,
            "threshold": float(threshold),
            "matches": matches,
        }

    def verify(
        self,
        speaker: str,
        embedding: np.ndarray,
        threshold: float,
    ) -> dict[str, object] | None:
        name = speaker.strip()
        if not self.loaded or name not in self.speakers:
            LOGGER.error("说话人不存在: %s", name)
            return None
        query = l2_normalize(embedding)
        if query is None or query.shape[0] != self.centroids.shape[1]:
            LOGGER.error("查询声纹维度与声纹库不一致")
            return None
        index = self.speakers.index(name)
        score = float(self.centroids[index] @ query)
        return {
            "speaker": name,
            "score": score,
            "accepted": score >= threshold,
            "threshold": float(threshold),
        }
