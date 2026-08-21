import argparse
import hashlib
import json
import logging
import math
from pathlib import Path

import numpy as np

from voiceprint import __version__
from voiceprint.audio import decode_audio, inspect_audio
from voiceprint.backend import OnnxEmbeddingBackend
from voiceprint.config import (
    DEFAULT_THRESHOLD,
    MODEL_PATH,
    PREPROCESSING_ID,
    REGISTRY_DATA_PATH,
    REGISTRY_META_PATH,
    SAMPLE_RATE,
)
from voiceprint.registry import VoiceprintRegistry
from voiceprint.service import VoiceprintService


LOGGER = logging.getLogger(__name__)


def print_json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        LOGGER.error("文件不存在: %s", path)
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        LOGGER.error("计算文件 SHA-256 失败: %s", error)
        return None
    return digest.hexdigest()


def create_runtime(
    model_path: Path,
    threads: int,
) -> tuple[VoiceprintService, VoiceprintRegistry] | None:
    model_sha256 = sha256_file(model_path)
    if model_sha256 is None:
        return None
    backend = OnnxEmbeddingBackend(model_path=model_path, num_threads=threads)
    if not backend.load():
        return None
    registry = VoiceprintRegistry(
        data_path=REGISTRY_DATA_PATH,
        metadata_path=REGISTRY_META_PATH,
        model_sha256=model_sha256,
        preprocessing_id=PREPROCESSING_ID,
    )
    if not registry.loaded:
        return None
    return VoiceprintService(backend), registry


def command_inspect(args: argparse.Namespace) -> int:
    details = inspect_audio(args.audio)
    if details is None:
        return 1
    samples = decode_audio(args.audio, sample_rate=SAMPLE_RATE)
    if samples is None:
        return 1
    details["decoded"] = {
        "sample_rate": SAMPLE_RATE,
        "channels": 1,
        "dtype": str(samples.dtype),
        "samples": int(samples.size),
        "duration_seconds": samples.size / float(SAMPLE_RATE),
        "peak": float(np.max(np.abs(samples))),
        "rms": float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)))),
    }
    print_json(details)
    return 0


def command_embed(
    args: argparse.Namespace,
    service: VoiceprintService,
) -> int:
    embedding = service.extract_embedding(args.audio)
    if embedding is None:
        return 1
    print_json(
        {
            "audio": str(Path(args.audio).resolve()),
            "dimension": int(embedding.shape[0]),
            "l2_norm": float(np.linalg.norm(embedding)),
            "embedding": embedding.tolist(),
        }
    )
    return 0


def command_enroll(
    args: argparse.Namespace,
    service: VoiceprintService,
    registry: VoiceprintRegistry,
) -> int:
    embeddings = []
    for audio in args.audio:
        embedding = service.extract_embedding(audio)
        if embedding is None:
            LOGGER.error("注册中止，无法提取声纹: %s", audio)
            return 1
        embeddings.append(embedding)
    if not registry.enroll(args.speaker, embeddings, replace=args.replace):
        return 1
    print_json(
        {
            "speaker": args.speaker.strip(),
            "enrolled_files": len(embeddings),
            "registry_size": len(registry.speakers),
        }
    )
    return 0


def resolve_threshold(args: argparse.Namespace) -> tuple[float, str]:
    if args.threshold is None:
        return DEFAULT_THRESHOLD, "demo_default_uncalibrated"
    return float(args.threshold), "command_line"


def command_identify(
    args: argparse.Namespace,
    service: VoiceprintService,
    registry: VoiceprintRegistry,
) -> int:
    embedding = service.extract_embedding(args.audio)
    if embedding is None:
        return 1
    threshold, threshold_source = resolve_threshold(args)
    result = registry.identify(embedding, threshold=threshold, top_k=args.top_k)
    if result is None:
        return 1
    result["threshold_source"] = threshold_source
    result["audio"] = str(Path(args.audio).resolve())
    print_json(result)
    return 0


def command_verify(
    args: argparse.Namespace,
    service: VoiceprintService,
    registry: VoiceprintRegistry,
) -> int:
    embedding = service.extract_embedding(args.audio)
    if embedding is None:
        return 1
    threshold, threshold_source = resolve_threshold(args)
    result = registry.verify(args.speaker, embedding, threshold=threshold)
    if result is None:
        return 1
    result["threshold_source"] = threshold_source
    result["audio"] = str(Path(args.audio).resolve())
    print_json(result)
    return 0


def command_list(registry: VoiceprintRegistry) -> int:
    speakers = registry.list_speakers()
    print_json({"count": len(speakers), "speakers": speakers})
    return 0


def command_remove(args: argparse.Namespace, registry: VoiceprintRegistry) -> int:
    if not registry.remove(args.speaker):
        return 1
    print_json({"removed": args.speaker.strip(), "registry_size": len(registry.speakers)})
    return 0


def threshold_value(value: str) -> float:
    threshold = float(value)
    if not math.isfinite(threshold) or threshold < -1.0 or threshold > 1.0:
        raise argparse.ArgumentTypeError("threshold 必须是 [-1, 1] 内的有限数值")
    return threshold


def positive_integer(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("数值必须大于 0")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voiceprint",
        description="单说话人音频开放集声纹识别 Demo",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)

    inspect_parser = commands.add_parser("inspect", help="查看音频编解码信息")
    inspect_parser.add_argument("audio", type=Path)

    embed_parser = commands.add_parser("embed", help="输出 256 维声纹")
    embed_parser.add_argument("audio", type=Path)

    enroll_parser = commands.add_parser("enroll", help="注册说话人")
    enroll_parser.add_argument("--speaker", required=True)
    enroll_parser.add_argument("--replace", action="store_true")
    enroll_parser.add_argument("audio", nargs="+", type=Path)

    identify_parser = commands.add_parser("identify", help="1:N 识别或返回 UNKNOWN")
    identify_parser.add_argument("audio", type=Path)
    identify_parser.add_argument("--threshold", type=threshold_value)
    identify_parser.add_argument("--top-k", type=positive_integer, default=3)

    verify_parser = commands.add_parser("verify", help="1:1 验证指定说话人")
    verify_parser.add_argument("--speaker", required=True)
    verify_parser.add_argument("audio", type=Path)
    verify_parser.add_argument("--threshold", type=threshold_value)

    commands.add_parser("list", help="列出已注册说话人")
    remove_parser = commands.add_parser("remove", help="删除已注册说话人")
    remove_parser.add_argument("--speaker", required=True)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if args.command == "inspect":
        return command_inspect(args)

    runtime = create_runtime(args.model, args.threads)
    if runtime is None:
        return 1
    service, registry = runtime

    handlers = {
        "embed": lambda: command_embed(args, service),
        "enroll": lambda: command_enroll(args, service, registry),
        "identify": lambda: command_identify(args, service, registry),
        "verify": lambda: command_verify(args, service, registry),
        "list": lambda: command_list(registry),
        "remove": lambda: command_remove(args, registry),
    }
    return handlers[args.command]()
