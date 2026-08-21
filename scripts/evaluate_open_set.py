import argparse
import hashlib
import json
import logging
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from voiceprint.backend import OnnxEmbeddingBackend
from voiceprint.config import (
    DEFAULT_THRESHOLD,
    MODEL_PATH,
    PREPROCESSING_ID,
)
from voiceprint.registry import UNKNOWN_SPEAKER, VoiceprintRegistry
from voiceprint.service import VoiceprintService


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets" / "mini_librispeech_open_set"
DEFAULT_MANIFEST = DEFAULT_DATASET_ROOT / "protocol" / "open_set_manifest.json"
DEFAULT_OUTPUT = DEFAULT_DATASET_ROOT / "results" / "open_set_evaluation.json"


def file_sha256(path: Path) -> str | None:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        LOGGER.error("计算模型 SHA-256 失败: %s", error)
        return None
    return digest.hexdigest()


def read_manifest(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        LOGGER.error("读取评测协议失败: %s", error)
        return None
    if not isinstance(payload, dict):
        LOGGER.error("评测协议必须是 JSON 对象")
        return None
    return payload


def ratio(numerator: float, denominator: int) -> float:
    return numerator / float(denominator) if denominator else 0.0


def evaluate(args: argparse.Namespace) -> int:
    manifest_path = args.manifest.resolve()
    manifest = read_manifest(manifest_path)
    if manifest is None:
        return 1
    dataset_root = manifest_path.parent.parent
    enrollment = manifest.get("enrollment")
    queries = manifest.get("queries")
    if not isinstance(enrollment, list) or not isinstance(queries, list):
        LOGGER.error("评测协议缺少 enrollment 或 queries 列表")
        return 1

    model_sha256 = file_sha256(args.model)
    if model_sha256 is None:
        return 1
    backend = OnnxEmbeddingBackend(args.model, num_threads=args.threads)
    if not backend.load():
        return 1
    service = VoiceprintService(backend)

    embeddings_by_speaker = defaultdict(list)
    started = time.perf_counter()
    for item in enrollment:
        speaker = str(item["expected"])
        audio_path = dataset_root / str(item["audio"])
        embedding = service.extract_embedding(audio_path)
        if embedding is None:
            LOGGER.error("注册音频提取失败: %s", audio_path)
            return 1
        embeddings_by_speaker[speaker].append(embedding)

    temporary_root = PROJECT_ROOT / ".tmp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="voiceprint-evaluation-",
        dir=temporary_root,
    ) as temporary:
        temporary_path = Path(temporary)
        registry = VoiceprintRegistry(
            temporary_path / "voiceprints.npz",
            temporary_path / "voiceprints.json",
            model_sha256=model_sha256,
            preprocessing_id=PREPROCESSING_ID,
        )
        for speaker in sorted(embeddings_by_speaker):
            if not registry.enroll(speaker, embeddings_by_speaker[speaker]):
                return 1

        rows = []
        for item in queries:
            audio_path = dataset_root / str(item["audio"])
            expected = str(item["expected"])
            role = str(item["role"])
            embedding = service.extract_embedding(audio_path)
            result = None
            if embedding is not None:
                result = registry.identify(
                    embedding,
                    threshold=args.threshold,
                    top_k=args.top_k,
                )

            predicted = "ERROR"
            score = None
            top1_speaker = None
            top1_score = None
            matches = []
            if result is not None:
                predicted = str(result["speaker"])
                score = float(result["score"])
                matches = result["matches"]
                if matches:
                    top1_speaker = str(matches[0]["speaker"])
                    top1_score = float(matches[0]["score"])
            rows.append(
                {
                    "audio": str(item["audio"]),
                    "utterance_id": item["utterance_id"],
                    "source_speaker_id": item["source_speaker_id"],
                    "role": role,
                    "expected": expected,
                    "predicted": predicted,
                    "correct": predicted == expected,
                    "score": score,
                    "top1_speaker": top1_speaker,
                    "top1_score": top1_score,
                    "matches": matches,
                }
            )

    elapsed = time.perf_counter() - started
    known_rows = [item for item in rows if item["role"] == "known"]
    unknown_rows = [item for item in rows if item["role"] == "unknown"]
    known_correct = sum(item["correct"] for item in known_rows)
    known_top1_correct = sum(
        item["top1_speaker"] == item["expected"] for item in known_rows
    )
    known_accepted = sum(
        item["predicted"] not in {UNKNOWN_SPEAKER, "ERROR"} for item in known_rows
    )
    known_rejected = sum(
        item["predicted"] == UNKNOWN_SPEAKER for item in known_rows
    )
    known_misidentified = sum(
        item["predicted"] not in {item["expected"], UNKNOWN_SPEAKER, "ERROR"}
        for item in known_rows
    )
    known_errors = sum(item["predicted"] == "ERROR" for item in known_rows)
    unknown_correct = sum(item["correct"] for item in unknown_rows)
    unknown_false_accepts = sum(
        item["predicted"] not in {UNKNOWN_SPEAKER, "ERROR"} for item in unknown_rows
    )
    unknown_errors = sum(item["predicted"] == "ERROR" for item in unknown_rows)
    total_correct = sum(item["correct"] for item in rows)
    query_errors = sum(item["predicted"] == "ERROR" for item in rows)

    per_speaker = {}
    for item in known_rows:
        speaker = str(item["expected"])
        counters = per_speaker.setdefault(speaker, {"total": 0, "correct": 0})
        counters["total"] += 1
        counters["correct"] += int(item["correct"])
    for counters in per_speaker.values():
        counters["accuracy"] = ratio(counters["correct"], counters["total"])
    macro_known_accuracy = ratio(
        sum(item["accuracy"] for item in per_speaker.values()),
        len(per_speaker),
    )
    known_accuracy = ratio(known_correct, len(known_rows))
    unknown_recall = ratio(unknown_correct, len(unknown_rows))

    metrics = {
        "known_query_count": len(known_rows),
        "known_correct": known_correct,
        "known_open_set_accuracy": known_accuracy,
        "known_macro_accuracy": macro_known_accuracy,
        "known_closed_set_top1_accuracy": ratio(
            known_top1_correct,
            len(known_rows),
        ),
        "known_acceptance_rate": ratio(known_accepted, len(known_rows)),
        "known_false_reject_count": known_rejected,
        "known_false_reject_rate": ratio(known_rejected, len(known_rows)),
        "known_misidentification_count": known_misidentified,
        "known_misidentification_rate": ratio(
            known_misidentified,
            len(known_rows),
        ),
        "known_error_count": known_errors,
        "known_error_rate": ratio(known_errors, len(known_rows)),
        "unknown_query_count": len(unknown_rows),
        "unknown_correct": unknown_correct,
        "unknown_recall": unknown_recall,
        "false_accept_rate": ratio(unknown_false_accepts, len(unknown_rows)),
        "unknown_error_count": unknown_errors,
        "unknown_error_rate": ratio(unknown_errors, len(unknown_rows)),
        "overall_query_count": len(rows),
        "overall_correct": total_correct,
        "overall_open_set_accuracy": ratio(total_correct, len(rows)),
        "balanced_accuracy": 0.5 * (known_accuracy + unknown_recall),
        "query_error_count": query_errors,
        "elapsed_seconds": elapsed,
        "seconds_per_query_including_enrollment": ratio(elapsed, len(rows)),
    }
    report = {
        "evaluation_version": 2,
        "manifest": str(manifest_path),
        "model": {
            "path": str(args.model.resolve()),
            "sha256": model_sha256,
            "preprocessing_id": PREPROCESSING_ID,
            "runtime": backend.runtime_info(),
        },
        "threshold": args.threshold,
        "top_k": args.top_k,
        "metrics": metrics,
        "per_known_speaker": per_speaker,
        "predictions": rows,
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_output, output_path)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "threshold": args.threshold,
                "metrics": metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="评测开放集声纹识别正确率")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    return evaluate(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
