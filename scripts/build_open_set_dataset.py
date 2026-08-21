import argparse
import hashlib
import json
import logging
import os
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from voiceprint.audio import decode_audio, trim_silence
from voiceprint.config import (
    DEFAULT_THRESHOLD,
    MIN_SPEECH_SECONDS,
    MODEL_PATH,
    PREPROCESSING_ID,
    SAMPLE_RATE,
)


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets" / "mini_librispeech_open_set"
ARCHIVE_MD5 = "6d7ab67ac6a1d2c993d050e16d61080d"
ARCHIVE_URL = "https://www.openslr.org/resources/31/dev-clean-2.tar.gz"
PROTOCOL_MIN_SECONDS = MIN_SPEECH_SECONDS + 0.5


def file_sha256(path: Path) -> str | None:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        LOGGER.error("计算文件 SHA-256 失败: %s", error)
        return None
    return digest.hexdigest()


def stable_rank(seed: int, *values: str) -> str:
    material = ":".join([str(seed), *values])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def discover_audio(source_root: Path) -> dict[str, list[Path]]:
    by_speaker: dict[str, list[Path]] = defaultdict(list)
    for audio_path in source_root.glob("*/*/*.flac"):
        speaker_id = audio_path.parent.parent.name
        if audio_path.stem.split("-", maxsplit=1)[0] == speaker_id:
            by_speaker[speaker_id].append(audio_path)
    return dict(by_speaker)


def effective_duration(audio_path: Path) -> float | None:
    samples = decode_audio(audio_path, sample_rate=SAMPLE_RATE)
    if samples is None:
        return None
    speech = trim_silence(samples, sample_rate=SAMPLE_RATE)
    duration = speech.size / float(SAMPLE_RATE)
    if duration < PROTOCOL_MIN_SECONDS:
        return None
    return duration


def select_usable_audio(
    candidates: list[Path],
    required_files: int,
    seed: int,
    scope: str,
) -> list[tuple[Path, float]]:
    usable = []
    ordered = sorted(
        candidates,
        key=lambda path: stable_rank(seed, scope, path.name),
    )
    for audio_path in ordered:
        duration = effective_duration(audio_path)
        if duration is not None:
            usable.append((audio_path, duration))
        if len(usable) == required_files:
            break
    return usable


def select_known_speakers(
    by_speaker: dict[str, list[Path]],
    speaker_count: int,
    enroll_files: int,
    query_files: int,
    seed: int,
) -> list[
    tuple[
        str,
        list[tuple[Path, float]],
        list[tuple[Path, float]],
    ]
]:
    selected = []
    speaker_ids = sorted(
        by_speaker,
        key=lambda speaker_id: stable_rank(seed, "known-speaker", speaker_id),
    )
    for speaker_id in speaker_ids:
        by_chapter: dict[str, list[Path]] = defaultdict(list)
        for audio_path in by_speaker[speaker_id]:
            by_chapter[audio_path.parent.name].append(audio_path)
        if len(by_chapter) < 2:
            continue
        chapter_ids = sorted(
            by_chapter,
            key=lambda chapter_id: stable_rank(
                seed,
                "known-chapter",
                speaker_id,
                chapter_id,
            ),
        )
        eligible_chapters = []
        required_files = max(enroll_files, query_files)
        for chapter_id in chapter_ids:
            usable = select_usable_audio(
                by_chapter[chapter_id],
                required_files=required_files,
                seed=seed,
                scope=f"known:{speaker_id}:{chapter_id}",
            )
            if len(usable) == required_files:
                eligible_chapters.append(usable)
            if len(eligible_chapters) == 2:
                break
        if len(eligible_chapters) == 2:
            selected.append(
                (
                    speaker_id,
                    eligible_chapters[0][:enroll_files],
                    eligible_chapters[1][:query_files],
                )
            )
        if len(selected) == speaker_count:
            break
    return selected


def select_unknown_speakers(
    by_speaker: dict[str, list[Path]],
    excluded_speakers: set[str],
    speaker_count: int,
    query_files: int,
    seed: int,
) -> list[tuple[str, list[tuple[Path, float]]]]:
    selected = []
    speaker_ids = sorted(
        set(by_speaker) - excluded_speakers,
        key=lambda speaker_id: stable_rank(seed, "unknown-speaker", speaker_id),
    )
    for speaker_id in speaker_ids:
        usable = select_usable_audio(
            by_speaker[speaker_id],
            required_files=query_files,
            seed=seed,
            scope=f"unknown:{speaker_id}",
        )
        if len(usable) == query_files:
            selected.append((speaker_id, usable))
        if len(selected) == speaker_count:
            break
    return selected


def transcode_mp3(source: Path, target: Path) -> bool:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size > 0:
        return True
    partial = target.with_suffix(target.suffix + ".part")
    command = [
        "gst-launch-1.0",
        "-q",
        "filesrc",
        f"location={source}",
        "!",
        "decodebin",
        "!",
        "audioconvert",
        "!",
        "audioresample",
        "!",
        "audio/x-raw,format=S16LE,rate=16000,channels=1",
        "!",
        "lamemp3enc",
        "target=bitrate",
        "bitrate=128",
        "cbr=true",
        "!",
        "filesink",
        f"location={partial}",
    ]
    result = subprocess.run(command, check=False, shell=False)
    if result.returncode != 0 or not partial.is_file():
        LOGGER.error("转码 MP3 失败: %s", source)
        return False
    os.replace(partial, target)
    return True


def make_record(
    dataset_root: Path,
    source_root: Path,
    source: Path,
    duration: float,
    speaker_id: str,
    role: str,
    expected: str,
) -> dict[str, object] | None:
    speaker_name = f"speaker_{speaker_id}"
    category = "enroll" if role == "enroll" else f"queries/{role}"
    target = dataset_root / "mp3" / category / speaker_name / f"{source.stem}.mp3"
    if not transcode_mp3(source, target):
        return None
    return {
        "audio": str(target.relative_to(dataset_root)),
        "source_audio": str(source.relative_to(source_root)),
        "utterance_id": source.stem,
        "source_speaker_id": speaker_id,
        "speaker": speaker_name,
        "chapter_id": source.parent.name,
        "role": role,
        "expected": expected,
        "effective_duration_seconds": round(duration, 6),
    }


def build_protocol(args: argparse.Namespace) -> int:
    dataset_root = args.dataset_root.resolve()
    source_root = dataset_root / "source" / "LibriSpeech" / "dev-clean-2"
    archive = dataset_root / "archives" / "dev-clean-2.tar.gz"
    if not source_root.is_dir() or not archive.is_file():
        LOGGER.error("Mini LibriSpeech 尚未下载或解压: %s", dataset_root)
        return 1

    archive_sha256 = file_sha256(archive)
    model_sha256 = file_sha256(MODEL_PATH)
    if archive_sha256 is None or model_sha256 is None:
        return 1
    by_speaker = discover_audio(source_root)
    known = select_known_speakers(
        by_speaker,
        speaker_count=args.known_speakers,
        enroll_files=args.enroll_files,
        query_files=args.query_files,
        seed=args.seed,
    )
    if len(known) != args.known_speakers:
        LOGGER.error(
            "满足跨 chapter 注册/查询条件的说话人不足: %d/%d",
            len(known),
            args.known_speakers,
        )
        return 1
    known_ids = {item[0] for item in known}
    unknown = select_unknown_speakers(
        by_speaker,
        excluded_speakers=known_ids,
        speaker_count=args.unknown_speakers,
        query_files=args.query_files,
        seed=args.seed,
    )
    if len(unknown) != args.unknown_speakers:
        LOGGER.error(
            "满足未知人查询条件的说话人不足: %d/%d",
            len(unknown),
            args.unknown_speakers,
        )
        return 1

    enrollment = []
    queries = []
    for speaker_id, enroll_audio, query_audio in known:
        speaker_name = f"speaker_{speaker_id}"
        for source, duration in enroll_audio:
            record = make_record(
                dataset_root,
                source_root,
                source,
                duration,
                speaker_id,
                "enroll",
                speaker_name,
            )
            if record is None:
                return 1
            enrollment.append(record)
        for source, duration in query_audio:
            record = make_record(
                dataset_root,
                source_root,
                source,
                duration,
                speaker_id,
                "known",
                speaker_name,
            )
            if record is None:
                return 1
            queries.append(record)

    for speaker_id, audio_files in unknown:
        for source, duration in audio_files:
            record = make_record(
                dataset_root,
                source_root,
                source,
                duration,
                speaker_id,
                "unknown",
                "UNKNOWN",
            )
            if record is None:
                return 1
            queries.append(record)

    source_paths = [item["source_audio"] for item in enrollment + queries]
    if len(source_paths) != len(set(source_paths)):
        LOGGER.error("评测协议存在跨角色重复音频")
        return 1

    manifest = {
        "protocol_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "Mini LibriSpeech dev-clean-2",
            "source": "OpenSLR SLR31",
            "url": ARCHIVE_URL,
            "license": "CC BY 4.0",
            "archive_bytes": archive.stat().st_size,
            "archive_md5": ARCHIVE_MD5,
            "archive_sha256": archive_sha256,
            "source_speakers": len(by_speaker),
            "source_utterances": sum(len(items) for items in by_speaker.values()),
        },
        "split": {
            "seed": args.seed,
            "known_speakers": [f"speaker_{item[0]}" for item in known],
            "unknown_source_speakers": [item[0] for item in unknown],
            "known_enroll_query_chapter_isolation": True,
            "minimum_source_effective_seconds": PROTOCOL_MIN_SECONDS,
            "enroll_files_per_known_speaker": args.enroll_files,
            "query_files_per_speaker": args.query_files,
            "enrollment_count": len(enrollment),
            "known_query_count": sum(item["role"] == "known" for item in queries),
            "unknown_query_count": sum(item["role"] == "unknown" for item in queries),
        },
        "evaluation_defaults": {
            "model_sha256": model_sha256,
            "preprocessing_id": PREPROCESSING_ID,
            "threshold": DEFAULT_THRESHOLD,
            "threshold_source": "demo_default_uncalibrated",
        },
        "enrollment": enrollment,
        "queries": queries,
    }
    protocol_path = dataset_root / "protocol" / "open_set_manifest.json"
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = protocol_path.with_suffix(protocol_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, protocol_path)
    print(
        json.dumps(
            {
                "manifest": str(protocol_path),
                "source_speakers": len(by_speaker),
                "source_utterances": sum(len(items) for items in by_speaker.values()),
                "known_speakers": args.known_speakers,
                "unknown_speakers": args.unknown_speakers,
                "enrollment": len(enrollment),
                "queries": len(queries),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从 Mini LibriSpeech 真值标签构造开放集 MP3 评测协议",
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--known-speakers", type=int, default=8)
    parser.add_argument("--unknown-speakers", type=int, default=8)
    parser.add_argument("--enroll-files", type=int, default=3)
    parser.add_argument("--query-files", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260821)
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    return build_protocol(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
