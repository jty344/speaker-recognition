#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

DATASET_DIR="${VOICEPRINT_ROOT}/datasets/mini_librispeech_open_set"
ARCHIVE_DIR="${DATASET_DIR}/archives"
SOURCE_DIR="${DATASET_DIR}/source"
ARCHIVE="${ARCHIVE_DIR}/dev-clean-2.tar.gz"
ARCHIVE_PART="${ARCHIVE}.part"
ARCHIVE_URL="https://www.openslr.org/resources/31/dev-clean-2.tar.gz"
ARCHIVE_MD5="6d7ab67ac6a1d2c993d050e16d61080d"

mkdir -p \
    "${ARCHIVE_DIR}" \
    "${SOURCE_DIR}" \
    "${DATASET_DIR}/mp3" \
    "${DATASET_DIR}/protocol" \
    "${DATASET_DIR}/results"

archive_is_valid() {
    [[ -f "${ARCHIVE}" ]] \
        && printf '%s  %s\n' "${ARCHIVE_MD5}" "${ARCHIVE}" \
            | md5sum --check --status
}

if ! archive_is_valid; then
    curl -fL --retry 4 --continue-at - \
        --output "${ARCHIVE_PART}" \
        "${ARCHIVE_URL}"
    printf '%s  %s\n' "${ARCHIVE_MD5}" "${ARCHIVE_PART}" \
        | md5sum --check --status
    mv "${ARCHIVE_PART}" "${ARCHIVE}"
fi

if ! find "${SOURCE_DIR}/LibriSpeech/dev-clean-2" \
    -type f -name '*.flac' -print -quit 2>/dev/null | grep -q .; then
    tar --extract --gzip \
        --file "${ARCHIVE}" \
        --directory "${SOURCE_DIR}" \
        --no-same-owner \
        --no-same-permissions
fi

python "${SCRIPT_DIR}/build_open_set_dataset.py" \
    --dataset-root "${DATASET_DIR}"
