#!/usr/bin/env bash

set -euo pipefail

VOICEPRINT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_DIR="${VOICEPRINT_ROOT}/data/test_assets/upstream"
GENERATED_DIR="${VOICEPRINT_ROOT}/data/test_assets/generated"
SPEECHBRAIN_COMMIT="e5cb1f65b940634215650aa1171e0440d0808123"

export TMPDIR="${VOICEPRINT_ROOT}/.tmp"
export XDG_CACHE_HOME="${VOICEPRINT_ROOT}/.cache/xdg"
export XDG_CONFIG_HOME="${VOICEPRINT_ROOT}/.runtime/xdg-config"
export XDG_DATA_HOME="${VOICEPRINT_ROOT}/.runtime/xdg-data"
export GST_REGISTRY="${VOICEPRINT_ROOT}/.cache/gstreamer/registry.bin"

mkdir -p \
    "${UPSTREAM_DIR}" \
    "${GENERATED_DIR}" \
    "${TMPDIR}" \
    "${XDG_CACHE_HOME}" \
    "${XDG_CONFIG_HOME}" \
    "${XDG_DATA_HOME}" \
    "${VOICEPRINT_ROOT}/.cache/gstreamer"

download_sample() {
    local name="$1"
    local expected_sha256="$2"
    local target="${UPSTREAM_DIR}/${name}.wav"
    local partial="${target}.part"

    if [[ -f "${target}" ]] \
        && printf '%s  %s\n' "${expected_sha256}" "${target}" \
            | sha256sum --check --status; then
        return
    fi

    curl -fL --retry 2 -o "${partial}" \
        "https://raw.githubusercontent.com/speechbrain/speechbrain/${SPEECHBRAIN_COMMIT}/tests/samples/ASR/${name}.wav"
    printf '%s  %s\n' "${expected_sha256}" "${partial}" \
        | sha256sum --check --status
    mv "${partial}" "${target}"
}

encode_mp3() {
    local name="$1"
    local target="${GENERATED_DIR}/${name}.mp3"
    local partial="${target}.part"

    gst-launch-1.0 -q \
        filesrc "location=${UPSTREAM_DIR}/${name}.wav" \
        ! wavparse ! audioconvert ! audioresample \
        ! audio/x-raw,format=S16LE,rate=16000,channels=1 \
        ! lamemp3enc target=bitrate bitrate=128 cbr=true \
        ! filesink "location=${partial}"
    mv "${partial}" "${target}"
}

download_sample spk1_snt1 \
    "2f7315ccf543b6368528b098cd895098c154f222cd55ff7b5f66b50feb383c56"
download_sample spk1_snt2 \
    "e1f125a318f4026991bf78f5c41481be0fbcf9b3b2fd72284f9824cb851c3034"
download_sample spk2_snt1 \
    "5f3f534ba99f159358c6ee2e4fb5e42b3b9399f37f4e290651405c76095f26f0"
encode_mp3 spk1_snt1
encode_mp3 spk1_snt2
encode_mp3 spk2_snt1

TWICE_TARGET="${GENERATED_DIR}/spk2_snt1_twice.mp3"
TWICE_PARTIAL="${TWICE_TARGET}.part"
gst-launch-1.0 -q \
    concat name=concat_audio \
    ! audioconvert ! audioresample \
    ! audio/x-raw,format=S16LE,rate=16000,channels=1 \
    ! lamemp3enc target=bitrate bitrate=128 cbr=true \
    ! filesink "location=${TWICE_PARTIAL}" \
    filesrc "location=${UPSTREAM_DIR}/spk2_snt1.wav" ! wavparse ! concat_audio. \
    filesrc "location=${UPSTREAM_DIR}/spk2_snt1.wav" ! wavparse ! concat_audio.
mv "${TWICE_PARTIAL}" "${TWICE_TARGET}"

echo "Generated MP3 files: ${GENERATED_DIR}"
