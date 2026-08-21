#!/usr/bin/env bash

set -euo pipefail

VOICEPRINT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BOOTSTRAP_DIR="${VOICEPRINT_ROOT}/.runtime/bootstrap"
VENV_DIR="${VOICEPRINT_ROOT}/.runtime/venvs/voiceprint"
MODEL_DIR="${VOICEPRINT_ROOT}/models/wespeaker-cnceleb-resnet34-lm"
GST_RUNTIME_DIR="${VOICEPRINT_ROOT}/.runtime/gstreamer"
GST_PACKAGE_DIR="${GST_RUNTIME_DIR}/packages"
GST_EXTRACT_DIR="${GST_RUNTIME_DIR}/extracted"
GST_PLUGIN_DIR="${GST_RUNTIME_DIR}/plugins"
GST_LIBRARY_DIR="${GST_RUNTIME_DIR}/lib"
GST_REQUIREMENTS="${VOICEPRINT_ROOT}/requirements/gstreamer-aac.txt"

export TMPDIR="${VOICEPRINT_ROOT}/.tmp"
export PIP_CACHE_DIR="${VOICEPRINT_ROOT}/.cache/pip"
export XDG_CACHE_HOME="${VOICEPRINT_ROOT}/.cache/xdg"
export XDG_CONFIG_HOME="${VOICEPRINT_ROOT}/.runtime/xdg-config"
export XDG_DATA_HOME="${VOICEPRINT_ROOT}/.runtime/xdg-data"
export GST_REGISTRY="${VOICEPRINT_ROOT}/.cache/gstreamer/registry.bin"
export GST_PLUGIN_PATH_1_0="${GST_PLUGIN_DIR}${GST_PLUGIN_PATH_1_0:+:${GST_PLUGIN_PATH_1_0}}"
export LD_LIBRARY_PATH="${GST_LIBRARY_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export HF_HOME="${VOICEPRINT_ROOT}/.cache/huggingface"
export TORCH_HOME="${VOICEPRINT_ROOT}/.cache/torch"
export VIRTUALENV_OVERRIDE_APP_DATA="${VOICEPRINT_ROOT}/.runtime/virtualenv-app-data"
export PYTHONNOUSERSITE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1

mkdir -p \
    "${TMPDIR}" \
    "${PIP_CACHE_DIR}" \
    "${XDG_CACHE_HOME}" \
    "${XDG_CONFIG_HOME}" \
    "${XDG_DATA_HOME}" \
    "${VOICEPRINT_ROOT}/.cache/gstreamer" \
    "${HF_HOME}" \
    "${TORCH_HOME}" \
    "${BOOTSTRAP_DIR}" \
    "${VIRTUALENV_OVERRIDE_APP_DATA}" \
    "${VOICEPRINT_ROOT}/.runtime/venvs" \
    "${GST_PACKAGE_DIR}" \
    "${GST_EXTRACT_DIR}" \
    "${GST_PLUGIN_DIR}" \
    "${GST_LIBRARY_DIR}" \
    "${MODEL_DIR}"

command -v gst-launch-1.0 >/dev/null
command -v gst-discoverer-1.0 >/dev/null
command -v gst-inspect-1.0 >/dev/null
command -v dpkg-deb >/dev/null

download_verified() {
    local url="$1"
    local target="$2"
    local expected_sha256="$3"
    local partial="${target}.part"

    if [[ -f "${target}" ]] \
        && printf '%s  %s\n' "${expected_sha256}" "${target}" \
            | sha256sum --check --status; then
        return
    fi

    curl -fL --retry 2 -o "${partial}" "${url}"
    printf '%s  %s\n' "${expected_sha256}" "${partial}" \
        | sha256sum --check --status
    mv "${partial}" "${target}"
}

install_gstreamer_aac() {
    local element
    local faad_details
    local filename
    local expected_sha256
    local url

    while read -r filename expected_sha256 url; do
        if [[ -z "${filename}" || "${filename}" == \#* ]]; then
            continue
        fi
        download_verified \
            "${url}" \
            "${GST_PACKAGE_DIR}/${filename}" \
            "${expected_sha256}"
        dpkg-deb --extract \
            "${GST_PACKAGE_DIR}/${filename}" \
            "${GST_EXTRACT_DIR}"
    done < "${GST_REQUIREMENTS}"

    install -m 0644 \
        "${GST_EXTRACT_DIR}/usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstfaad.so" \
        "${GST_PLUGIN_DIR}/libgstfaad.so"
    install -m 0644 \
        "${GST_EXTRACT_DIR}/usr/lib/x86_64-linux-gnu/libfaad.so.2.0.0" \
        "${GST_LIBRARY_DIR}/libfaad.so.2.0.0"
    ln -sfn "libfaad.so.2.0.0" "${GST_LIBRARY_DIR}/libfaad.so.2"

    if ldd "${GST_PLUGIN_DIR}/libgstfaad.so" | grep -F "not found"; then
        echo "GStreamer AAC plugin has unresolved shared libraries." >&2
        return 1
    fi

    for element in \
        aacparse \
        audioconvert \
        audioresample \
        decodebin \
        flacparse \
        mpegaudioparse \
        wavparse; do
        gst-inspect-1.0 "${element}" >/dev/null
    done

    faad_details="$(gst-inspect-1.0 faad)"
    if [[ "${faad_details}" != *"${GST_PLUGIN_DIR}/libgstfaad.so"* ]]; then
        echo "GStreamer did not load the workspace FAAD plugin." >&2
        return 1
    fi
}

install_gstreamer_aac

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    python3 -m pip install \
        --upgrade \
        --target "${BOOTSTRAP_DIR}" \
        virtualenv==20.26.6
    PYTHONPATH="${BOOTSTRAP_DIR}" python3 -m virtualenv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install \
    --requirement "${VOICEPRINT_ROOT}/requirements/runtime.txt" \
    --requirement "${VOICEPRINT_ROOT}/requirements/dev.txt"
"${VENV_DIR}/bin/python" -m pip install \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --requirement "${VOICEPRINT_ROOT}/requirements/audio-fbank.txt"

download_verified \
    "https://huggingface.co/Wespeaker/wespeaker-cnceleb-resnet34-LM/resolve/main/cnceleb_resnet34_LM.onnx?download=true" \
    "${MODEL_DIR}/cnceleb_resnet34_LM.onnx" \
    "e7584940aeac8d5512d875e58ce6c09ba4ddad65d8128e1dac0d93aadd087ebb"
download_verified \
    "https://huggingface.co/Wespeaker/wespeaker-cnceleb-resnet34-LM/resolve/main/config.yaml?download=true" \
    "${MODEL_DIR}/config.yaml" \
    "deef9f002d895dbdb3748768acd4d588459570bd85e9d3f235cfe5631355578e"

(
    cd "${MODEL_DIR}"
    sha256sum --check CHECKSUMS.sha256
)

"${VENV_DIR}/bin/python" -m pip list --format=freeze \
    > "${VOICEPRINT_ROOT}/requirements/runtime-lock.txt"
"${VENV_DIR}/bin/python" -c \
    'import numpy, onnx, onnxruntime, torch, torchaudio; print("Python runtime imports: OK")'
"${VENV_DIR}/bin/python" -c \
    'import onnx, sys; onnx.checker.check_model(sys.argv[1]); print("ONNX model check: OK")' \
    "${MODEL_DIR}/cnceleb_resnet34_LM.onnx"

echo "Runtime: ${VENV_DIR}"
echo "Model:   ${MODEL_DIR}/cnceleb_resnet34_LM.onnx"
echo "AAC:     ${GST_PLUGIN_DIR}/libgstfaad.so"
