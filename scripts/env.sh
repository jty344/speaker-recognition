#!/usr/bin/env bash

set -euo pipefail

VOICEPRINT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export VOICEPRINT_ROOT
export TMPDIR="${VOICEPRINT_ROOT}/.tmp"
export PIP_CACHE_DIR="${VOICEPRINT_ROOT}/.cache/pip"
export XDG_CACHE_HOME="${VOICEPRINT_ROOT}/.cache/xdg"
export XDG_CONFIG_HOME="${VOICEPRINT_ROOT}/.runtime/xdg-config"
export XDG_DATA_HOME="${VOICEPRINT_ROOT}/.runtime/xdg-data"
export GST_REGISTRY="${VOICEPRINT_ROOT}/.cache/gstreamer/registry.bin"
export HF_HOME="${VOICEPRINT_ROOT}/.cache/huggingface"
export TORCH_HOME="${VOICEPRINT_ROOT}/.cache/torch"
export VIRTUALENV_OVERRIDE_APP_DATA="${VOICEPRINT_ROOT}/.runtime/virtualenv-app-data"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${VOICEPRINT_ROOT}/src"
export VIRTUAL_ENV="${VOICEPRINT_ROOT}/.runtime/venvs/voiceprint"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
