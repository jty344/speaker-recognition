#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

if [[ ! -x "${VIRTUAL_ENV}/bin/python" ]]; then
    echo "Voiceprint runtime not found. Run ./scripts/bootstrap.sh first." >&2
    exit 1
fi

exec "${VIRTUAL_ENV}/bin/python" -m voiceprint "$@"
