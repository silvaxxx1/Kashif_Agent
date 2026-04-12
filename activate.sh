#!/usr/bin/env bash
# activate.sh — activate the Kashif isolated environment
#
# Usage (must be sourced, not executed):
#   source activate.sh
#
# What it does:
#   1. Deactivates any currently active venv
#   2. Activates kashif_core/.venv
#   3. Verifies the correct Python is active

# Must be sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ERROR: source this script, don't run it directly."
    echo "       source activate.sh"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Deactivate any active venv
if command -v deactivate &>/dev/null; then
    deactivate
fi
unset VIRTUAL_ENV

# Activate Kashif venv
source "$SCRIPT_DIR/.venv/bin/activate"

echo "Kashif env active → $(which python) ($(python --version))"
