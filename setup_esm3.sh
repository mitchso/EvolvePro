#!/bin/bash

# Print that the script is running:
echo "Setting up ESM3 environment..."

#!/usr/bin/env bash
set -euo pipefail

BASE="envs/esm3.yml"
LINUX_OVERLAY="envs/plm_linux.yml"

if [[ "$(uname)" == "Darwin" ]]; then
    echo "macOS detected"
    conda env create -f "$BASE"
else
    echo "Linux/Windows OS detected"
    # conda-merge combines yml files cleanly before creating
    pip install conda-merge --quiet
    conda-merge "$BASE" "$LINUX_OVERLAY" | conda env create -f -
fi

echo "ESM3 environment setup complete!"