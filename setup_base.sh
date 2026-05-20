#!/bin/bash

# Print that the script is running:
echo "Setting up base environment..."

set -euo pipefail

BASE="envs/base.yml"

conda env create -f "$BASE"

echo "Base environment setup complete!"