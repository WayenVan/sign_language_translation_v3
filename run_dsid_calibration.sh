#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if [[ ! -f "$PROJECT_DIR/.venv/bin/activate" ]]; then
  echo "Missing virtual environment: $PROJECT_DIR/.venv" >&2
  exit 1
fi

source "$PROJECT_DIR/.venv/bin/activate"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

python -m csi_slt.commands.calibrate_dsid_tau \
  model=base_model \
  calibration.batch_size=2 \
  calibration.num_workers=4 \
  calibration.max_samples=null \
  calibration.output_path=outputs/dsid_calibration/dsid_tau.json
