#!/bin/bash
# Populate triton_repo/*/1/model.onnx from the exported ONNX files.
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC=${RPM_ONNX_DIR:-$HERE/onnx}
for m in edge_rp90 edge_rp90_int8; do
  install -D "$SRC/$m.onnx" "$HERE/triton_repo/$m/1/model.onnx"
  echo "  $m <- $SRC/$m.onnx"
done
echo "repository ready: $HERE/triton_repo"
