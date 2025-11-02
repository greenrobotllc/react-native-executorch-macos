#!/usr/bin/env bash
# Copy real native sources from AlienTavern into the open-source tree
# Run from repo root (alientavern-v2)
set -euo pipefail

SRC_ROOT="mobile/macos"
DST_ROOT="open-source/react-native-executorch-macos/macos"

mkdir -p "$DST_ROOT/RNExecuTorchRunner" "$DST_ROOT/RNExecuTorch"

cp -v "$SRC_ROOT/RNExecuTorchRunner/RNExecuTorchRunner.h" "$DST_ROOT/RNExecuTorchRunner/"
cp -v "$SRC_ROOT/RNExecuTorchRunner/RNExecuTorchRunner.mm" "$DST_ROOT/RNExecuTorchRunner/"

cp -v "$SRC_ROOT/RNExecuTorch/RNExecuTorch.h" "$DST_ROOT/RNExecuTorch/" || true
cp -v "$SRC_ROOT/RNExecuTorch/RNExecuTorch.mm" "$DST_ROOT/RNExecuTorch/" || true

# Optional: also copy legacy podspec
if [ -f "$SRC_ROOT/RNExecuTorch/RNExecuTorch.podspec" ]; then
  cp -v "$SRC_ROOT/RNExecuTorch/RNExecuTorch.podspec" "$DST_ROOT/RNExecuTorch/"
fi

echo "Done. Replaced wrapper files with actual sources."

