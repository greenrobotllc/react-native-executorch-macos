#!/usr/bin/env bash
set -euo pipefail

APP_NAME="RNExecuTorchRunnerExample"

echo "Creating React Native macOS example app: $APP_NAME"

# 1) Initialize RN macOS project
npx @react-native-community/cli@latest init "$APP_NAME" --template react-native@npm:react-native-macos@latest

cd "$APP_NAME"

# 2) Add local package dependency
# Prefer Yarn if available; otherwise npm
if command -v yarn >/dev/null 2>&1; then
  yarn add file:..
else
  npm install --save file:..
fi

# 3) Replace App.tsx with template that calls ExecuTorchRunner
TEMPLATE_PATH="../templates/App.tsx"
if [ -f "$TEMPLATE_PATH" ]; then
  cp "$TEMPLATE_PATH" "App.tsx"
fi

# 4) Add RNExecuTorchRunner pod to macOS Podfile
PODFILE="macos/Podfile"
if ! grep -q "RNExecuTorchRunner" "$PODFILE"; then
  cat >> "$PODFILE" <<'RNPOD'

  # ExecuTorch runner pod (local)
  pod 'RNExecuTorchRunner', :path => '../../macos/RNExecuTorchRunner'
RNPOD
fi

# 5) Remind user to place ExecuTorch assets
cat <<'MSG'

[Manual step]
Place ExecuTorch assets in the package repository (not inside the example app):
  - ../macos/ExecuTorchFrameworks/
  - ../macos/executorch-install/include
  - ../macos/executorch-install/lib

Ensure the RNExecuTorchRunner.podspec paths match these locations.

MSG

# 6) Install pods
cd macos
pod install

echo "\nDone. Open macos/${APP_NAME}.xcworkspace in Xcode and run the macOS target."

