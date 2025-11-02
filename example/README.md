# Example app (macOS)

This folder contains scripts and notes to spin up a tiny React Native macOS app that uses the RNExecuTorchRunner pod.

We don’t commit a full app to keep this repo light. Use the script below to generate one locally.

## Prereqs
- Node + Yarn/NPM
- Xcode 15+
- Ruby + CocoaPods (`gem install cocoapods`)
- react-native-macos CLI available (npx will fetch it)
- ExecuTorch binaries/headers available on disk

## Create the app

```bash
# From repo root
cd example
./create-example.sh
```

This will:
- Initialize a new React Native macOS app `RNExecuTorchRunnerExample`
- Add a dependency on this local package
- Add the CocoaPods entry for `RNExecuTorchRunner`
- Remind you where to place ExecuTorch frameworks/headers/libs
- Run `pod install` for the macOS workspace

## Run

```bash
cd RNExecuTorchRunnerExample
# Open the workspace in Xcode and run the macOS target
open macos/RNExecuTorchRunnerExample.xcworkspace
```

When the app starts, App.tsx will attempt to load a model and generate a short sample (you may need to update the file paths).

## Template App.tsx

See `templates/App.tsx` for a minimal component that uses this package.

