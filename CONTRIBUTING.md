# Contributing

Thanks for your interest in contributing! A few notes to keep things smooth:

## Development setup
- Clone this repo alongside your React Native macOS app
- Place ExecuTorch binaries under `macos/` as described in README
- Point your app's Podfile at `macos/RNExecuTorchRunner`
- `cd macos && pod install`

## Making changes
- Keep changes small and focused
- Include a brief description in PRs of what and why
- Avoid committing large binary artifacts (xcframeworks, .a files)
- For docs or examples that need binaries, prefer download/build instructions

## Code style
- Objective-C++: follow existing file style; C++17
- TypeScript: strict, small modules, clear names
- Prefer composition over magic build-time logic

## Testing
- Validate `pod install` works on a clean checkout
- Build the macOS example/app and run a quick generation
- If you change Podspec linking, verify no duplicate or missing symbols

## Reporting issues
- Include macOS version, RN macOS version, Xcode version
- Attach key build logs (linker errors) or runtime logs (OperatorMissing, etc.)
- Share model export variant if relevant (xnnpack/simple/portable)

## License
By contributing, you agree your contributions are licensed under the MIT License in this repository.

