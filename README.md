# react-native-executorch-macos

Python-free LLM inference for React Native macOS using ExecuTorch’s C++ Runner API.

This package exposes a native module (ExecuTorchRunner) that loads a .pte model and streams tokens back to JS. It links prebuilt ExecuTorch xcframeworks (executorch, executorch_llm, kernels_llm, kernels_optimized, kernels_torchao, backend_xnnpack, threadpool) and uses ExecuTorch’s native tokenizers (HuggingFace / SentencePiece).

Status: macOS (Apple Silicon, arm64) supported. iOS/Android out of scope here (use Meta’s react-native-executorch for those).

## Features
- No Python on device (C++ Runner via TextLLMRunner)
- Streaming tokens via RN events (onToken, onComplete, onError)
- XNNPACK backend and LLM custom kernels (sdpa/update_cache)
- Fallback logic to load simplified/portable variants if a model misses operators
- Diagnostics: logs registered backends/kernels and can inspect PTE operators (schema-based when available)

## Requirements
- macOS 12+
- Apple Silicon (arm64)
- React Native macOS 0.74+ (matching your app)
- ExecuTorch 1.0 built locally to produce:
  - Prebuilt xcframeworks or binaries for:
    - executorch.xcframework
    - executorch_llm.xcframework
    - backend_xnnpack.xcframework
    - kernels_llm.xcframework
    - kernels_optimized.xcframework (contains cpublas)
    - kernels_torchao.xcframework
    - threadpool.xcframework
  - install/include and install/lib with third‑party libs:
    - libtokenizers.a, libsentencepiece.a, libre2.a, libpcre2-8.a, libpcre2-posix.a
    - absl libraries (labsl_*) built by ExecuTorch

## Repository layout (recommended)
```
react-native-executorch-macos/
  package.json
  src/
    ExecuTorchRunner.ts
    index.ts
  macos/
    RNExecuTorchRunner/
      RNExecuTorchRunner.h
      RNExecuTorchRunner.mm
      RNExecuTorchRunner.podspec
    ExecuTorchFrameworks/                  # Place execuTorch *.xcframework here (not committed by default)
    executorch-install/
      include/                             # from your `executorch-build/install/include`
      lib/                                 # from your `executorch-build/install/lib` (tokenizers, absl, etc.)
  LICENSE
  README.md
  CONTRIBUTING.md
  CODE_OF_CONDUCT.md
```

> Note: We do not ship the large ExecuTorch binaries in this repo. Add them locally under macos/ as shown above, or automate fetching them in your environment.

## Install in an app (CocoaPods)

1) Add the pod to your macOS Podfile (pointing at this repo’s macos/RNExecuTorchRunner):

```ruby
pod 'RNExecuTorchRunner', :path => '../relative/path/to/react-native-executorch-macos/macos/RNExecuTorchRunner'
```

2) Ensure the ExecuTorch binaries and headers are in the expected locations inside this repo:
- macos/ExecuTorchFrameworks/(each .xcframework)
- macos/executorch-install/include (headers)
- macos/executorch-install/lib (tokenizers + absl)

3) Run:
```bash
cd macos
pod install
```

That’s it. The RNExecuTorchRunner pod’s xcconfig will:
- Add header/library search paths for ExecuTorch xcframeworks and executorch-install
- Force‑load the static libraries that register kernels/backends to avoid missing-operator issues

## TypeScript usage

```ts
import { ExecuTorchRunner } from 'react-native-executorch-macos';

// Load model and tokenizer
const runner = await ExecuTorchRunner.loadModel({
  modelPath: '/absolute/path/to/model.pte',
  tokenizerPath: '/absolute/path/to/tokenizer.json',
  tokenizerType: 'huggingface', // or 'sentencepiece'
});

// Generate with streaming callbacks
const text = await runner.generate(
  'Hello there!',
  { maxNewTokens: 64, temperature: 0.8 },
  {
    onToken: (t) => process.stdout.write(t),
    onComplete: (stats) => console.log('\nDone:', stats),
    onError: (e) => console.error('Error:', e),
  },
);
```

## Exporting/placing ExecuTorch binaries
- Build ExecuTorch with LLM runner enabled (so TextLLMRunner and tokenizers are produced)
- Copy the produced xcframeworks into macos/ExecuTorchFrameworks
- Copy install/include and install/lib into macos/executorch-install
- Verify the Podspec paths match (they default to these relative locations)

Required xcframeworks (macos-arm64 slice must exist):
- executorch, executorch_llm, backend_xnnpack, kernels_llm, kernels_optimized, kernels_torchao, threadpool

Required static libs under executorch-install/lib:
- tokenizers, sentencepiece, re2, pcre2-8, pcre2-posix, and the absl_* set produced by your build

## Notes and tips
- If you see OperatorMissing on your model, ensure kernels_llm and kernels_optimized are force‑loaded (our Podspec does this).
- If you see undefined cpublas::gemm, ensure kernels_optimized is present (cpublas lives there).
- If tokenizers fail to link, confirm executorch-install/lib contains those third‑party libs and the Podspec is pointing at it.
- Logs: the module installs an ExecuTorch log sink so ET_LOG messages appear in RN logs.

## License
MIT

## Contributing
PRs welcome! See CONTRIBUTING.md. Be mindful of large binary assets; prefer fetch/build scripts or documentation over committing binaries.

