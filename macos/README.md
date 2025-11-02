# macOS integration notes

This folder contains the CocoaPods pod for the native macOS bridge (RNExecuTorchRunner).

## Place ExecuTorch assets here

- ExecuTorch xcframeworks under `ExecuTorchFrameworks/`
  - executorch.xcframework
  - executorch_llm.xcframework
  - backend_xnnpack.xcframework
  - kernels_llm.xcframework
  - kernels_optimized.xcframework
  - kernels_torchao.xcframework
  - threadpool.xcframework

- ExecuTorch install artifacts under `executorch-install/`
  - include/  (headers: executorch/... including schema + LLM runner headers)
  - lib/      (tokenizers, sentencepiece, re2, pcre2-8, pcre2-posix, absl_*)

## Using the pod in your app

In your app's macOS Podfile:

```ruby
pod 'RNExecuTorchRunner', :path => '../path/to/react-native-executorch-macos/macos/RNExecuTorchRunner'
```

Then:

```bash
cd macos
pod install
```

The Podspec sets header and library search paths and force-loads the ExecuTorch libraries that register kernels/backends to avoid missing-operator errors.

