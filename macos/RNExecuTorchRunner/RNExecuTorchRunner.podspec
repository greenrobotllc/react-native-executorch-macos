require 'json'

# If you publish this as an npm package, you can read version from package.json
package = JSON.parse(File.read(File.join(__dir__, '../../package.json'))) rescue { 'version' => '0.1.0', 'license' => 'MIT' }

Pod::Spec.new do |s|
  s.name         = "RNExecuTorchRunner"
  s.version      = package['version']
  s.summary      = "React Native bridge for ExecuTorch C++ Runner (Python-free LLM inference)"
  s.homepage     = "https://github.com/YOUR_ORG/react-native-executorch-macos"
  s.license      = package['license'] || "MIT"
  s.author       = { "Green Robot LLC" => "andy.triboletti@gmail.com" }
  s.platform     = :osx, "11.0"
  s.source       = { :path => '.' }

  s.source_files = "*.{h,mm}"
  s.requires_arc = true

  # React Native dependency
  s.dependency "React-Core"

  # Compiler settings
  s.pod_target_xcconfig = {
    'CLANG_CXX_LANGUAGE_STANDARD' => 'c++17',
    'CLANG_CXX_LIBRARY' => 'libc++',
    'CLANG_ENABLE_MODULES' => 'NO',
    'HEADER_SEARCH_PATHS' => [
      '"$(PODS_ROOT)/../../node_modules/react-native-executorch-macos/macos/ExecuTorchFrameworks/executorch.xcframework/macos-arm64/Headers"',
      '"$(PODS_ROOT)/../../node_modules/react-native-executorch-macos/macos/ExecuTorchFrameworks/executorch_llm.xcframework/macos-arm64/Headers"',
      # Local ExecuTorch install (schema + C++ LLM runner headers)
      '"$(PODS_ROOT)/../../node_modules/react-native-executorch-macos/macos/executorch-install/include"',
      '"$(inherited)"'
    ].join(' '),
    'LIBRARY_SEARCH_PATHS' => '$(inherited) ' + [
      '"$(PODS_ROOT)/../../node_modules/react-native-executorch-macos/macos/ExecuTorchFrameworks/executorch.xcframework/macos-arm64"',
      '"$(PODS_ROOT)/../../node_modules/react-native-executorch-macos/macos/ExecuTorchFrameworks/executorch_llm.xcframework/macos-arm64"',
      '"$(PODS_ROOT)/../../node_modules/react-native-executorch-macos/macos/ExecuTorchFrameworks/backend_xnnpack.xcframework/macos-arm64"',
      '"$(PODS_ROOT)/../../node_modules/react-native-executorch-macos/macos/ExecuTorchFrameworks/kernels_llm.xcframework/macos-arm64"',
      '"$(PODS_ROOT)/../../node_modules/react-native-executorch-macos/macos/ExecuTorchFrameworks/kernels_torchao.xcframework/macos-arm64"',
      '"$(PODS_ROOT)/../../node_modules/react-native-executorch-macos/macos/ExecuTorchFrameworks/kernels_optimized.xcframework/macos-arm64"',
      '"$(PODS_ROOT)/../../node_modules/react-native-executorch-macos/macos/ExecuTorchFrameworks/threadpool.xcframework/macos-arm64"',
      # Third-party static libs (tokenizers, absl, etc.) produced by ExecuTorch build
      '"$(PODS_ROOT)/../../node_modules/react-native-executorch-macos/macos/executorch-install/lib"',
      # Swift toolchain compatibility libs (needed when the app has no Swift sources)
      '"$(TOOLCHAIN_DIR)/usr/lib/swift/$(PLATFORM_NAME)"',
      '"$(DEVELOPER_DIR)/Toolchains/XcodeDefault.xctoolchain/usr/lib/swift/$(PLATFORM_NAME)"'
    ].join(' '),
    'OTHER_CPLUSPLUSFLAGS' => '$(inherited)',
    'OTHER_CFLAGS' => '$(inherited)',
    'OTHER_LDFLAGS' => '$(inherited)',
    'GCC_PREPROCESSOR_DEFINITIONS' => '$(inherited) USE_EXECUTORCH_RUNNER=1 C10_USING_CUSTOM_GENERATED_MACROS=1',
    'CLANG_ALLOW_NON_MODULAR_INCLUDES_IN_FRAMEWORK_MODULES' => 'YES',
    'USE_HEADERMAP' => 'YES',
    'ALWAYS_SEARCH_USER_PATHS' => 'YES',
  }

  # Ensure final app target receives link flags to register kernels
  s.user_target_xcconfig = {
    'LIBRARY_SEARCH_PATHS' => '$(inherited) ' + [
      '"$(PODS_ROOT)/../../node_modules/react-native-executorch-macos/macos/ExecuTorchFrameworks/executorch.xcframework/macos-arm64"',
      '"$(PODS_ROOT)/../../node_modules/react-native-executorch-macos/macos/ExecuTorchFrameworks/executorch_llm.xcframework/macos-arm64"',
      '"$(PODS_ROOT)/../../node_modules/react-native-executorch-macos/macos/ExecuTorchFrameworks/backend_xnnpack.xcframework/macos-arm64"',
      '"$(PODS_ROOT)/../../node_modules/react-native-executorch-macos/macos/ExecuTorchFrameworks/kernels_llm.xcframework/macos-arm64"',
      '"$(PODS_ROOT)/../../node_modules/react-native-executorch-macos/macos/ExecuTorchFrameworks/kernels_torchao.xcframework/macos-arm64"',
      '"$(PODS_ROOT)/../../node_modules/react-native-executorch-macos/macos/ExecuTorchFrameworks/kernels_optimized.xcframework/macos-arm64"',
      '"$(PODS_ROOT)/../../node_modules/react-native-executorch-macos/macos/ExecuTorchFrameworks/threadpool.xcframework/macos-arm64"',
      '"$(PODS_ROOT)/../../node_modules/react-native-executorch-macos/macos/executorch-install/lib"',
      # Swift toolchain compatibility libs (needed when the app has no Swift sources)
      '"$(TOOLCHAIN_DIR)/usr/lib/swift/$(PLATFORM_NAME)"',
      '"$(DEVELOPER_DIR)/Toolchains/XcodeDefault.xctoolchain/usr/lib/swift/$(PLATFORM_NAME)"'
    ].join(' '),
    'OTHER_LDFLAGS' => '$(inherited) \
      -Wl,-force_load,$(PODS_ROOT)/../../node_modules/react-native-executorch-macos/macos/ExecuTorchFrameworks/executorch.xcframework/macos-arm64/libexecutorch_macos.a \
      $(PODS_ROOT)/../../node_modules/react-native-executorch-macos/macos/ExecuTorchFrameworks/executorch_llm.xcframework/macos-arm64/libexecutorch_llm_macos.a \
      $(PODS_ROOT)/../../node_modules/react-native-executorch-macos/macos/ExecuTorchFrameworks/threadpool.xcframework/macos-arm64/libthreadpool_macos.a \
      -Wl,-force_load,$(PODS_ROOT)/../../node_modules/react-native-executorch-macos/macos/ExecuTorchFrameworks/backend_xnnpack.xcframework/macos-arm64/libbackend_xnnpack_macos.a \
      -Wl,-force_load,$(PODS_ROOT)/../../node_modules/react-native-executorch-macos/macos/ExecuTorchFrameworks/kernels_llm.xcframework/macos-arm64/libkernels_llm_macos.a \
      -Wl,-force_load,$(PODS_ROOT)/../../node_modules/react-native-executorch-macos/macos/ExecuTorchFrameworks/kernels_torchao.xcframework/macos-arm64/libkernels_torchao_macos.a \
      -Wl,-force_load,$(PODS_ROOT)/../../node_modules/react-native-executorch-macos/macos/ExecuTorchFrameworks/kernels_optimized.xcframework/macos-arm64/libkernels_optimized_macos.a \
      -ltokenizers -lsentencepiece -lre2 -lpcre2-8 -lpcre2-posix \
      -labsl_base -labsl_spinlock_wait -labsl_malloc_internal -labsl_raw_logging_internal -labsl_throw_delegate \
      -labsl_log_internal_message -labsl_log_internal_check_op -labsl_log_internal_format -labsl_log_internal_globals \
      -labsl_log_internal_log_sink_set -labsl_log_internal_proto -labsl_log_internal_nullguard -labsl_log_internal_conditions \
      -labsl_log_severity -labsl_log_sink -labsl_log_entry -labsl_log_globals -labsl_vlog_config_internal -labsl_log_flags \
      -labsl_hash -labsl_low_level_hash -labsl_city -labsl_raw_hash_set -labsl_hashtablez_sampler -labsl_exponential_biased \
      -labsl_str_format_internal -labsl_strings -labsl_strings_internal -labsl_string_view -labsl_int128 \
      -labsl_cord -labsl_cord_internal -labsl_cordz_functions -labsl_cordz_handle -labsl_cordz_info -labsl_cordz_sample_token \
      -labsl_crc_cord_state -labsl_crc32c -labsl_crc_internal -labsl_crc_cpu_detect \
      -labsl_synchronization -labsl_graphcycles_internal -labsl_kernel_timeout_internal \
      -labsl_time -labsl_time_zone -labsl_civil_time -labsl_stacktrace -labsl_symbolize -labsl_debugging_internal \
      -labsl_demangle_internal -labsl_demangle_rust -labsl_decode_rust_punycode -labsl_examine_stack \
      -labsl_failure_signal_handler -labsl_strerror -labsl_flags_internal -labsl_flags_marshalling -labsl_flags_reflection \
      -labsl_flags_config -labsl_flags_program_name -labsl_flags_private_handle_accessor -labsl_flags_commandlineflag \
      -labsl_flags_commandlineflag_internal -labsl_status -labsl_statusor -labsl_utf8_for_code_point \
      -lswiftCompatibility56 -lswiftCompatibilityPacks'
  }
end

