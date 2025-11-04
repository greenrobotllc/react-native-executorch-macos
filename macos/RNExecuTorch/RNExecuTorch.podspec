require 'json'

package = JSON.parse(File.read(File.join(__dir__, '../../package.json')))

Pod::Spec.new do |s|
  s.name         = "RNExecuTorch"
  s.version      = package['version']
  s.summary      = "React Native bridge for PyTorch ExecuTorch on macOS"
  s.homepage     = "https://github.com/greenrobotllc/react-native-executorch-macos"
  s.license      = package['license']
  s.author       = { "Green Robot LLC" => "andy.triboletti@gmail.com" }
  s.platform     = :osx, "11.0"
  if ENV['LOCAL_POD']
    s.source       = { :path => '.' }
  else
    s.source       = { :git => "https://github.com/greenrobotllc/react-native-executorch-macos.git", :tag => "v#{s.version}" }
  end
  
  s.source_files = "**/*.{h,m,mm}"
  s.requires_arc = true
  
  # React Native dependencies
  s.dependency "React-Core"
  
  # ExecuTorch dependencies
  # We'll use CocoaPods to manage ExecuTorch
  s.dependency "executorch", "~> 0.7.0"
  
  # Compiler flags for C++17
  s.pod_target_xcconfig = {
    'CLANG_CXX_LANGUAGE_STANDARD' => 'c++17',
    'CLANG_CXX_LIBRARY' => 'libc++',
    'OTHER_CPLUSPLUSFLAGS' => '-DFOLLY_NO_CONFIG -DFOLLY_MOBILE=1 -DFOLLY_USE_LIBCPP=1',
  }
end

