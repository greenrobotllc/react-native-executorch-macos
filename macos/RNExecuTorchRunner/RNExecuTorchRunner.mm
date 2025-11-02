//
//  RNExecuTorchRunner.mm
//  React Native ExecuTorch Runner
//
//  Python-free LLM inference using ExecuTorch C++ Runner API
//

#import "RNExecuTorchRunner.h"
#import <React/RCTLog.h>

// ExecuTorch C++ headers (prefer xcframework headers; fall back to local build includes for LLM)
// LLM Runner headers are not shipped in the xcframework headers; include from local install include
#if __has_include(<executorch/extension/llm/runner/text_llm_runner.h>)
#import <executorch/extension/llm/runner/text_llm_runner.h>
#else
#import "../../executorch-build/install/include/executorch/extension/llm/runner/text_llm_runner.h"
#endif

#if __has_include(<executorch/extension/llm/runner/llm_runner_helper.h>)
#import <executorch/extension/llm/runner/llm_runner_helper.h>
#else
#import "../../executorch-build/install/include/executorch/extension/llm/runner/llm_runner_helper.h"
#endif

#if __has_include(<executorch/extension/llm/sampler/sampler.h>)
#import <executorch/extension/llm/sampler/sampler.h>
#else
#import "../../executorch-build/install/include/executorch/extension/llm/sampler/sampler.h"
#endif

// Core/runtime/module headers: use ExecuTorch (xcframework) first, fallback to local
#if __has_include(<ExecuTorch/extension/module/module.h>)
#import <ExecuTorch/extension/module/module.h>
#else
#import "../../executorch-build/install/include/executorch/extension/module/module.h"
#endif

#if __has_include(<ExecuTorch/extension/tensor/tensor_ptr.h>)
#import <ExecuTorch/extension/tensor/tensor_ptr.h>
#else
#import "../../executorch-build/install/include/executorch/extension/tensor/tensor_ptr.h"
#endif

#if __has_include(<ExecuTorch/runtime/core/evalue.h>)
#import <ExecuTorch/runtime/core/evalue.h>
#else
#import "../../executorch-build/install/include/executorch/runtime/core/evalue.h"
#endif

#if __has_include(<ExecuTorch/runtime/core/exec_aten/util/scalar_type_util.h>)
#import <ExecuTorch/runtime/core/exec_aten/util/scalar_type_util.h>
#else
#import "../../executorch-build/install/include/executorch/runtime/core/exec_aten/util/scalar_type_util.h"
#endif

#if __has_include(<ExecuTorch/runtime/core/error.h>)
#import <ExecuTorch/runtime/core/error.h>
#else
#import "../../executorch-build/install/include/executorch/runtime/core/error.h"
#endif

#if __has_include(<ExecuTorch/runtime/kernel/operator_registry.h>)
#import <ExecuTorch/runtime/kernel/operator_registry.h>
#else
#import "../../executorch-build/install/include/executorch/runtime/kernel/operator_registry.h"
#endif

#if __has_include(<ExecuTorch/runtime/platform/platform.h>)
#import <ExecuTorch/runtime/platform/platform.h>
#else
#import "../../executorch-build/install/include/executorch/runtime/platform/platform.h"
#endif

// Force-enable schema inspector if the header is available in our repo include path
#ifndef RN_FORCE_SCHEMA_INSPECTOR
#define RN_FORCE_SCHEMA_INSPECTOR 1
#endif
#if RN_FORCE_SCHEMA_INSPECTOR
  #if __has_include(<executorch/schema/program_generated.h>)
    #import <executorch/schema/program_generated.h>
    #define RN_HAVE_EXECUTORCH_SCHEMA 1
  #elif __has_include("../../executorch-build/install/include/executorch/schema/program_generated.h")
    #import "../../executorch-build/install/include/executorch/schema/program_generated.h"
    #define RN_HAVE_EXECUTORCH_SCHEMA 1
  #else
    #define RN_HAVE_EXECUTORCH_SCHEMA 0
  #endif
#else
  #if __has_include(<executorch/schema/program_generated.h>)
    #import <executorch/schema/program_generated.h>
    #define RN_HAVE_EXECUTORCH_SCHEMA 1
  #elif __has_include("../../executorch-build/install/include/executorch/schema/program_generated.h")
    #import "../../executorch-build/install/include/executorch/schema/program_generated.h"
    #define RN_HAVE_EXECUTORCH_SCHEMA 1
  #else
    #define RN_HAVE_EXECUTORCH_SCHEMA 0
  #endif
#endif

// Expose which PTE inspector path was compiled in (schema vs fallback)
#if RN_HAVE_EXECUTORCH_SCHEMA
static const char *RN_ET_PTE_INSPECTOR_MODE = "schema";
#else
static const char *RN_ET_PTE_INSPECTOR_MODE = "fallback";
#endif

#ifdef __cplusplus
#include <atomic>
#include <algorithm>
#include <optional>
#include <string>
#include <vector>
#include <memory>
#include <set>
#include <cstring>
#endif



// Forward declare backend query APIs to avoid depending on framework header paths
namespace executorch { namespace runtime {
  size_t get_num_registered_backends();
  ::executorch::runtime::Result<const char*> get_backend_name(size_t index);
}} // namespace executorch::runtime

// Bridge ExecuTorch logs into React Native logs so we can see missing-operator details
static void RN_ET_EmitLog(
    et_timestamp_t /*ts*/,
    et_pal_log_level_t level,
    const char* filename,
    const char* /*function*/,
    size_t line,
    const char* message,
    size_t length) {
  NSString *msg = [[NSString alloc] initWithBytes:message length:length encoding:NSUTF8StringEncoding];
  if (!msg) { msg = [NSString stringWithUTF8String:message]; }
  switch (level) {
    case kError:
    case kFatal:
      RCTLogError(@"[ExecuTorch] %s:%zu] %@", filename, line, msg);
      break;
    default:
      RCTLogInfo(@"[ExecuTorch] %s:%zu] %@", filename, line, msg);
      break;
  }
}

static void RN_ET_InstallLogSinkOnce() {
  static dispatch_once_t onceToken;
  dispatch_once(&onceToken, ^{
    auto impl = executorch::runtime::PalImpl::create(&RN_ET_EmitLog, __FILE__);
    (void)executorch::runtime::register_pal(impl);
    // Ensure PAL is initialized so timestamps/logging work
    executorch::runtime::pal_init();
  });
}

#if RN_HAVE_EXECUTORCH_SCHEMA
// Inspect a PTE's operator table and log operators that have no registered kernel
static void RN_ET_LogMissingOpsInPte(NSString *path) {
  @try {
    NSData *data = [NSData dataWithContentsOfFile:path];
    if (!data) {
      RCTLogInfo(@"[RNExecuTorchRunner] Unable to read PTE at %@ for op inspection", path);
      return;
    }
    const uint8_t *buf = (const uint8_t *)[data bytes];
    const auto *program = executorch_flatbuffer::GetProgram(buf);
    if (!program || !program->execution_plan() || program->execution_plan()->size() == 0) {
      RCTLogInfo(@"[RNExecuTorchRunner] PTE lacks execution plans (cannot inspect ops)");
      return;
    }
    const executorch_flatbuffer::ExecutionPlan *plan = nullptr;
    auto plans = program->execution_plan();
    // Prefer method named "forward", otherwise take first
    for (flatbuffers::uoffset_t i = 0; i < plans->size(); ++i) {
      auto p = plans->Get(i);
      if (p && p->name() && std::string(p->name()->c_str()) == "forward") { plan = p; break; }
    }
    if (!plan) { plan = plans->Get(0); }
    if (!plan || !plan->operators()) {
      RCTLogInfo(@"[RNExecuTorchRunner] No operators to inspect in selected plan");
      return;
    }
    auto ops = plan->operators();
    int total = (int)ops->size();
    int missing = 0;
    int logged = 0;
    for (flatbuffers::uoffset_t idx = 0; idx < ops->size(); ++idx) {
      auto op = ops->Get(idx);
      const char *name = (op && op->name()) ? op->name()->c_str() : "";
      const char *ov = (op && op->overload()) ? op->overload()->c_str() : "";
      std::string full = name ? std::string(name) : std::string("");
      if (ov && ov[0] != '\0') { full += "."; full += ov; }
      bool has = ::executorch::ET_RUNTIME_NAMESPACE::registry_has_op_function(full.c_str());
      if (!has) {
        missing++;
        if (logged < 30) {
          RCTLogError(@"[RNExecuTorchRunner] Missing kernel: [%u] %s", (unsigned)idx, full.c_str());
          // Common suffix mismatch hint ('.default' vs '.out')
          if (ov && std::string(ov) == "default") {
            std::string outName = std::string(name) + ".out";
            if (::executorch::ET_RUNTIME_NAMESPACE::registry_has_op_function(outName.c_str())) {
              RCTLogError(@"[RNExecuTorchRunner] Hint: '.default' missing but '.out' exists for %s — build/export variant mismatch.", name);
            }
          }
          logged++;
        }
      }
    }
    if (missing == 0) {
      RCTLogInfo(@"[RNExecuTorchRunner] All %d operators have a registered kernel name", total);
    } else {
      RCTLogError(@"[RNExecuTorchRunner] Missing kernels for %d/%d operators (logged first 30)", missing, total);
    }
  } @catch (...) {
    // best-effort only
  }
}
#else
static void RN_ET_LogMissingOpsInPte(NSString *path) {
  // Fallback best-effort scanner when program_generated.h isn't available.
  // We scan the FlatBuffer for operator name strings like "aten_foo"/"llama_bar"
  // and then check both ".default" and ".out" variants against the runtime registry.
  @try {
    NSData *data = [NSData dataWithContentsOfFile:path];
    if (!data) {
      RCTLogInfo(@"[RNExecuTorchRunner] Unable to read PTE at %@ for fallback op inspection", path);
      return;
    }
    const uint8_t *buf = (const uint8_t *)[data bytes];
    size_t sz = (size_t)[data length];

    struct Prefix { const char* ns; const char* pat; size_t n; };
    static const Prefix kPrefixes[] = {
      {"aten",    "aten_",     5},
      {"aten",    "aten::",    6},
      {"llama",   "llama_",    6},
      {"llama",   "llama::",   7},
      {"torchao", "torchao_",  8},
      {"torchao", "torchao::", 9},
    };

    auto is_name_char = [](unsigned char c) -> bool {
      return (c == '_') || (c == ':') || (c == '.') || (c >= '0' && c <= '9') ||
             (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
    };

    std::set<std::string> op_candidates; // Full runtime-style names like "aten::add.out" or bases to expand
    for (size_t i = 0; i + 8 < sz; ++i) {
      for (const auto &pref : kPrefixes) {
        if (i + pref.n < sz && std::memcmp(buf + i, pref.pat, pref.n) == 0) {
          size_t j = i + pref.n;
          size_t start = j;
          // Cap name length to avoid scanning across the whole file if not null-terminated nearby
          const size_t kMax = 128;
          while (j < sz && (j - start) < kMax && is_name_char(buf[j])) { ++j; }
          if (j > start) {
            std::string ident(reinterpret_cast<const char*>(buf + start), j - start);
            if (ident.size() >= 3) {
              std::string base = std::string(pref.ns) + "::" + ident; // e.g., "aten::add" or "aten::add.out"
              // If suffix is already present (contains '.'), test as-is; otherwise test both default/out variants.
              if (ident.find('.') != std::string::npos) {
                op_candidates.insert(base);
              } else {
                op_candidates.insert(base + ".default");
                op_candidates.insert(base + ".out");
              }
            }
          }
          i = j; // advance past the token
        }
      }
    }

    if (op_candidates.empty()) {
      RCTLogInfo(@"[RNExecuTorchRunner] Fallback inspection found no operator-like strings in PTE");
      return;
    }

    int total = (int)op_candidates.size();
    int missing = 0;
    int logged = 0;
    for (const auto &cand : op_candidates) {
      bool has = ::executorch::ET_RUNTIME_NAMESPACE::registry_has_op_function(cand.c_str());
      if (!has) {
        missing++;
        if (logged < 40) {
          RCTLogError(@"[RNExecuTorchRunner] Missing kernel (fallback scan): %s", cand.c_str());
          logged++;
        }
      }
    }
    if (missing == 0) {
      RCTLogInfo(@"[RNExecuTorchRunner] Fallback scan: all %d operator bases appear to have a registered variant", total);
    } else {
      RCTLogError(@"[RNExecuTorchRunner] Fallback scan: %d/%d operator bases appear missing (tried .default/.out)", missing, total);
    }
  } @catch (...) {
    // best-effort only
  }
}
#endif




// Forward declare ExecuTorch prim op function to force-link its TU
namespace torch { namespace executor { namespace function {
  void et_view(::executorch::ET_RUNTIME_NAMESPACE::KernelRuntimeContext&, ::executorch::ET_RUNTIME_NAMESPACE::Span<::executorch::ET_RUNTIME_NAMESPACE::EValue*>);
}}}

using namespace executorch::extension::llm;

@interface RNExecuTorchRunner () {
  std::unique_ptr<TextLLMRunner> _runner;
  std::atomic<bool> _shouldStop;
}
@property (nonatomic) dispatch_queue_t generationQueue;
@property (nonatomic) BOOL modelLoaded;
@property (nonatomic, strong) NSString *storedModelPath;
@property (nonatomic, strong) NSString *storedTokenizerPath;
@end

@implementation RNExecuTorchRunner

RCT_EXPORT_MODULE(ExecuTorchRunner);

+ (BOOL)requiresMainQueueSetup {
  return NO;
}

- (instancetype)init {
  if (self = [super init]) {
    _shouldStop.store(false);
    _modelLoaded = NO;
    _generationQueue = dispatch_queue_create("com.alientavern.executorch.generation", DISPATCH_QUEUE_SERIAL);
  }
  return self;
}

- (NSArray<NSString *> *)supportedEvents {
  return @[@"onToken", @"onComplete", @"onError"];
}

#pragma mark - Public Methods

RCT_EXPORT_METHOD(loadModel:(NSDictionary *)config
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)
{
  dispatch_async(self.generationQueue, ^{
    @try {
      NSString *modelPath = config[@"modelPath"];
      NSString *tokenizerPath = config[@"tokenizerPath"];
      NSString *tokenizerType = config[@"tokenizerType"] ?: @"huggingface";

      if (!modelPath || !tokenizerPath) {
        reject(@"INVALID_CONFIG", @"modelPath and tokenizerPath are required", nil);
        return;
      }


          // Install ExecuTorch log sink once so ET_LOG messages appear in RN logs
          ::executorch::runtime::pal_init();
          RN_ET_InstallLogSinkOnce();

      RCTLogInfo(@"[RNExecuTorchRunner] Loading model: %@", modelPath);
      RCTLogInfo(@"[RNExecuTorchRunner] Loading tokenizer: %@ (type: %@)", tokenizerPath, tokenizerType);

      // Validate paths exist
      NSFileManager *fileManager = [NSFileManager defaultManager];
      if (![fileManager fileExistsAtPath:modelPath]) {
        reject(@"MODEL_NOT_FOUND", [NSString stringWithFormat:@"Model file not found: %@", modelPath], nil);
        return;
      }
      // Persist paths for fallback decode loop
      self.storedModelPath = modelPath;
      self.storedTokenizerPath = tokenizerPath;


      if (![fileManager fileExistsAtPath:tokenizerPath]) {
        reject(@"TOKENIZER_NOT_FOUND", [NSString stringWithFormat:@"Tokenizer file not found: %@", tokenizerPath], nil);
        return;
      }

      // Load tokenizer
      auto tokenizer = load_tokenizer(
        [tokenizerPath UTF8String],
        nullptr,
        std::nullopt,
        0,
        0
      );

      if (!tokenizer) {
        reject(@"TOKENIZER_ERROR", @"Failed to load tokenizer", nil);
        return;
      }

      RCTLogInfo(@"[RNExecuTorchRunner] Tokenizer loaded successfully");

      // Create runner
      _runner = create_text_llm_runner(
        [modelPath UTF8String],
        std::move(tokenizer)
      );

      if (!_runner) {
        reject(@"RUNNER_ERROR", @"Failed to create runner", nil);
        return;
      }

      RCTLogInfo(@"[RNExecuTorchRunner] Runner created, loading model...");
      // Manually register missing prim fallback kernel(s) to avoid needing -force_load
      auto ensureOp = [&](const char* name) {
        if (!::executorch::ET_RUNTIME_NAMESPACE::registry_has_op_function(name)) {
          auto err = ::executorch::ET_RUNTIME_NAMESPACE::register_kernel(
            ::executorch::ET_RUNTIME_NAMESPACE::Kernel(
              name,
              &::torch::executor::function::et_view // function is the same for both keys
            )
          );
          (void)err;
          RCTLogInfo(@"[RNExecuTorchRunner] Registered fallback kernel for %s", name);
        }
        bool has = ::executorch::ET_RUNTIME_NAMESPACE::registry_has_op_function(name);
        RCTLogInfo(@"[RNExecuTorchRunner] registry_has_op '%s' = %s", name, has ? "true" : "false");
      };
      ensureOp("executorch_prim::et_view");
      // Log registered backends (expect to see xnnpack)
      size_t nBk = ::executorch::ET_RUNTIME_NAMESPACE::get_num_registered_backends();
      RCTLogInfo(@"[RNExecuTorchRunner] Registered backends: %zu", nBk);
      for (size_t i = 0; i < nBk; ++i) {
        auto nm = ::executorch::ET_RUNTIME_NAMESPACE::get_backend_name(i);
        if (nm.ok()) {
          RCTLogInfo(@"[RNExecuTorchRunner] backend[%zu] = %s", i, nm.get());
        }
      }

      // Debug: list a sample of registered kernels to validate registry linkage
      {
        auto span = ::executorch::ET_RUNTIME_NAMESPACE::get_registered_kernels();
        size_t total = span.size();
        RCTLogInfo(@"[RNExecuTorchRunner] total registered kernels: %zu", total);
        size_t sampleN = std::min<size_t>(total, 40);
        for (size_t i = 0; i < sampleN; ++i) {
          const auto &k = span[i];
          if (k.name_) {
            RCTLogInfo(@"[RNExecuTorchRunner] kernel[%zu]: %s", i, k.name_);
          }
        }
      }


      ensureOp("executorch_prim::et_view.default");

      // Debug: check if common portable ops are registered
      auto __logKernel = ^(const char* name) {
        bool has = ::executorch::ET_RUNTIME_NAMESPACE::registry_has_op_function(name);
        RCTLogInfo(@"[RNExecuTorchRunner] registry_has_op '%s' = %s", name, has ? "true" : "false");
      };
      RCTLogInfo(@"[RNExecuTorchRunner] Checking operator registry for common ops...");
      __logKernel("aten::_softmax.out");
      __logKernel("aten::_to_copy.out");
      __logKernel("aten::add.out");
      __logKernel("aten::add.Scalar_out");
      __logKernel("aten::any.out");
      __logKernel("aten::bmm.out");
      __logKernel("aten::cat.out");
      __logKernel("aten::clone.out");
      __logKernel("aten::copy_");
      __logKernel("aten::cos.out");
      __logKernel("aten::embedding.out");
      __logKernel("aten::eq.Scalar_out");
      __logKernel("aten::expand_copy.out");
      __logKernel("aten::full_like.out");
      __logKernel("aten::index_put.out");
      __logKernel("aten::le.Scalar_out");
      __logKernel("aten::logical_not.out");
      __logKernel("aten::mean.out");
      __logKernel("aten::mm.out");
      __logKernel("aten::mul.out");
      __logKernel("aten::neg.out");
      __logKernel("aten::permute_copy.out");
      __logKernel("aten::pow.Scalar_out");
      __logKernel("aten::rsqrt.out");
      __logKernel("aten::sigmoid.out");
      __logKernel("aten::sin.out");
      __logKernel("aten::slice_copy.Tensor_out");
      __logKernel("aten::unsqueeze_copy.out");
      __logKernel("aten::view_copy.out");
      __logKernel("aten::where.self_out");
      __logKernel("executorch_prim::et_view");
      __logKernel("executorch_prim::et_view.default");

      // LLM custom kernels (kernels_llm)
      __logKernel("llama::sdpa_with_kv_cache.out");
      __logKernel("llama::custom_sdpa.out");
      __logKernel("llama::custom_quantized_sdpa.out");
      __logKernel("llama::fallback.out");


      // Load model
      auto error = _runner->load();
      if (error != ::executorch::runtime::Error::Ok) {
        // Log numeric error code and enum string to help diagnose (e.g., OperatorMissing, NotFound, InvalidArgument)
        RCTLogError(@"[RNExecuTorchRunner] load() failed with error code: %d (%s)", (int)error, ::executorch::runtime::to_string(error));

        // If missing operator, try automatic fallback to Simple XNNPACK variant, then Portable
        if (error == ::executorch::runtime::Error::OperatorMissing) {
          // Inspect PTE to identify exactly which operators are missing before attempting fallbacks
          RCTLogInfo(@"[RNExecuTorchRunner] PTE inspector mode: %s", RN_ET_PTE_INSPECTOR_MODE);
          RN_ET_LogMissingOpsInPte(modelPath);

          NSFileManager *fm = [NSFileManager defaultManager];
          NSString *dir = [modelPath stringByDeletingLastPathComponent];
          NSString *simplePath = [dir stringByAppendingPathComponent:@"smollm2-135m-simple.pte"];
          NSString *portablePath = [dir stringByAppendingPathComponent:@"smollm2-135m-portable.pte"];

          auto tryLoadAt = [&](NSString *candidatePath, const char *label) -> bool {
            if (![fm fileExistsAtPath:candidatePath]) return false;
            RCTLogInfo(@"[RNExecuTorchRunner] OperatorMissing: attempting %@: %@", [NSString stringWithUTF8String:label], candidatePath);
            // Recreate runner with same tokenizer (runner takes ownership)
            _runner.reset();
            auto tok2 = load_tokenizer([tokenizerPath UTF8String], nullptr, std::nullopt, 0, 0);
            if (!tok2) { RCTLogError(@"[RNExecuTorchRunner] Fallback: tokenizer reload failed"); return false; }
            _runner = create_text_llm_runner([candidatePath UTF8String], std::move(tok2));
            if (!_runner) { RCTLogError(@"[RNExecuTorchRunner] Fallback: runner create failed"); return false; }
            auto e2 = _runner->load();
            if (e2 == ::executorch::runtime::Error::Ok) {
              self.storedModelPath = candidatePath;
              self.modelLoaded = YES;
              RCTLogInfo(@"[RNExecuTorchRunner] Fallback loaded successfully: %@", candidatePath);
              return true;
            } else {
              RCTLogError(@"[RNExecuTorchRunner] Fallback load failed (%s)", ::executorch::runtime::to_string(e2));
              return false;
            }
          };

          if (tryLoadAt(simplePath, "Simple XNNPACK variant")) {
            // proceed as success
          } else if (tryLoadAt(portablePath, "Portable variant")) {
            // proceed as success (slower)
          } else {
            reject(@"LOAD_ERROR", @"Failed to load model into runner (and fallbacks)", nil);
            return;
          }
        } else {
          reject(@"LOAD_ERROR", @"Failed to load model into runner", nil);
          return;
        }
      }

      self.modelLoaded = YES;

	      // Introspect model methods and forward signature for diagnostics
	      try {
	        // Use the actual loaded path (may be changed by fallback)
        NSString *activePath = self.storedModelPath ?: modelPath;
        ::executorch::extension::ET_MODULE_NAMESPACE::Module mod([activePath UTF8String]);
	        auto modErr = mod.load();
	        if (modErr == ::executorch::runtime::Error::Ok) {
	          auto namesRes = mod.method_names();
	          if (namesRes.ok()) {
	            std::string allNames;
	            for (const auto &n : namesRes.get()) {
	              allNames += n + ", ";
	            }
	            RCTLogInfo(@"[RNExecuTorchRunner] Methods: %s", allNames.c_str());
	          }
	          auto mm = mod.method_meta("forward");
	          if (mm.ok()) {
	            size_t nIn = mm->num_inputs();
	            size_t nOut = mm->num_outputs();
	            RCTLogInfo(@"[RNExecuTorchRunner] forward inputs=%zu outputs=%zu", nIn, nOut);
            // Backends used by this method
            size_t nb = mm->num_backends();
            RCTLogInfo(@"[RNExecuTorchRunner] forward num_backends=%zu", nb);
            for (size_t i = 0; i < nb; ++i) {
              auto b = mm->get_backend_name(i);
              if (b.ok()) {
                RCTLogInfo(@"[RNExecuTorchRunner] forward backend[%zu]=%s", i, b.get());
              }
            }
            bool usesXNN = mm->uses_backend("xnnpack");
            RCTLogInfo(@"[RNExecuTorchRunner] forward uses_backend('xnnpack')=%s", usesXNN ? "true" : "false");
	            for (size_t i = 0; i < nIn; ++i) {
	              auto tagRes = mm->input_tag(i);
	              if (tagRes.ok()) {
	                RCTLogInfo(@"[RNExecuTorchRunner] input[%zu] tag=%d", i, (int)tagRes.get());
	              }
	            }

		          // Additional tensor meta (best-effort)
		          for (size_t i = 0; i < nIn; ++i) {
		            auto tmeta = mm->input_tensor_meta(i);
		            if (tmeta.ok()) {
		              auto sz = tmeta->sizes();
		              std::string dims;
		              for (size_t k = 0; k < sz.size(); ++k) {
		                dims += std::to_string(sz[k]);
		                if (k + 1 < sz.size()) dims += ",";
		              }
		              int st = (int)tmeta->scalar_type();
		              RCTLogInfo(@"[RNExecuTorchRunner] input[%zu] tensor dtype=%d sizes=[%s]",
		                         i, st, dims.c_str());
		            }
		          }
		          if (nOut > 0) {
		            auto oMeta = mm->output_tensor_meta(0);
		            if (oMeta.ok()) {
		              auto sz = oMeta->sizes();
		              std::string dims;
		              for (size_t k = 0; k < sz.size(); ++k) {
		                dims += std::to_string(sz[k]);
		                if (k + 1 < sz.size()) dims += ",";
		              }
		              int st = (int)oMeta->scalar_type();
		              RCTLogInfo(@"[RNExecuTorchRunner] output[0] tensor dtype=%d sizes=[%s]",
		                         st, dims.c_str());
		            }
		          }

	          }
	        }
	      } catch (...) {
	        // best-effort diagnostics only
	      }

      RCTLogInfo(@"[RNExecuTorchRunner] Model loaded successfully! Active PTE: %@", self.storedModelPath ?: modelPath);
      resolve(@YES);
    } @catch (NSException *exception) {
      RCTLogError(@"[RNExecuTorchRunner] Exception loading model: %@", exception.reason);
      reject(@"LOAD_ERROR", exception.reason, nil);
    }
  });
}

RCT_EXPORT_METHOD(isLoaded:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)
{
  BOOL loaded = _runner && _runner->is_loaded();
  resolve(@(loaded));
}

RCT_EXPORT_METHOD(generate:(NSString *)prompt
                  config:(NSDictionary *)config
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)
{
  if (!self.modelLoaded) {
    reject(@"NOT_LOADED", @"Model not loaded. Call loadModel first.", nil);
    return;
  }

  dispatch_async(self.generationQueue, ^{

      int maxNewTokens = [config[@"maxNewTokens"] intValue] ?: 100;
      float temperature = [config[@"temperature"] floatValue] ?: 0.8f;
      BOOL echo = [config[@"echo"] boolValue];

      RCTLogInfo(@"[RNExecuTorchRunner] Generating with prompt: %@", prompt);
      RCTLogInfo(@"[RNExecuTorchRunner] Config: maxTokens=%d, temp=%.2f, echo=%d", maxNewTokens, temperature, echo);

      self->_shouldStop.store(false);

      // Reset runner state before a new generation to clear any prefilled KV cache
      if (_runner) {
        _runner->reset();
      }


      // Configure generation
      GenerationConfig genConfig;
      genConfig.max_new_tokens = maxNewTokens;
      genConfig.temperature = temperature;
      genConfig.echo = echo;

      NSMutableString *fullResponse = [NSMutableString string];
      int tokenCount = 0;
      NSTimeInterval startTime = [[NSDate date] timeIntervalSince1970];
      NSTimeInterval firstTokenTime = 0;

      // Capture self directly in C++ lambda (no weak/strong needed for synchronous execution)
      RNExecuTorchRunner *selfPtr = self;

      // Generate with callbacks
      auto error = _runner->generate(
        [prompt UTF8String],
        genConfig,
        // Token callback
        [selfPtr, &tokenCount, &firstTokenTime, &fullResponse](const std::string& token) {
          if (selfPtr->_shouldStop.load()) {
            return;
          }

          if (tokenCount == 0) {
            firstTokenTime = [[NSDate date] timeIntervalSince1970];
          }

          tokenCount++;
          NSString *tokenStr = [NSString stringWithUTF8String:token.c_str()];
          [fullResponse appendString:tokenStr];

          // Emit token event to JavaScript
          dispatch_async(dispatch_get_main_queue(), ^{
            [selfPtr sendEventWithName:@"onToken" body:tokenStr];
          });
        },
        // Stats callback
        [selfPtr, startTime, firstTokenTime](const Stats& stats) {
          NSTimeInterval endTime = [[NSDate date] timeIntervalSince1970];
          NSTimeInterval totalTime = (endTime - startTime) * 1000; // ms
          NSTimeInterval timeToFirstToken = (firstTokenTime - startTime) * 1000; // ms

          // Calculate tokens per second
          double inferenceTimeSeconds = (double)(stats.inference_end_ms - stats.inference_start_ms) / stats.SCALING_FACTOR_UNITS_PER_SECOND;
          double tokensPerSecond = inferenceTimeSeconds > 0 ? stats.num_generated_tokens / inferenceTimeSeconds : 0;

          NSDictionary *statsDict = @{
            @"promptTokens": @(stats.num_prompt_tokens),
            @"generatedTokens": @(stats.num_generated_tokens),
            @"totalTime": @(totalTime),
            @"tokensPerSecond": @(tokensPerSecond),
            @"timeToFirstToken": @(timeToFirstToken)
          };

          // Emit complete event to JavaScript
          dispatch_async(dispatch_get_main_queue(), ^{
            [selfPtr sendEventWithName:@"onComplete" body:statsDict];
          });
        }
      );

      if (error != ::executorch::runtime::Error::Ok) {
        if (error == ::executorch::runtime::Error::NotSupported) {
          // Quick unblock: fallback to manual forward loop using Module + Tokenizer
          RCTLogInfo(@"[RNExecuTorchRunner] Falling back to manual decode loop (NotSupported from TextLLMRunner)");

          // Load a fresh tokenizer (runner owns its tokenizer)
          auto fbTokenizer = load_tokenizer(
            [self.storedTokenizerPath UTF8String],
            nullptr,
            std::nullopt,
            0,
            0
          );
          if (!fbTokenizer) {
            RCTLogError(@"[RNExecuTorchRunner] Fallback: failed to load tokenizer");
            dispatch_async(dispatch_get_main_queue(), ^{
              [selfPtr sendEventWithName:@"onError" body:@{ @"message": @"Fallback decode: tokenizer load failed" }];
            });
            reject(@"GENERATION_ERROR", @"Fallback decode failed (tokenizer)", nil);
            return;
          }

          // Format prompt with Qwen2.5 ChatML template
          // Format: <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
          NSString *formattedPrompt = [NSString stringWithFormat:@"<|im_start|>user\n%@<|im_end|>\n<|im_start|>assistant\n", prompt];
          RCTLogInfo(@"[RNExecuTorchRunner] Original prompt: %@", prompt);
          RCTLogInfo(@"[RNExecuTorchRunner] Formatted prompt: %@", formattedPrompt);

          // Encode prompt
          auto enc = fbTokenizer->encode([formattedPrompt UTF8String], /*bos*/0, /*eos*/0);
          if (!enc.ok()) {
            RCTLogError(@"[RNExecuTorchRunner] Fallback: encode() failed");
            dispatch_async(dispatch_get_main_queue(), ^{
              [selfPtr sendEventWithName:@"onError" body:@{ @"message": @"Fallback decode: encode failed" }];
            });
            reject(@"GENERATION_ERROR", @"Fallback decode failed (encode)", nil);
            return;
          }
          std::vector<uint64_t> ids_u = enc.get();
          std::vector<int64_t> ids;
          ids.reserve(ids_u.size());
          for (auto v : ids_u) ids.push_back(static_cast<int64_t>(v));
          const int promptTokens = (int)ids.size();

          // Log token IDs to verify encoding
          NSMutableString *tokenStr = [NSMutableString stringWithString:@"["];
          for (size_t i = 0; i < std::min(ids.size(), (size_t)20); ++i) {
            [tokenStr appendFormat:@"%lld", ids[i]];
            if (i + 1 < std::min(ids.size(), (size_t)20)) [tokenStr appendString:@", "];
          }
          if (ids.size() > 20) [tokenStr appendString:@", ..."];
          [tokenStr appendString:@"]"];
          RCTLogInfo(@"[RNExecuTorchRunner] Encoded %zu tokens: %@", ids.size(), tokenStr);

          // Optional echo of prompt tokens
          if (echo && ids.size() >= 2) {
            for (size_t i = 1; i < ids.size(); ++i) {
              auto d = fbTokenizer->decode((uint64_t)ids[i-1], (uint64_t)ids[i]);
              if (d.ok()) {
                NSString *tokenStr = [NSString stringWithUTF8String:d.get().c_str()];
                [fullResponse appendString:tokenStr];
                if (tokenCount == 0) firstTokenTime = [[NSDate date] timeIntervalSince1970];
                tokenCount++;
                dispatch_async(dispatch_get_main_queue(), ^{
                  [selfPtr sendEventWithName:@"onToken" body:tokenStr];
                });
              }
            }
          }

          // Load module
          ::executorch::extension::ET_MODULE_NAMESPACE::Module mod([self.storedModelPath UTF8String]);
          auto modErr = mod.load();
          if (modErr != ::executorch::runtime::Error::Ok) {
            RCTLogError(@"[RNExecuTorchRunner] Fallback: Module load failed: %d (%s)", (int)modErr, ::executorch::runtime::to_string(modErr));
            dispatch_async(dispatch_get_main_queue(), ^{
              [selfPtr sendEventWithName:@"onError" body:@{ @"message": @"Fallback decode: module load failed" }];
            });
            reject(@"GENERATION_ERROR", @"Fallback decode failed (module load)", nil);
            return;
          }

          // Best-effort: enable dynamic shapes if the method exists
          try {
            auto dm = mod.method_meta("enable_dynamic_shape");
            if (dm.ok()) {
              std::vector<::executorch::runtime::EValue> emptyInputs;
              auto res = mod.execute("enable_dynamic_shape", emptyInputs);
              if (res.ok()) {
                RCTLogInfo(@"[RNExecuTorchRunner] enable_dynamic_shape invoked");
              }
            }
          } catch (...) {
            // ignore
          }


          // Try enabling KV cache and SDPA-with-KV if available (some exports gate kernels on these flags)
          try {
            auto mkv = mod.method_meta("use_kv_cache");
            if (mkv.ok()) {
              std::vector<::executorch::runtime::EValue> emptyInputs;
              auto r = mod.execute("use_kv_cache", emptyInputs);
              if (r.ok()) {
                RCTLogInfo(@"[RNExecuTorchRunner] use_kv_cache invoked");
              }
            }
          } catch (...) {
            // ignore
          }
          try {
            auto msdpa = mod.method_meta("use_sdpa_with_kv_cache");
            if (msdpa.ok()) {
              std::vector<::executorch::runtime::EValue> emptyInputs;
              auto r = mod.execute("use_sdpa_with_kv_cache", emptyInputs);
              if (r.ok()) {
                RCTLogInfo(@"[RNExecuTorchRunner] use_sdpa_with_kv_cache invoked");
              }
            }
          } catch (...) {
            // ignore
          }

            // Introspect forward() signature (success path)
            {
              auto mm2 = mod.method_meta("forward");
              if (mm2.ok()) {
                size_t nIn = mm2->num_inputs();
                size_t nOut = mm2->num_outputs();
                RCTLogInfo(@"[RNExecuTorchRunner] (fallback) forward inputs=%zu outputs=%zu", nIn, nOut);
                size_t nb2 = mm2->num_backends();
                RCTLogInfo(@"[RNExecuTorchRunner] (fallback) forward num_backends=%zu", nb2);
                for (size_t j = 0; j < nb2; ++j) {
                  auto b2 = mm2->get_backend_name(j);
                  if (b2.ok()) {
                    RCTLogInfo(@"[RNExecuTorchRunner] (fallback) forward backend[%zu]=%s", j, b2.get());
                  }
                }
                bool usesXNN2 = mm2->uses_backend("xnnpack");
                RCTLogInfo(@"[RNExecuTorchRunner] (fallback) forward uses_backend('xnnpack')=%s", usesXNN2 ? "true" : "false");
                for (size_t i = 0; i < nIn; ++i) {
                  auto tagRes = mm2->input_tag(i);
                  if (tagRes.ok()) {
                    RCTLogInfo(@"[RNExecuTorchRunner] (fallback) input[%zu] tag=%d", i, (int)tagRes.get());
                  }
                  auto tmeta = mm2->input_tensor_meta(i);
                  if (tmeta.ok()) {
                    auto sz = tmeta->sizes();
                    std::string dims;
                    for (size_t k = 0; k < sz.size(); ++k) {
                      dims += std::to_string(sz[k]);
                      if (k + 1 < sz.size()) dims += ",";
                    }
                    int st = (int)tmeta->scalar_type();
                    RCTLogInfo(@"[RNExecuTorchRunner] (fallback) input[%zu] tensor dtype=%d sizes=[%s]",
                               i, st, dims.c_str());
                  }
                }
                if (nOut > 0) {
                  auto oMeta = mm2->output_tensor_meta(0);
                  if (oMeta.ok()) {
                    auto sz = oMeta->sizes();
                    std::string dims;
                    for (size_t k = 0; k < sz.size(); ++k) {
                      dims += std::to_string(sz[k]);
                      if (k + 1 < sz.size()) dims += ",";
                    }
                    int st = (int)oMeta->scalar_type();
                    RCTLogInfo(@"[RNExecuTorchRunner] (fallback) output[0] tensor dtype=%d sizes=[%s]",
                               st, dims.c_str());
                  }
                }
              }
            }

          // Generation loop
          int generated = 0;
          int64_t current_pos = 0; // Track actual sequence position for single-token mode

          // Determine a working input format once, then reuse
          bool formatLocked = false;
          bool use2DTokens = false; // 1D [T] by default, fallback [1, T]
          bool orderSwappedChosen = false; // tokens,start_pos by default (only for 2-input models)
          bool singleTokenOnly = false; // enforce [1,1] token input if meta requires (2-input models)
          bool needsPrefill = false; // if true, run prefill with all prompt tokens first
          bool oneInputForward = false; // some exports have forward(tokens) only
          bool oneInputPrefilled = false; // after first full-prompt call, stream 1 token at a time
          int64_t tokenSeqDimHint = -1;   // if meta suggests fixed [1,N], use N as hint for 1-input models
          executorch::aten::ScalarType tokenDTypeChosen = executorch::aten::ScalarType::Long; // try Long then Int
          int startPosShape = 0; // 0: scalar [], 1: [1], 2: [1,1], 3: [T], 4: [1,T]
          int posModeChosen = 0; // 0=zeros/0, 1=len(T), 2=arange(0..T-1), 3=ones
          executorch::aten::ScalarType startDTypeChosen = executorch::aten::ScalarType::Long;
          // Inspect meta to detect 1-input vs 2-input signatures
          {
            auto mm3 = mod.method_meta("forward");
            if (mm3.ok()) {
              size_t nIn_meta = mm3->num_inputs();
              oneInputForward = (nIn_meta == 1);
              if (oneInputForward) {
                // Prefer [1,T] if meta suggests 2D input; capture fixed T hint if provided
                auto t0 = mm3->input_tensor_meta(0);
                if (t0.ok()) {
                  auto s0 = t0->sizes();
                  if (s0.size() == 2 && s0[0] == 1) {
                    use2DTokens = true;
                    if (s0[1] > 0) tokenSeqDimHint = (int64_t)s0[1];
                  }
                  tokenDTypeChosen = t0->scalar_type();
                }
                // For 1-input models, start by sending the full prompt (no prefill),
                // and if needed we will slice to the hinted length
                singleTokenOnly = false;
                needsPrefill = false;
                RCTLogInfo(@"[RNExecuTorchRunner] Detected 1-input forward(tokens). Will pass only tokens (no start_pos). No prefill; using full-prompt on probe.");
              } else {
                // 2-input models: try to recognize single-token mode
                auto t0 = mm3->input_tensor_meta(0);
                auto t1 = mm3->input_tensor_meta(1);
                if (t0.ok() && t1.ok()) {
                  auto s0 = t0->sizes();
                  auto s1 = t1->sizes();
                  RCTLogInfo(@"[RNExecuTorchRunner] Checking meta: s0.size=%zu s0[0]=%lld s0[1]=%lld s1.size=%zu s1[0]=%lld",
                             s0.size(), s0.size() >= 1 ? (long long)s0[0] : -1, s0.size() >= 2 ? (long long)s0[1] : -1,
                             s1.size(), s1.size() >= 1 ? (long long)s1[0] : -1);
                  if (s0.size() == 2 && s0[0] == 1 && s0[1] == 1 && s1.size() == 1 && s1[0] == 1) {
                    singleTokenOnly = true;
                    needsPrefill = true; // Need to prefill prompt tokens one-by-one
                    use2DTokens = true;
                    tokenDTypeChosen = t0->scalar_type();
                    startDTypeChosen = t1->scalar_type();
                    startPosShape = 1; // [1]
                    posModeChosen = 0; // start at 0
                    orderSwappedChosen = false;
                    RCTLogInfo(@"[RNExecuTorchRunner] Fallback: detected single-token mode from meta: tokens dtype=%d [1,1], start_pos dtype=%d [1]", (int)tokenDTypeChosen, (int)startDTypeChosen);
                  } else {
                    RCTLogInfo(@"[RNExecuTorchRunner] Meta check: not single-token mode, will probe");
                  }
                } else {
                  RCTLogInfo(@"[RNExecuTorchRunner] Could not get full tensor meta for 2-input forward (t0.ok=%d t1.ok=%d)", t0.ok(), t1.ok());
                }
              }
            } else {
              RCTLogInfo(@"[RNExecuTorchRunner] Failed to get method_meta for forward");
            }
          }


          while (generated < maxNewTokens && !selfPtr->_shouldStop.load()) {
            // Helper to build inputs and run forward once
            auto build_and_run = [&](executorch::aten::ScalarType tokDT,
                                     bool twoDim,
                                     executorch::aten::ScalarType posDT,
                                     int posShape,
                                     int posMode,
                                     bool swappedOrder,
                                     int64_t explicit_pos = -1) -> ::executorch::runtime::Result<std::vector<::executorch::runtime::EValue>> {
              // Build tokens tensor (respect singleTokenOnly if enforced by meta)
              int32_t Tsend = singleTokenOnly ? 1 : (int32_t)ids.size();
              // Always send the full prompt for 1-input models; ExecuTorch allows dynamic T.
              std::vector<int64_t> ids_to_send;
              ids_to_send.reserve(Tsend);
              if (singleTokenOnly) {
                // Use last token of current context
                ids_to_send.push_back(ids.empty() ? 0 : ids.back());
              } else {
                if ((size_t)Tsend < ids.size()) {
                  ids_to_send.assign(ids.end() - Tsend, ids.end()); // last Tsend tokens
                } else {
                  ids_to_send.assign(ids.begin(), ids.end());
                }
              }

              std::vector<executorch::aten::SizesType> tokenSizes = twoDim
                ? std::vector<executorch::aten::SizesType>{1, (executorch::aten::SizesType)Tsend}
                : std::vector<executorch::aten::SizesType>{(executorch::aten::SizesType)Tsend};

              ::executorch::extension::TensorPtr tokensTensor;
              if (tokDT == executorch::aten::ScalarType::Int) {
                std::vector<int32_t> ids32; ids32.reserve(ids_to_send.size());
                for (auto v : ids_to_send) ids32.push_back(static_cast<int32_t>(v));
                tokensTensor = executorch::extension::make_tensor_ptr<int32_t>(
                  tokenSizes,
                  ids32,
                  /*dim_order*/{},
                  /*strides*/{},
                  executorch::aten::ScalarType::Int
                );
              } else {
                tokensTensor = executorch::extension::make_tensor_ptr<int64_t>(
                  tokenSizes,
                  ids_to_send,
                  /*dim_order*/{},
                  /*strides*/{},
                  executorch::aten::ScalarType::Long
                );
              }

              std::vector<::executorch::runtime::EValue> inputs;
              inputs.emplace_back(tokensTensor);

              if (!oneInputForward) {
                // Build start_pos tensor only for 2-input models
                ::executorch::extension::TensorPtr startPosTensor;
                const auto T = (int32_t)Tsend;

                // Use explicit_pos if provided (for mid-generation), otherwise use posMode logic (for probe)
                int64_t pos_value = (explicit_pos >= 0) ? explicit_pos : (posMode == 1 ? (int64_t)T : 0);

                if (posShape == 0) {
                  // scalar []
                  if (posDT == executorch::aten::ScalarType::Int) {
                    startPosTensor = executorch::extension::make_tensor_ptr<int32_t>((int32_t)pos_value);
                  } else {
                    startPosTensor = executorch::extension::make_tensor_ptr<int64_t>(pos_value);
                  }
                } else if (posShape == 1) {
                  // [1]
                  std::vector<executorch::aten::SizesType> s{1};
                  if (posDT == executorch::aten::ScalarType::Int) {
                    std::vector<int32_t> v{(int32_t)pos_value};
                    startPosTensor = executorch::extension::make_tensor_ptr<int32_t>(s, v, {}, {}, executorch::aten::ScalarType::Int);
                  } else {
                    std::vector<int64_t> v{pos_value};
                    startPosTensor = executorch::extension::make_tensor_ptr<int64_t>(s, v, {}, {}, executorch::aten::ScalarType::Long);
                  }
                } else if (posShape == 2) { // [1,1]
                  std::vector<executorch::aten::SizesType> s{1,1};
                  if (posDT == executorch::aten::ScalarType::Int) {
                    std::vector<int32_t> v{(int32_t)pos_value};
                    startPosTensor = executorch::extension::make_tensor_ptr<int32_t>(s, v, {}, {}, executorch::aten::ScalarType::Int);
                  } else {
                    std::vector<int64_t> v{pos_value};
                    startPosTensor = executorch::extension::make_tensor_ptr<int64_t>(s, v, {}, {}, executorch::aten::ScalarType::Long);
                  }
                } else if (posShape == 3) { // [T]: either arange(0..T-1) or ones or explicit_pos
                  std::vector<executorch::aten::SizesType> s{(executorch::aten::SizesType)Tsend};
                  if (posDT == executorch::aten::ScalarType::Int) {
                    std::vector<int32_t> v; v.reserve(T);
                    if (explicit_pos >= 0) { v.push_back((int32_t)explicit_pos); }
                    else if (posMode == 2) { for (int32_t i = 0; i < T; ++i) v.push_back(i); }
                    else { v.assign(T, 1); }
                    startPosTensor = executorch::extension::make_tensor_ptr<int32_t>(s, v, {}, {}, executorch::aten::ScalarType::Int);
                  } else {
                    std::vector<int64_t> v; v.reserve(T);
                    if (explicit_pos >= 0) { v.push_back(explicit_pos); }
                    else if (posMode == 2) { for (int32_t i = 0; i < T; ++i) v.push_back((int64_t)i); }
                    else { v.assign(T, 1); }
                    startPosTensor = executorch::extension::make_tensor_ptr<int64_t>(s, v, {}, {}, executorch::aten::ScalarType::Long);
                  }
                } else { // posShape == 4 -> [1,T]: either arange(0..T-1) or ones or explicit_pos
                  std::vector<executorch::aten::SizesType> s{1, (executorch::aten::SizesType)Tsend};
                  if (posDT == executorch::aten::ScalarType::Int) {
                    std::vector<int32_t> v; v.reserve(T);
                    if (explicit_pos >= 0) { v.push_back((int32_t)explicit_pos); }
                    else if (posMode == 2) { for (int32_t i = 0; i < T; ++i) v.push_back(i); }
                    else { v.assign(T, 1); }
                    startPosTensor = executorch::extension::make_tensor_ptr<int32_t>(s, v, {}, {}, executorch::aten::ScalarType::Int);
                  } else {
                    std::vector<int64_t> v; v.reserve(T);
                    if (explicit_pos >= 0) { v.push_back(explicit_pos); }
                    else if (posMode == 2) { for (int32_t i = 0; i < T; ++i) v.push_back((int64_t)i); }
                    else { v.assign(T, 1); }
                    startPosTensor = executorch::extension::make_tensor_ptr<int64_t>(s, v, {}, {}, executorch::aten::ScalarType::Long);
                  }
                }

                if (!swappedOrder) {
                  inputs.emplace_back(startPosTensor);
                } else {
                  // swappedOrder == true means start_pos first, then tokens
                  // We already emplaced tokens; rebuild correct order
                  std::vector<::executorch::runtime::EValue> swapped;
                  swapped.emplace_back(startPosTensor);
                  swapped.emplace_back(inputs[0]);
                  auto res = mod.execute("forward", swapped);
                  if (!res.ok() && res.error() == ::executorch::runtime::Error::NotSupported) {
                    RCTLogInfo(@"[RNExecuTorchRunner] (fallback) forward failed with 2 inputs (NotSupported); trying tokens-only");
                    std::vector<::executorch::runtime::EValue> tokensOnly;
                    tokensOnly.emplace_back(inputs[0]);
                    return mod.execute("forward", tokensOnly);
                  }
                  return res;
                }
              }

              auto res = mod.execute("forward", inputs);
              if (!res.ok() && res.error() == ::executorch::runtime::Error::NotSupported) {
                RCTLogInfo(@"[RNExecuTorchRunner] (fallback) forward failed with 2 inputs (NotSupported); trying tokens-only");
                std::vector<::executorch::runtime::EValue> tokensOnly;
                tokensOnly.emplace_back(inputs[0]);
                return mod.execute("forward", tokensOnly);
              }
              return res;
            };

            std::vector<::executorch::runtime::EValue> outputs;

            // Prefill: try full-prompt once (dynamic shapes), then fall back to one-by-one
            if (needsPrefill) {
              bool prefilledFull = false;
              if (!oneInputForward && singleTokenOnly) {
                // Try a single full-prompt prefill with tokens [1,T] and start_pos arange [1,T]
                ::executorch::runtime::Error last = ::executorch::runtime::Error::InvalidArgument;
                for (int ps : {4, 3}) { // [1,T], then [T]
                  int pm = 2; // arange(0..T-1)
                  for (bool swp : {false, true}) {
                    auto res = build_and_run(tokenDTypeChosen, /*twoDim=*/true, startDTypeChosen, ps, pm, swp, /*explicit_pos*/-1);
                    if (res.ok()) {
                      outputs = res.get();
                      use2DTokens = true;
                      startPosShape = ps;
                      posModeChosen = pm;
                      orderSwappedChosen = swp;
                      formatLocked = true;
                      current_pos = (int64_t)ids.size();
                      needsPrefill = false;
                      prefilledFull = true;
                      RCTLogInfo(@"[RNExecuTorchRunner] Prefill: full-prompt succeeded with tokens [1,T], start_pos shape=%d mode=%d order=%s; current_pos=%lld",
                                 ps, pm, swp ? "start_pos,tokens" : "tokens,start_pos", (long long)current_pos);
                      break;
                    } else {
                      last = res.error();
                    }
                  }
                  if (prefilledFull) break;
                }
                if (!prefilledFull) {
                  RCTLogInfo(@"[RNExecuTorchRunner] Prefill: full-prompt attempt failed, falling back to one-by-one");
                }
              }

              if (!prefilledFull) {
                RCTLogInfo(@"[RNExecuTorchRunner] Prefilling %zu prompt tokens one-by-one", ids.size());
                for (size_t i = 0; i < ids.size(); ++i) {
                  // Temporarily set ids to contain only the current token for build_and_run
                  if (selfPtr->_shouldStop.load()) {
                    RCTLogInfo(@"[RNExecuTorchRunner] Fallback: cancellation requested during prefill");
                    break;
                  }

                  std::vector<int64_t> original_ids = ids;
                  ids = {original_ids[i]};

                  auto tryRes = build_and_run(tokenDTypeChosen, use2DTokens, startDTypeChosen, startPosShape, posModeChosen, orderSwappedChosen, (int64_t)i);
                  bool recovered = false;
                  std::vector<::executorch::runtime::EValue> recoveredOutputs;

                  // If Long fails, try Int32 tokens as a fallback
                  if (!tryRes.ok() && tokenDTypeChosen == executorch::aten::ScalarType::Long) {
                    auto altRes = build_and_run(executorch::aten::ScalarType::Int, use2DTokens, startDTypeChosen, startPosShape, posModeChosen, orderSwappedChosen, (int64_t)i);
                    if (altRes.ok()) {
                      recovered = true;
                      recoveredOutputs = altRes.get();
                      tokenDTypeChosen = executorch::aten::ScalarType::Int;
                      RCTLogInfo(@"[RNExecuTorchRunner] Prefill: recovered by switching token dtype to Int32");
                    }
                  }
                  // If still failing and start_pos is Long, try Int32 start_pos as a fallback
                  if (!tryRes.ok() && startDTypeChosen == executorch::aten::ScalarType::Long) {
                    auto altRes2 = build_and_run(tokenDTypeChosen, use2DTokens, executorch::aten::ScalarType::Int, startPosShape, posModeChosen, orderSwappedChosen, (int64_t)i);
                    if (altRes2.ok()) {
                      recovered = true;
                      recoveredOutputs = altRes2.get();
                      startDTypeChosen = executorch::aten::ScalarType::Int;
                      RCTLogInfo(@"[RNExecuTorchRunner] Prefill: recovered by switching start_pos dtype to Int32");
                    }
                  }

                // Restore full ids
                ids = original_ids;

                if (!tryRes.ok() && !recovered) {
                  // For 2-input models, if the first token fails, try alternative shapes/modes
                  if (i == 0 && !oneInputForward) {
                    ::executorch::runtime::Error last = tryRes.error();
                    bool fixed = false;
                    // If meta indicated strict single-token mode, prefer tokens [1,1] and start_pos [1]
                    bool strictSingleToken = (singleTokenOnly && use2DTokens && startPosShape == 1);
                    if (strictSingleToken) {
                      for (int pm : {0, 1, 2}) {
                        for (bool swp : {orderSwappedChosen, !orderSwappedChosen}) {
                          auto alt = build_and_run(tokenDTypeChosen, /*twoDimAlt=*/true, startDTypeChosen, /*ps=*/1, pm, swp, (int64_t)i);
                          if (alt.ok()) {
                            outputs = alt.get();
                            use2DTokens = true;      // enforce [1,1]
                            startPosShape = 1;       // enforce [1]
                            posModeChosen = pm;
                            orderSwappedChosen = swp;
                            formatLocked = true;
                            fixed = true;
                            RCTLogInfo(@"[RNExecuTorchRunner] Prefill: recovered (strict) with tokens [1,1], start_pos [1], mode=%d order=%s",
                                       pm, swp ? "start_pos,tokens" : "tokens,start_pos");
                            break;
                          } else {
                            last = alt.error();
                          }
                        }
                        if (fixed) break;
                      }
                    } else {
                      for (bool twoDimAlt : {true, false}) { // prefer 2D first
                        for (int ps : {1, 0, 2, 3, 4}) { // prefer [1], then scalar, then others
                          for (int pm : {0, 1, 2}) {    // zeros, len(T), arange
                            for (bool swp : {orderSwappedChosen, !orderSwappedChosen}) {
                              auto alt = build_and_run(tokenDTypeChosen, twoDimAlt, startDTypeChosen, ps, pm, swp, (int64_t)i);
                              if (alt.ok()) {
                                outputs = alt.get();
                                use2DTokens = twoDimAlt;
                                startPosShape = ps;
                                posModeChosen = pm;
                                orderSwappedChosen = swp;
                                formatLocked = true;
                                fixed = true;
                                RCTLogInfo(@"[RNExecuTorchRunner] Prefill: recovered with tokens %s, start_pos shape=%d mode=%d order=%s",
                                           twoDimAlt ? "[1,1]" : "[1]",
                                           ps, pm, swp ? "start_pos,tokens" : "tokens,start_pos");
                                break;
                              } else {
                                last = alt.error();
                              }
                            }
                            if (fixed) break;
                          }

	                    // If strict branch didn't fix it, try the broader search as a secondary fallback
	                    if (!fixed && strictSingleToken) {
	                      for (bool twoDimAlt : {true, false}) { // try both [1,1] and [1]
	                        for (int ps : {1, 0, 2, 3, 4}) { // [1], [], [1,1], [T], [1,T]
	                          for (int pm : {0, 1, 2}) {     // zeros, len(T), arange
	                            for (bool swp : {orderSwappedChosen, !orderSwappedChosen}) {
	                              auto alt2 = build_and_run(tokenDTypeChosen, twoDimAlt, startDTypeChosen, ps, pm, swp, (int64_t)i);
	                              if (alt2.ok()) {
	                                outputs = alt2.get();
	                                use2DTokens = twoDimAlt;
	                                startPosShape = ps;
	                                posModeChosen = pm;
	                                orderSwappedChosen = swp;
	                                formatLocked = true;
	                                fixed = true;
	                                RCTLogInfo(@"[RNExecuTorchRunner] Prefill: recovered (strict->general) with tokens %s, start_pos shape=%d mode=%d order=%s",
	                                           twoDimAlt ? "[1,1]" : "[1]",
	                                           ps, pm, swp ? "start_pos,tokens" : "tokens,start_pos");
	                                break;
	                              } else {
	                                last = alt2.error();
	                              }
	                            }
	                            if (fixed) break;
	                          }
	                          if (fixed) break;
	                        }
	                        if (fixed) break;
	                      }
	                    }

                          if (fixed) break;
                        }
                        if (fixed) break;
                      }
                    }
                    if (!fixed) {
                      RCTLogError(@"[RNExecuTorchRunner] Fallback: prefill failed at token %zu: %d (%s)", i, (int)last, ::executorch::runtime::to_string(last));
                      dispatch_async(dispatch_get_main_queue(), ^{
                        [selfPtr sendEventWithName:@"onError" body:@{ @"message": @"Fallback decode: prefill failed" }];
                      });
                      reject(@"GENERATION_ERROR", @"Fallback decode failed (prefill)", nil);
                      return;
                    }
                  } else {
                    RCTLogError(@"[RNExecuTorchRunner] Fallback: prefill failed at token %zu: %d (%s)", i, (int)tryRes.error(), ::executorch::runtime::to_string(tryRes.error()));
                    dispatch_async(dispatch_get_main_queue(), ^{
                      [selfPtr sendEventWithName:@"onError" body:@{ @"message": @"Fallback decode: prefill failed" }];
                    });
                    reject(@"GENERATION_ERROR", @"Fallback decode failed (prefill)", nil);
                    return;
                  }
                } else {
                  outputs = recovered ? recoveredOutputs : tryRes.get();
                }
              }
              current_pos = ids.size();
              needsPrefill = false;
              formatLocked = true;
              RCTLogInfo(@"[RNExecuTorchRunner] Prefill complete, current_pos=%lld", (long long)current_pos);
            } else if (!formatLocked) {
              // Probe combinations
              ::executorch::runtime::Error lastErr = ::executorch::runtime::Error::InvalidState;
              bool found = false;
              for (auto tokDT : {executorch::aten::ScalarType::Long, executorch::aten::ScalarType::Int}) {
                for (bool twoDim : {false, true}) {
                  for (auto posDT : {executorch::aten::ScalarType::Long, executorch::aten::ScalarType::Int}) {
                    for (int posShape : {0, 1, 2, 3, 4}) {
                      for (int posMode : {0, 1, 2, 3}) {
                        for (bool swapped : {false, true}) {
                          auto tryRes = build_and_run(tokDT, twoDim, posDT, posShape, posMode, swapped);
                          if (tryRes.ok()) {
                            outputs = tryRes.get();
                            tokenDTypeChosen = tokDT;
                            use2DTokens = twoDim;
                            startDTypeChosen = posDT;
                            startPosShape = posShape;
                            posModeChosen = posMode;
                            orderSwappedChosen = swapped;
                            formatLocked = true;
                            if (!oneInputForward) {
                              RCTLogInfo(@"[RNExecuTorchRunner] Fallback: using format tokens=%s %s, start_pos=%s shape=%d posMode=%d, order=%s",
                                         tokDT == executorch::aten::ScalarType::Long ? "int64" : "int32",
                                         twoDim ? "[1,T]" : "[T]",
                                         posDT == executorch::aten::ScalarType::Long ? "int64" : "int32",
                                         posShape,
                                         posMode,
                                         swapped ? "start_pos,tokens" : "tokens,start_pos");
                            } else {
                              // For 1-input models, treat this successful call as prefill with full prompt
                              oneInputPrefilled = true;
                              singleTokenOnly = true; // stream single tokens afterwards
                              current_pos = ids.size();
                              RCTLogInfo(@"[RNExecuTorchRunner] Fallback: using format tokens=%s %s (1-input forward: prefilled with full prompt; now streaming single tokens)",
                                         tokDT == executorch::aten::ScalarType::Long ? "int64" : "int32",
                                         twoDim ? "[1,T]" : "[T]");
                            }
                            found = true;
                            break;
                          } else {
                            lastErr = tryRes.error();
                            // Retry for 1-input models using meta-hinted token length if available
                            if (oneInputForward && tokenSeqDimHint > 0 && !singleTokenOnly) {
                              std::vector<int64_t> original_ids = ids;
                              size_t Tcut = (size_t)std::min<int64_t>(tokenSeqDimHint, (int64_t)original_ids.size());
                              if (original_ids.size() > Tcut) {
                                ids.assign(original_ids.end() - Tcut, original_ids.end());
                              }
                              // For meta-hinted tensors, prefer 2D [1,N]
                              auto tryRes2 = build_and_run(tokDT, /*twoDim=*/true, posDT, posShape, posMode, swapped);
                              // Restore full ids regardless of outcome
                              ids = original_ids;
                              if (tryRes2.ok()) {
                                outputs = tryRes2.get();
                                tokenDTypeChosen = tokDT;
                                use2DTokens = true; // [1,N]
                                startDTypeChosen = posDT;
                                startPosShape = posShape;
                                posModeChosen = posMode;
                                orderSwappedChosen = swapped;
                                formatLocked = true;
                                // Treat as prefill success; stream single tokens afterwards
                                oneInputPrefilled = true;
                                singleTokenOnly = true;
                                current_pos = (int64_t)Tcut;
                                RCTLogInfo(@"[RNExecuTorchRunner] Fallback: retry with meta-hinted [1,%lld] succeeded; now streaming single tokens", (long long)tokenSeqDimHint);
                                found = true;
                                break;
                              }
                            }
                          }
                        }
                        if (found) break;
                      }
                      if (found) break;
                    }
                    if (found) break;
                  }
                  if (found) break;
                }
                if (found) break;
              }
              if (!found) {
                RCTLogError(@"[RNExecuTorchRunner] Fallback: forward() failed during probe: %d (%s)", (int)lastErr, ::executorch::runtime::to_string(lastErr));
                dispatch_async(dispatch_get_main_queue(), ^{
                  [selfPtr sendEventWithName:@"onError" body:@{ @"message": [NSString stringWithFormat:@"Fallback decode: forward failed (%d)", (int)lastErr] }];
                });
                reject(@"GENERATION_ERROR", @"Fallback decode failed (forward probe)", nil);
                return;
              }
              // After successful probe, set current_pos to the number of prompt tokens
              if (found && singleTokenOnly) {
                current_pos = ids.size();
              }
            } else {
              // Mid-generation: pass explicit position
              auto tryRes = build_and_run(tokenDTypeChosen, use2DTokens, startDTypeChosen, startPosShape, posModeChosen, orderSwappedChosen, current_pos);
              if (!tryRes.ok()) {
                // If this is a 1-input model and we were attempting single-token streaming,
                // fall back to full-context recompute per step.
                if (oneInputForward && singleTokenOnly) {
                  RCTLogInfo(@"[RNExecuTorchRunner] Fallback: mid-gen streaming not supported (%d). Switching to full-context recompute.", (int)tryRes.error());
                  singleTokenOnly = false; // use full [1,T] tokens each step (or meta-hinted [1,N])
                  std::vector<int64_t> original_ids = ids;
                  if (oneInputForward && tokenSeqDimHint > 0) {
                    size_t Tclip = (size_t)std::min<int64_t>(tokenSeqDimHint, (int64_t)original_ids.size());
                    if (original_ids.size() > Tclip) {
                      ids.assign(original_ids.end() - Tclip, original_ids.end());
                      RCTLogInfo(@"[RNExecuTorchRunner] Recompute: clipping to meta-hinted window [1,%lld]", (long long)tokenSeqDimHint);
                    }
                  }
                  auto retryRes = build_and_run(tokenDTypeChosen, /*twoDim=*/true, startDTypeChosen, startPosShape, posModeChosen, orderSwappedChosen, /*explicit_pos*/-1);
                  // Restore ids regardless of outcome
                  ids = original_ids;
                  if (!retryRes.ok()) {
                    RCTLogError(@"[RNExecuTorchRunner] Fallback: forward() failed after switching to recompute: %d (%s)", (int)retryRes.error(), ::executorch::runtime::to_string(retryRes.error()));
                    dispatch_async(dispatch_get_main_queue(), ^{
                      [selfPtr sendEventWithName:@"onError" body:@{ @"message": @"Fallback decode: forward failed mid-generation (recompute)" }];
                    });
                    reject(@"GENERATION_ERROR", @"Fallback decode failed (forward mid-gen, recompute)", nil);
                    return;
                  }
                  outputs = retryRes.get();
                } else {
                  RCTLogError(@"[RNExecuTorchRunner] Fallback: forward() failed mid-generation: %d (%s)", (int)tryRes.error(), ::executorch::runtime::to_string(tryRes.error()));
                  dispatch_async(dispatch_get_main_queue(), ^{
                    [selfPtr sendEventWithName:@"onError" body:@{ @"message": @"Fallback decode: forward failed mid-generation" }];
                  });
                  reject(@"GENERATION_ERROR", @"Fallback decode failed (forward mid-gen)", nil);
                  return;
                }
              } else {
                outputs = tryRes.get();
              }
              current_pos++; // Increment position for next token
            }

            if (outputs.empty() || !outputs[0].isTensor()) {
              RCTLogError(@"[RNExecuTorchRunner] Fallback: invalid outputs");
              dispatch_async(dispatch_get_main_queue(), ^{
                [selfPtr sendEventWithName:@"onError" body:@{ @"message": @"Fallback decode: invalid outputs" }];
              });
              reject(@"GENERATION_ERROR", @"Fallback decode failed (outputs)", nil);
              return;
            }
            auto logits_tensor = outputs[0].toTensor();

            // Sample next token (supports Float/Half/BFloat16/UInt16)
            int32_t next_id = 0;
            {
              // Minimal context for ET_SWITCH — do NOT abort on unsupported dtype; handle gracefully
              bool unsupportedLogits = false;
              struct { bool* p; void fail(torch::executor::Error){ if (p) *p = true; } } ctx{ &unsupportedLogits };
              ET_SWITCH_FOUR_TYPES(
                Float,
                Half,
                BFloat16,
                UInt16,
                logits_tensor.scalar_type(),
                ctx,
                "fallback_logits",
                CTYPE,
                [&]() {
                  auto* logits = logits_tensor.mutable_data_ptr<CTYPE>();
                  ssize_t vocab_size = logits_tensor.size(logits_tensor.dim() - 1);
                  if (logits_tensor.dim() >= 2) {
                    auto timesteps = logits_tensor.size(logits_tensor.dim() - 2);
                    if (timesteps > 0) {
                      logits += (timesteps - 1) * vocab_size;
                    }
                  }
                  ::executorch::extension::llm::Sampler sampler((int32_t)vocab_size, temperature);
                  next_id = sampler.sample<CTYPE>(logits);
                }
              );
              if (unsupportedLogits) {
                RCTLogError(@"[RNExecuTorchRunner] Fallback: unsupported logits dtype=%d", (int)logits_tensor.scalar_type());
                dispatch_async(dispatch_get_main_queue(), ^{
                  [selfPtr sendEventWithName:@"onError" body:@{ @"message": @"Fallback decode: unsupported logits dtype" }];
                });
                reject(@"GENERATION_ERROR", @"Fallback decode failed (unsupported logits dtype)", nil);
                return;
              }
            }

            int64_t prev = ids.back();
            ids.push_back((int64_t)next_id);

            auto d = fbTokenizer->decode((uint64_t)prev, (uint64_t)next_id);
            if (d.ok()) {
              if (tokenCount == 0) firstTokenTime = [[NSDate date] timeIntervalSince1970];
              NSString *piece = [NSString stringWithUTF8String:d.get().c_str()];
              [fullResponse appendString:piece];
              tokenCount++;
              generated++;
              dispatch_async(dispatch_get_main_queue(), ^{
                [selfPtr sendEventWithName:@"onToken" body:piece];
              });
            }

            if ((uint64_t)next_id == fbTokenizer->eos_tok()) {
              RCTLogInfo(@"[RNExecuTorchRunner] Fallback: hit EOS");
              break;
            }
          }

          // Emit completion stats
          NSTimeInterval endTime = [[NSDate date] timeIntervalSince1970];
          NSTimeInterval totalTime = (endTime - startTime) * 1000; // ms
          NSTimeInterval timeToFirstToken = (firstTokenTime - startTime) * 1000; // ms
          double tokensPerSecond = (endTime - (firstTokenTime > 0 ? firstTokenTime : startTime)) > 0
            ? (double)tokenCount / (endTime - (firstTokenTime > 0 ? firstTokenTime : startTime))
            : 0.0;

          NSDictionary *statsDict = @{
            @"promptTokens": @(promptTokens),
            @"generatedTokens": @(tokenCount - (echo ? (int)std::max((size_t)0, ids_u.size() - 1) : 0)),
            @"totalTime": @(totalTime),
            @"tokensPerSecond": @(tokensPerSecond),
            @"timeToFirstToken": @(timeToFirstToken)
          };
          dispatch_async(dispatch_get_main_queue(), ^{
            [selfPtr sendEventWithName:@"onComplete" body:statsDict];
          });

          RCTLogInfo(@"[RNExecuTorchRunner] Fallback generation complete: %d tokens", tokenCount);
          resolve([fullResponse copy]);
          return;
        }

        // Treat user cancellation as a non-fatal outcome (SDK may not define Error::Cancelled)
        if (selfPtr->_shouldStop.load()) {
          RCTLogInfo(@"[RNExecuTorchRunner] Generation cancelled by user");
          NSDictionary *statsDict = @{ @"promptTokens": @0, @"generatedTokens": @0, @"totalTime": @0, @"tokensPerSecond": @0, @"timeToFirstToken": @0 };
          dispatch_async(dispatch_get_main_queue(), ^{
            [selfPtr sendEventWithName:@"onComplete" body:statsDict];
          });
          resolve(@"");
          return;
        } else {
          // Default error handling
          RCTLogError(@"[RNExecuTorchRunner] generate() failed with error code: %d (%s)", (int)error, ::executorch::runtime::to_string(error));
          dispatch_async(dispatch_get_main_queue(), ^{
            [selfPtr sendEventWithName:@"onError" body:@{ @"message": [NSString stringWithFormat:@"ExecuTorch generation failed: %d (%s)", (int)error, ::executorch::runtime::to_string(error)] }];
          });
          reject(@"GENERATION_ERROR", @"Failed to generate text", nil);
          return;
        }
      }
      }


      RCTLogInfo(@"[RNExecuTorchRunner] Generation complete: %d tokens", tokenCount);
      resolve([fullResponse copy]);

  });
}

RCT_EXPORT_METHOD(stop:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)
{
  RCTLogInfo(@"[RNExecuTorchRunner] Stopping generation");
  _shouldStop.store(true);

  // If the underlying runner exposes a stop/cancel API, call it when available.
  // This block is compiled only if your build defines one of these macros.
  #if defined(ET_RUNNER_HAS_STOP)
    if (_runner) { try { _runner->stop(); } catch (...) {} }
  #elif defined(ET_RUNNER_HAS_CANCEL)
    if (_runner) { try { _runner->cancel(); } catch (...) {} }
  #endif

  // We also gate emission using the _shouldStop flag inside callbacks/loops.
  resolve(@YES);
}

RCT_EXPORT_METHOD(warmup:(NSString *)prompt
                  numTokens:(NSInteger)numTokens
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)
{
  if (!_runner || !_runner->is_loaded()) {
    reject(@"NOT_LOADED", @"Model not loaded. Call loadModel first.", nil);
    return;
  }

  dispatch_async(self.generationQueue, ^{
    @try {
      RCTLogInfo(@"[RNExecuTorchRunner] Warming up with %ld tokens", (long)numTokens);

      // Run a warmup generation
      GenerationConfig warmupConfig;
      warmupConfig.max_new_tokens = (int32_t)numTokens;
      warmupConfig.warming = true;
      warmupConfig.echo = false;

      auto error = self->_runner->generate(
        [prompt UTF8String],
        warmupConfig,
        [](const std::string&) {}, // Empty token callback
        [](const Stats&) {}         // Empty stats callback
      );

      if (error != ::executorch::runtime::Error::Ok) {
        reject(@"WARMUP_ERROR", @"Failed to warmup model", nil);
        return;
      }

      RCTLogInfo(@"[RNExecuTorchRunner] Warmup complete");
      resolve(@YES);

    } @catch (NSException *exception) {
      RCTLogError(@"[RNExecuTorchRunner] Exception during warmup: %@", exception.reason);
      reject(@"WARMUP_ERROR", exception.reason, nil);
    }
  });
}

RCT_EXPORT_METHOD(unload:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)
{
  dispatch_async(self.generationQueue, ^{
    @try {
      RCTLogInfo(@"[RNExecuTorchRunner] Unloading model");

      // Reset the runner to free memory
      self->_runner.reset();

      self.modelLoaded = NO;
      RCTLogInfo(@"[RNExecuTorchRunner] Model unloaded successfully");
      resolve(@YES);

    } @catch (NSException *exception) {
      RCTLogError(@"[RNExecuTorchRunner] Exception during unload: %@", exception.reason);
      reject(@"UNLOAD_ERROR", exception.reason, nil);
    }
  });
}

@end

