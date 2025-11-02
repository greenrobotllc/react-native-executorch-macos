//
//  RNExecuTorch.mm
//  AlienTavernMobile
//
//  Native macOS bridge for PyTorch ExecuTorch 1.0
//  Migrated from C++ API to native Objective-C API
//

#import "RNExecuTorch.h"
#import <React/RCTLog.h>
#import <mach/mach.h>

// ExecuTorch 1.0 native Objective-C API
#if __has_include(<ExecuTorch/ExecuTorch.h>)

#import <ExecuTorch/ExecuTorch.h>

@implementation RNExecuTorch {
    ExecuTorchModule *_module;
    NSString *_currentModelPath;
    BOOL _isModelLoaded;
    BOOL _debugLogging;
    NSString *_inferenceMethodName;
}

RCT_EXPORT_MODULE();

+ (BOOL)requiresMainQueueSetup
{
    return NO;
}

- (instancetype)init
{
    self = [super init];
    if (self) {
        _isModelLoaded = NO;
        _currentModelPath = nil;
        _debugLogging = NO;
        _inferenceMethodName = @"forward";
    }
    return self;
}

- (NSArray<NSString *> *)supportedEvents
{
    return @[@"onGenerationProgress"];
}

/**
 * Return the dev cache directory path (macOS): ~/\.alientavern/models/
 */
RCT_EXPORT_METHOD(getDevCachePath:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)
{
    @try {
        NSString *home = NSHomeDirectory();
        NSString *path = [home stringByAppendingPathComponent:@".alientavern/models/"];
        resolve(path);
    } @catch (NSException *exception) {
        reject(@"DEV_CACHE_ERROR", exception.reason, nil);
    }
}

RCT_EXPORT_METHOD(setDebugLogging:(BOOL)enabled)
{
    _debugLogging = enabled;
    RCTLogInfo(@"[RNExecuTorch] Debug logging %@", enabled ? @"ENABLED" : @"DISABLED");
}

#pragma mark - Module Methods

/**
 * Load a model from the given file path
 * @param modelPath Absolute path to the .pte model file
 */
RCT_EXPORT_METHOD(loadModel:(NSString *)modelPath
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)
{
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        @try {
            RCTLogInfo(@"[RNExecuTorch] Loading model from: %@", modelPath);

            // Check if file exists
            if (![[NSFileManager defaultManager] fileExistsAtPath:modelPath]) {
                reject(@"FILE_NOT_FOUND",
                       [NSString stringWithFormat:@"Model file not found at path: %@", modelPath],
                       nil);
                return;
            }

            // Unload existing model if any
            if (self->_isModelLoaded) {
                RCTLogInfo(@"[RNExecuTorch] Unloading existing model");
                self->_module = nil;
                self->_isModelLoaded = NO;
            }

            // Create module with memory-mapped loading for efficiency
            self->_module = [[ExecuTorchModule alloc] initWithFilePath:modelPath
                                                              loadMode:ExecuTorchModuleLoadModeMmap];

            NSError *error = nil;

            // Inspect available methods before loading
            NSSet<NSString *> *methodNames = [self->_module methodNames:&error];
            if (methodNames) {
                RCTLogInfo(@"[RNExecuTorch] Available methods: %@", methodNames);
            }

            // Skip enable_dynamic_shape for now; the "simple" export does not require it
            // and some builds expose it as a no-arg method. We'll avoid calling it to reduce noise.

            // Choose a method to load (prefer 'forward')
            NSString *desiredMethod = @"forward";
            if (!methodNames || ![methodNames containsObject:@"forward"]) {
                NSArray<NSString *> *candidates = @[ @"forward", @"tokens_to_logits", @"generate", @"run", @"__forward__" ];
                for (NSString *cand in candidates) {
                    if (methodNames && [methodNames containsObject:cand]) { desiredMethod = cand; break; }
                }
                if (methodNames && ![methodNames containsObject:desiredMethod]) {
                    desiredMethod = [[methodNames allObjects] firstObject] ?: @"forward";
                }
            }

            BOOL success = [self->_module loadMethod:desiredMethod error:&error];
            if (!success) {
                NSString *desc = error ? (error.debugDescription ?: error.localizedDescription) : @"(no error)";
                NSString *domain = error ? error.domain : @"(no domain)";
                NSInteger code = error ? error.code : 0;
                NSDictionary *userInfo = error.userInfo ?: @{};
                NSString *errorMsg = [NSString stringWithFormat:@"Failed to load method '%@': %@ (available: %@) [domain=%@ code=%ld userInfo=%@]",
                                     desiredMethod, desc, methodNames, domain, (long)code, userInfo];
                RCTLogError(@"[RNExecuTorch] %@", errorMsg);
                self->_module = nil;
                reject(@"LOAD_ERROR", errorMsg, error);
                return;
            }

            self->_inferenceMethodName = desiredMethod;
            RCTLogInfo(@"[RNExecuTorch] Loaded method: %@", desiredMethod);

            // Check if this is a KV cache model and initialize it
            NSSet<NSString *> *kvMethodNames = [self->_module methodNames:&error];
            if (kvMethodNames && [kvMethodNames containsObject:@"use_kv_cache"]) {
                RCTLogInfo(@"[RNExecuTorch] Detected KV cache model, initializing...");

                // Load use_kv_cache method
                BOOL kvLoadSuccess = [self->_module loadMethod:@"use_kv_cache" error:&error];
                if (kvLoadSuccess) {
                    // Explicitly DISABLE KV cache for 'simple' models to avoid custom LLM ops
                    ExecuTorchValue *falseValue = [ExecuTorchValue valueWithBoolean:NO];
                    NSArray<ExecuTorchValue *> *kvResult = [self->_module executeMethod:@"use_kv_cache"
                                                                               withInput:falseValue
                                                                                   error:&error];
                    if (kvResult) {
                        RCTLogInfo(@"[RNExecuTorch] KV cache disabled successfully");
                    } else {
                        RCTLogError(@"[RNExecuTorch] use_kv_cache call failed: %@", error.localizedDescription);
                    }
                }
            }

            self->_currentModelPath = modelPath;
            self->_isModelLoaded = YES;

            RCTLogInfo(@"[RNExecuTorch] Model loaded successfully with ExecuTorch 1.0 native API");
            resolve(@{
                @"success": @YES,
                @"modelPath": modelPath
            });

        } @catch (NSException *exception) {
            NSString *errorMsg = [NSString stringWithFormat:@"Exception loading model: %@", exception.reason];
            RCTLogError(@"[RNExecuTorch] %@", errorMsg);
            reject(@"LOAD_EXCEPTION", errorMsg, nil);
        }
    });
}

/**
 * Unload the current model and free memory
 */
RCT_EXPORT_METHOD(unloadModel:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)
{
    @try {
        if (self->_isModelLoaded) {
            RCTLogInfo(@"[RNExecuTorch] Unloading model");
            self->_module = nil;
            self->_isModelLoaded = NO;
            self->_currentModelPath = nil;
            resolve(@{@"success": @YES});
        } else {
            resolve(@{@"success": @YES, @"message": @"No model loaded"});
        }
    } @catch (NSException *exception) {
        reject(@"UNLOAD_ERROR", exception.reason, nil);
    }
}

/**
 * Check if a model is currently loaded
 */
RCT_EXPORT_METHOD(isModelLoaded:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)
{
    resolve(@{
        @"loaded": @(self->_isModelLoaded),
        @"modelPath": self->_currentModelPath ?: [NSNull null]
    });
}

/**
 * Generate text from a prompt
 * @param prompt The input text prompt
 * @param maxTokens Maximum number of tokens to generate
 */
RCT_EXPORT_METHOD(generate:(NSString *)prompt
                  maxTokens:(NSInteger)maxTokens
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)
{
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        @try {
            if (!self->_isModelLoaded) {
                reject(@"NO_MODEL", @"No model is loaded. Call loadModel first.", nil);
                return;
            }

            // 1) Minimal tokenizer: byte-level (ASCII) -> int64 tokens
            NSMutableArray<NSNumber *> *tokens = [NSMutableArray array];
            NSData *data = [prompt dataUsingEncoding:NSUTF8StringEncoding];
            const uint8_t *bytes = (const uint8_t *)[data bytes];
            NSUInteger len = [data length];
            for (NSUInteger i = 0; i < len; ++i) {
                [tokens addObject:@((int64_t)bytes[i])];
            }

            // Cap by maxTokens for this demo to keep shapes reasonable
            if (maxTokens > 0 && (NSUInteger)maxTokens < tokens.count) {
                NSUInteger original = tokens.count;
                [tokens removeObjectsInRange:NSMakeRange(maxTokens, tokens.count - maxTokens)];
                if (self->_debugLogging) {
                    RCTLogInfo(@"[RNExecuTorch] Truncated prompt tokens from %lu to %ld due to maxTokens",
                              (unsigned long)original, (long)maxTokens);
                }
            }
            if (tokens.count == 0) {
                [tokens addObject:@0];
            }

            // 2) Create input tensor [1, L] int64 using native API
            // ExecuTorch 0.7 echo model was exported with fixed input shape [1, 16] int64.
            const NSInteger kEchoLen = 16;
            while (tokens.count < kEchoLen) {
                [tokens addObject:@0];
            }
            if (tokens.count > kEchoLen) {
                [tokens removeObjectsInRange:NSMakeRange(kEchoLen, tokens.count - kEchoLen)];
            }

            RCTLogInfo(@"[RNExecuTorch] Building input tensor with shape [1, %ld]", (long)kEchoLen);

            // Convert NSArray to C array for tensor creation
            int64_t *tokenData = (int64_t *)malloc(sizeof(int64_t) * tokens.count);
            for (NSUInteger i = 0; i < tokens.count; i++) {
                tokenData[i] = [tokens[i] longLongValue];
            }

            ExecuTorchTensor *inputTensor = [[ExecuTorchTensor alloc] initWithBytes:tokenData
                                                                              shape:@[@1, @(kEchoLen)]
                                                                           dataType:ExecuTorchDataTypeLong];
            free(tokenData);

            // 3) Run forward()
            NSError *error = nil;
            ExecuTorchValue *inputValue = [ExecuTorchValue valueWithTensor:inputTensor];
            NSArray<ExecuTorchValue *> *outputs = [self->_module executeMethod:self->_inferenceMethodName withInput:inputValue error:&error];

            if (!outputs || outputs.count == 0) {
                NSString *errorMsg = [NSString stringWithFormat:@"Forward pass failed: %@",
                                     error.localizedDescription];
                reject(@"INFERENCE_ERROR", errorMsg, error);
                return;
            }

            ExecuTorchValue *firstOutput = outputs.firstObject;
            if (!firstOutput.isTensor) {
                reject(@"INVALID_OUTPUT", @"Model output is not a tensor", nil);
                return;
            }

            ExecuTorchTensor *outTensor = firstOutput.tensorValue;

            // 4) Decode: read first row as int64 tokens and convert back to text
            __block NSMutableString *generatedText = [NSMutableString string];
            __block NSInteger totalTokens = 0;

            [outTensor bytesWithHandler:^(const void *pointer, NSInteger count, ExecuTorchDataType dataType) {
                if (dataType == ExecuTorchDataTypeLong) {
                    const int64_t *out_ptr = (const int64_t *)pointer;
                    totalTokens = count;

                    for (NSInteger i = 0; i < count; ++i) {
                        uint8_t ch = (uint8_t)MAX(0, MIN(255, out_ptr[i]));
                        [generatedText appendFormat:@"%c", (char)ch];
                    }
                }
            }];

            resolve(@{
                @"text": generatedText,
                @"tokensGenerated": @(totalTokens)
            });
        } @catch (NSException *exception) {
            NSString *errorMsg = [NSString stringWithFormat:@"Generation failed: %@", exception.reason];
            RCTLogError(@"[RNExecuTorch] %@", errorMsg);
            reject(@"GENERATION_ERROR", errorMsg, nil);
        }
    });
}

/**
 * Autoregressive generation from pre-tokenized IDs (generative models)
 * @param inputIds Array<number> prompt token IDs (int64)
 * @param maxTokens Max new tokens to generate
 * @param temperature Sampling temperature (<=0 => greedy)
 */
RCT_EXPORT_METHOD(generateIds:(NSArray<NSNumber *> *)inputIds
                  maxTokens:(NSInteger)maxTokens
                  temperature:(double)temperature
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)
{
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        @try {
            if (!self->_isModelLoaded) {
                reject(@"NO_MODEL", @"No model is loaded. Call loadModel first.", nil);
                return;
            }

            // Copy input tokens
            NSMutableArray<NSNumber *> *tokens = [NSMutableArray arrayWithArray:inputIds];
            if (tokens.count == 0) {
                [tokens addObject:@0];
            }

            if (self->_debugLogging) {
                RCTLogInfo(@"[RNExecuTorch] generateIds: tokens=%lu, maxTokens=%ld, temperature=%.3f",
                          (unsigned long)tokens.count, (long)maxTokens, temperature);
            }

            NSError *error = nil;

            // Get method metadata to understand input signature
            ExecuTorchMethodMetadata *metadata = [self->_module methodMetadata:self->_inferenceMethodName error:&error];
            if (!metadata) {
                NSString *errorMsg = [NSString stringWithFormat:@"Failed to get method metadata: %@",
                                     error.localizedDescription];
                RCTLogError(@"[RNExecuTorch] %@", errorMsg);
                reject(@"METADATA_ERROR", errorMsg, error);
                return;
            }

            NSInteger numInputs = metadata.inputValueTags.count;
            if (self->_debugLogging) {
                RCTLogInfo(@"[RNExecuTorch] forward() expects %ld inputs", (long)numInputs);
            }

            // Clamp to a sane upper bound to avoid runaway loops
            NSInteger steps = MAX(0, MIN(maxTokens, 4096));
            if (self->_debugLogging && steps != maxTokens) {
                RCTLogInfo(@"[RNExecuTorch] Clamped maxTokens from %ld to %ld", (long)maxTokens, (long)steps);
            }

            // Autoregressive generation loop
            for (NSInteger step = 0; step < steps; ++step) {
                NSInteger L = tokens.count;

                // Create input tensor [1, L] with int64 token IDs
                int64_t *tokenData = (int64_t *)malloc(L * sizeof(int64_t));
                for (NSInteger i = 0; i < L; i++) {
                    tokenData[i] = [tokens[i] longLongValue];
                }

                ExecuTorchTensor *inputTensor = [[ExecuTorchTensor alloc] initWithBytes:tokenData
                                                                                   shape:@[@1, @(L)]
                                                                                dataType:ExecuTorchDataTypeLong];
                free(tokenData);

                if (!inputTensor) {
                    reject(@"TENSOR_ERROR", @"Failed to create input tensor", nil);
                    return;
                }

                // Run forward pass
                ExecuTorchValue *inputValue = [ExecuTorchValue valueWithTensor:inputTensor];
                NSArray<ExecuTorchValue *> *outputs = [self->_module executeMethod:self->_inferenceMethodName withInput:inputValue error:&error];
                if (!outputs || outputs.count == 0) {
                    NSString *errorMsg = [NSString stringWithFormat:@"Forward pass failed at step %ld: %@",
                                         (long)step, error.localizedDescription];
                    RCTLogError(@"[RNExecuTorch] %@", errorMsg);
                    reject(@"INFERENCE_ERROR", errorMsg, error);
                    return;
                }

                // Extract logits from output
                ExecuTorchValue *outputValue = outputs[0];
                if (![outputValue isTensor]) {
                    reject(@"INVALID_OUTPUT", @"Model output is not a tensor", nil);
                    return;
                }

                ExecuTorchTensor *logitsTensor = outputValue.tensorValue;
                NSArray<NSNumber *> *logitsShape = logitsTensor.shape;

                // Get vocab size (last dimension)
                NSInteger vocabSize = [logitsShape.lastObject integerValue];
                if (vocabSize <= 0) {
                    reject(@"INVALID_OUTPUT", @"Invalid vocab size in output tensor", nil);
                    return;
                }

                // Get logits data and sample next token
                __block NSInteger bestIdx = 0;
                __block float bestVal = -INFINITY;

                [logitsTensor bytesWithHandler:^(const void *pointer, NSInteger count, ExecuTorchDataType dataType) {
                    const float *logits = (const float *)pointer;

                    // Sample next token (greedy or temperature-based)
                    for (NSInteger i = 0; i < vocabSize; i++) {
                        float v = logits[i];
                        if (temperature > 0.01) {
                            v = v / (float)temperature;
                        }
                        if (v > bestVal) {
                            bestVal = v;
                            bestIdx = i;
                        }
                    }
                }];

                // Add new token to sequence
                [tokens addObject:@(bestIdx)];

                // Send progress event
                [self sendEventWithName:@"onGenerationProgress" body:@{
                    @"tokensGenerated": @(step + 1),
                    @"tokenId": @(bestIdx),
                    @"totalTokens": @(tokens.count)
                }];
            }

            // Resolve with final token list
            resolve(@{ @"tokens": tokens });
        } @catch (NSException *exception) {
            NSString *errorMsg = [NSString stringWithFormat:@"GenerationIds failed: %@", exception.reason];
            RCTLogError(@"[RNExecuTorch] %@", errorMsg);
            reject(@"GENERATION_ERROR", errorMsg, nil);
        }
    });
}


/**
 * Forward pass with raw tensor input (advanced usage)
 * @param inputData Array of input tensor data
 */
RCT_EXPORT_METHOD(forward:(NSArray *)inputData
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)
{
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        @try {
            if (!self->_isModelLoaded) {
                reject(@"NO_MODEL", @"No model is loaded. Call loadModel first.", nil);
                return;
            }

            RCTLogInfo(@"[RNExecuTorch] Running forward pass");

            // TODO: Implement forward pass
            // This requires converting inputData to ExecuTorch tensors
            // and running the model's forward method

            reject(@"NOT_IMPLEMENTED", @"Forward pass not yet implemented", nil);

        } @catch (NSException *exception) {
            reject(@"FORWARD_ERROR", exception.reason, nil);
        }
    });
}

/**
 * Get model metadata
 */
RCT_EXPORT_METHOD(getModelInfo:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)
{
    @try {
        if (!self->_isModelLoaded) {
            reject(@"NO_MODEL", @"No model is loaded", nil);
            return;
        }

        // Get model metadata
        NSMutableDictionary *info = [NSMutableDictionary dictionary];
        info[@"modelPath"] = self->_currentModelPath;
        info[@"loaded"] = @YES;

        // Get file size
        NSError *error = nil;
        NSDictionary *attributes = [[NSFileManager defaultManager]
                                   attributesOfItemAtPath:self->_currentModelPath
                                   error:&error];
        if (!error) {
            info[@"fileSize"] = attributes[NSFileSize];
        }

        // Model type heuristic from filename
        NSString *lower = [self->_currentModelPath lowercaseString];
        NSString *modelType = @"unknown";
        if ([lower containsString:@"echo"]) modelType = @"echo";
        else if ([lower containsString:@"tinyllama"] || [lower containsString:@"stories110m"] || [lower containsString:@"smollm"] || [lower containsString:@"chat"] || [lower containsString:@"llama"]) modelType = @"generative";
        info[@"modelType"] = modelType;

        // TODO: Add more metadata from the model
        // - Number of parameters
        // - Input/output shapes

        resolve(info);

    } @catch (NSException *exception) {
        reject(@"INFO_ERROR", exception.reason, nil);
    }
}

/**
 * Get memory usage statistics
 */
RCT_EXPORT_METHOD(getMemoryStats:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)
{
    @try {
        // Get current memory usage
        struct task_basic_info info;
        mach_msg_type_number_t size = TASK_BASIC_INFO_COUNT;
        kern_return_t kerr = task_info(mach_task_self(),
                                       TASK_BASIC_INFO,
                                       (task_info_t)&info,
                                       &size);

        if (kerr == KERN_SUCCESS) {
            resolve(@{
                @"residentSize": @(info.resident_size),
                @"virtualSize": @(info.virtual_size)
            });
        } else {
            reject(@"MEMORY_ERROR", @"Failed to get memory stats", nil);
        }

    } @catch (NSException *exception) {
        reject(@"MEMORY_ERROR", exception.reason, nil);
    }
}

@end



#endif // __has_include(<ExecuTorch/ExecuTorch.h>)