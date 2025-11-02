import { NativeModules, NativeEventEmitter } from 'react-native';

// Native module name exported by RNExecuTorchRunner.mm
const { ExecuTorchRunner: NativeRunner } = NativeModules as any;
if (!NativeRunner) {
  throw new Error(
    'ExecuTorchRunner native module not found. Make sure the pod is installed and linked.'
  );
}

const emitter = new NativeEventEmitter(NativeRunner);

export type ModelConfig = {
  modelPath: string;           // Absolute path to .pte
  tokenizerPath: string;       // Absolute path to tokenizer.json/model
  tokenizerType?: 'huggingface' | 'sentencepiece';
};

export type GenerationConfig = {
  maxNewTokens?: number;       // default 100
  temperature?: number;        // default 0.8
  echo?: boolean;              // default false
};

export type GenerationCallbacks = {
  onToken?: (token: string) => void;
  onComplete?: (stats: any) => void;
  onError?: (err: Error) => void;
};

export class ExecuTorchRunner {
  private loaded = false;
  private generating = false;

  static async loadModel(config: ModelConfig): Promise<ExecuTorchRunner> {
    const inst = new ExecuTorchRunner();
    await NativeRunner.loadModel({
      modelPath: config.modelPath,
      tokenizerPath: config.tokenizerPath,
      tokenizerType: config.tokenizerType || 'huggingface',
    });
    inst.loaded = true;
    return inst;
  }

  async isLoaded(): Promise<boolean> {
    const ok = await NativeRunner.isLoaded();
    return !!ok;
  }

  async generate(
    prompt: string,
    config?: GenerationConfig,
    callbacks?: GenerationCallbacks,
  ): Promise<string> {
    if (!this.loaded) throw new Error('Model not loaded');
    if (this.generating) throw new Error('Already generating');

    this.generating = true;

    const tokenListener = emitter.addListener('onToken', (t: string) => {
      callbacks?.onToken?.(t);
    });
    const completeListener = emitter.addListener('onComplete', (stats: any) => {
      callbacks?.onComplete?.(stats);
      tokenListener.remove();
      completeListener.remove();
      errorListener.remove();
      this.generating = false;
    });
    const errorListener = emitter.addListener('onError', (err: any) => {
      const e = new Error(err?.message || 'Generation error');
      callbacks?.onError?.(e);
      tokenListener.remove();
      completeListener.remove();
      errorListener.remove();
      this.generating = false;
    });

    const result: string = await NativeRunner.generate(
      prompt,
      config || { maxNewTokens: 100, temperature: 0.8 },
    );

    // If native side did not stream, return the final string
    return result || '';
  }
}

export default ExecuTorchRunner;

