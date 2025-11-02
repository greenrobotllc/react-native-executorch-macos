import React, {useEffect, useState} from 'react';
import {SafeAreaView, ScrollView, Text, View} from 'react-native';
import {ExecuTorchRunner} from 'react-native-executorch-macos';

export default function App() {
  const [log, setLog] = useState<string>('Starting...');

  useEffect(() => {
    (async () => {
      try {
        // TODO: update these paths before running
        const modelPath = '/absolute/path/to/model.pte';
        const tokenizerPath = '/absolute/path/to/tokenizer.json';

        setLog(l => l + '\nLoading model...');
        const runner = await ExecuTorchRunner.loadModel({
          modelPath,
          tokenizerPath,
          tokenizerType: 'huggingface',
        });

        setLog(l => l + '\nGenerating...');
        await runner.generate(
          'Hello from ExecuTorch on React Native macOS!',
          {maxNewTokens: 16, temperature: 0.8},
          {
            onToken: t => setLog(l => l + t),
            onComplete: s => setLog(l => l + `\nDone: ${JSON.stringify(s)}`),
            onError: e => setLog(l => l + `\nError: ${String(e)}`),
          },
        );
      } catch (e) {
        setLog(l => l + `\nException: ${String(e)}`);
      }
    })();
  }, []);

  return (
    <SafeAreaView>
      <ScrollView>
        <View style={{padding: 16}}>
          <Text style={{fontFamily: 'Menlo'}}>{log}</Text>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}
