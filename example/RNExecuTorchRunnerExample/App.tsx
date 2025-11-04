import React, {useEffect, useRef, useState} from 'react';
import {SafeAreaView, ScrollView, Text, View, TextInput, Button} from 'react-native';
import {ExecuTorchRunner} from 'react-native-executorch-macos';

export default function App() {
  const [status, setStatus] = useState<string>('Starting...');
  const [prompt, setPrompt] = useState<string>('Hey everyone,\n\nI\'m trying to use the React Native ');
  const [output, setOutput] = useState<string>('');
  const [isLoaded, setIsLoaded] = useState<boolean>(false);
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const runnerRef = useRef<ExecuTorchRunner | null>(null);

  useEffect(() => {
    (async () => {
      try {
        // Update these absolute paths for your machine
        const modelPath = '/Users/andytriboletti/Documents/GitHub/alientavern-v2/mobile/local-models/smollm2-135m-xnnpack.pte';
        const tokenizerPath = '/Users/andytriboletti/Documents/GitHub/alientavern-v2/mobile/local-models/smollm2_135m_tokenizer/tokenizer.json';

        setStatus(s => s + '\nLoading model...');
        const runner = await ExecuTorchRunner.loadModel({
          modelPath,
          tokenizerPath,
          tokenizerType: 'huggingface',
        });
        runnerRef.current = runner;
        setIsLoaded(true);
        setStatus(s => s + '\nReady.');
      } catch (e) {
        setStatus(s => s + `\nException: ${String(e)}`);
      }
    })();
  }, []);

  const onGenerate = async () => {
    if (!runnerRef.current || isGenerating) return;
    setIsGenerating(true);
    setStatus('Generating...');
    setOutput('');
    try {
      await runnerRef.current.generate(
        prompt,
        {maxNewTokens: 64, temperature: 0.8},
        {
          onToken: t => setOutput(o => o + t),
          onComplete: s => {
            setStatus('Done: ' + JSON.stringify(s));
            setIsGenerating(false);
          },
          onError: e => {
            setStatus('Error: ' + String(e));
            setIsGenerating(false);
          },
        },
      );
    } catch (e) {
      setStatus('Error: ' + String(e));
      setIsGenerating(false);
    }
  };

  return (
    <SafeAreaView>
      <ScrollView>
        <View style={{padding: 16, gap: 12}}>
          <Text style={{fontFamily: 'Menlo'}}>{status}</Text>
          <Text style={{marginTop: 8}}>Prompt:</Text>
          <TextInput
            value={prompt}
            onChangeText={setPrompt}
            multiline
            editable={isLoaded && !isGenerating}
            style={{borderWidth: 1, borderColor: '#888', padding: 8, minHeight: 100}}
          />
          <Button title={isGenerating ? 'Generating…' : 'Generate'} onPress={onGenerate} disabled={!isLoaded || isGenerating} />
          <Text style={{marginTop: 8}}>Output:</Text>
          <Text style={{fontFamily: 'Menlo'}}>{output}</Text>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}
