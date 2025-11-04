const path = require('path');
const {getDefaultConfig, mergeConfig} = require('@react-native/metro-config');

/**
 * Metro configuration for the example app
 * - Enables symlinked package resolution for local "file:../.." dependency
 * - Watches the open-source package root so Metro can resolve its sources
 */
const defaultConfig = getDefaultConfig(__dirname);

const config = {
  resolver: {
    ...defaultConfig.resolver,
    unstable_enableSymlinks: true,
    sourceExts: Array.from(new Set([...(defaultConfig.resolver?.sourceExts || []), 'ts', 'tsx'])),
    extraNodeModules: {
      '@babel/runtime': path.resolve(__dirname, 'node_modules/@babel/runtime'),
      'react': path.resolve(__dirname, 'node_modules/react'),
      'react-native': path.resolve(__dirname, 'node_modules/react-native'),
      'react-native-macos': path.resolve(__dirname, 'node_modules/react-native-macos'),
    },
  },
  watchFolders: [
    path.resolve(__dirname, '..', '..'), // <repo>/open-source/react-native-executorch-macos
  ],
};

module.exports = mergeConfig(defaultConfig, config);
