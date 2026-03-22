import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import reactPlugin from 'eslint-plugin-react'
import { defineConfig, globalIgnores } from 'eslint/config'

export default defineConfig([
  globalIgnores(['dist']),
  // Node.js files (server, config)
  {
    files: ['server.js'],
    extends: [js.configs.recommended],
    languageOptions: {
      ecmaVersion: 'latest',
      globals: globals.node,
      sourceType: 'module',
    },
  },
  // React app files
  {
    files: ['src/**/*.{js,jsx}'],
    extends: [
      js.configs.recommended,
      reactHooks.configs.flat.recommended,
      reactRefresh.configs.vite,
      reactPlugin.configs.flat.recommended,
      reactPlugin.configs.flat['jsx-runtime'],
    ],
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
      parserOptions: {
        ecmaVersion: 'latest',
        ecmaFeatures: { jsx: true },
        sourceType: 'module',
      },
    },
    rules: {
      'no-unused-vars': ['error', { varsIgnorePattern: '^[A-Z_]' }],
      'react/prop-types': 'off',
      'react/no-unknown-property': ['error', {
        ignore: [
          'args', 'attach', 'array', 'count', 'itemSize', 'object',
          'emissive', 'emissiveIntensity', 'toneMapped', 'transparent',
          'opacity', 'depthWrite', 'blending', 'sizeAttenuation',
          'vertexColors', 'wireframe', 'intensity', 'position', 'castShadow',
          'receiveShadow', 'dispose', 'rotation', 'scale',
          'luminanceThreshold', 'luminanceSmoothing', 'mipmapBlur',
          'blendFunction', 'offset', 'darkness',
        ],
      }],
    },
    settings: {
      react: {
        version: 'detect',
      },
    },
  },
])
