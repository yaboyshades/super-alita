module.exports = {
  root: true,
  env: {
    node: true,
    es2022: true,
  },
  extends: [
    'plugin:@typescript-eslint/recommended',
    'plugin:@typescript-eslint/recommended-requiring-type-checking',
    'prettier',
  ],
  parser: '@typescript-eslint/parser',
  parserOptions: {
    ecmaVersion: 2022,
    sourceType: 'module',
    project: './tsconfig.json',
    tsconfigRootDir: __dirname,
  },
  plugins: ['@typescript-eslint', 'import'],
  rules: {
    // Pragmatic TypeScript rules for VS Code extension development
    '@typescript-eslint/no-explicit-any': 'off',
    '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
    '@typescript-eslint/explicit-function-return-type': 'off',
    '@typescript-eslint/explicit-module-boundary-types': 'off',
    '@typescript-eslint/no-unsafe-assignment': 'off',
    '@typescript-eslint/no-unsafe-call': 'off',
    '@typescript-eslint/no-unsafe-member-access': 'off',
    '@typescript-eslint/no-unsafe-return': 'off',
    '@typescript-eslint/no-unsafe-argument': 'off',
    '@typescript-eslint/prefer-readonly': 'off',
    '@typescript-eslint/prefer-nullish-coalescing': 'off',
    '@typescript-eslint/prefer-optional-chain': 'off',
    '@typescript-eslint/no-unnecessary-type-assertion': 'off',
    '@typescript-eslint/no-floating-promises': ['error', { ignoreVoid: true }],
    '@typescript-eslint/await-thenable': 'error',
    '@typescript-eslint/require-await': 'off',
    '@typescript-eslint/restrict-template-expressions': 'off',
    '@typescript-eslint/unbound-method': 'off',

    // Import rules (relaxed to avoid churn)
    'import/order': 'off',
    'import/no-duplicates': 'error',
    'import/no-unresolved': 'error',

    // Code quality rules (relaxed pragmatically)
    'no-console': 'off',
    'no-debugger': 'error',
    'no-eval': 'error',
    'no-implied-eval': 'error',
    'no-new-func': 'error',
    'no-script-url': 'error',
    'no-multi-str': 'error',
    'no-return-assign': 'error',
    'no-sequences': 'error',
    'no-throw-literal': 'error',
    'no-void': ['error', { allowAsStatement: true }],
    'prefer-promise-reject-errors': 'error',
    'prefer-template': 'off',

    // Best practices (curly only for multi-line to reduce noise)
    'eqeqeq': ['error', 'always'],
    'curly': ['error', 'multi-line'],
    'no-var': 'error',
    'prefer-const': 'error',
    'prefer-arrow-callback': 'error',
    'prefer-template': 'error',
    'object-shorthand': 'error',

    // Performance
    'no-extend-native': 'error',
    'no-global-assign': 'error',
    'no-implicit-globals': 'error',
    'no-loop-func': 'error',

    // Security
    'no-buffer-constructor': 'error',
    'no-new-require': 'error',
    'no-path-concat': 'error',
  },
  overrides: [
    {
      files: ['src/worker.ts'],
      rules: {
        // Worker has special requirements
        '@typescript-eslint/no-explicit-any': 'off',
        'no-console': 'off', // Worker needs console for debugging
      },
    },
    {
      files: ['**/*.test.ts', '**/*.spec.ts'],
      env: {
        jest: true,
        mocha: true,
      },
      rules: {
        // Test files can be lenient
        '@typescript-eslint/no-explicit-any': 'off',
        '@typescript-eslint/no-unsafe-assignment': 'off',
        '@typescript-eslint/no-unsafe-member-access': 'off',
        '@typescript-eslint/no-unsafe-call': 'off',
        '@typescript-eslint/no-unsafe-return': 'off',
        '@typescript-eslint/no-unsafe-argument': 'off',
        '@typescript-eslint/require-await': 'off',
        '@typescript-eslint/explicit-function-return-type': 'off',
        'import/order': 'off',
        'no-console': 'off',
        'prefer-template': 'off',
      },
    },
    {
      files: ['scripts/**/*.js', 'scripts/**/*.cjs'],
      env: {
        node: true,
      },
      rules: {
        // Scripts can use CommonJS and be more flexible
        '@typescript-eslint/no-var-requires': 'off',
        '@typescript-eslint/explicit-function-return-type': 'off',
      },
    },
  ],
  settings: {
    'import/resolver': {
      typescript: {
        alwaysTryTypes: true,
        project: './tsconfig.json',
      },
    },
  },
  ignorePatterns: [
    'out/**',
    'node_modules/**',
    'dist/**',
    '*.js',
    '!scripts/**/*.js',
    'src/generated/**',
  ],
};
