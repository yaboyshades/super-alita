# Copilot Agent Middleware (Prototype)

Tiny JS modules to normalize and amplify prompts before sending to a chat agent. Can be composed in any agent pipeline.

```js
import { optimize, amplify } from 'copilot-agent-middleware';

const cleaned = optimize(userPrompt);
const enhanced = amplify(cleaned, {
  repoName: 'my-repo',
  filePath: '/path/to/file.ts',
  languageId: 'typescript',
  selectionPreview: 'function greet() {\n  return "hi";\n}'
});
```

