export function amplify(prompt, context = {}) {
  const repo = context.repoName || context.repo || '';
  const file = context.filePath || context.file || '';
  const lang = context.languageId || context.lang || '';
  const snippet = context.selectionPreview || context.snippet || '';

  const header = [
    repo && `Repository: ${repo}`,
    file && `File: ${file}`,
    lang && `Language: ${lang}`,
    snippet && `Context snippet:\n\u0060\u0060\u0060\n${snippet}\n\u0060\u0060\u0060`
  ].filter(Boolean).join('\n');

  const role = 'You are a precise, helpful coding assistant.';
  return [role, header, prompt].filter(Boolean).join('\n\n').trim();
}

