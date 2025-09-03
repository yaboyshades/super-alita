export function optimize(prompt) {
  if (!prompt) return '';
  let s = String(prompt);
  s = s.replace(/[\t ]+/g, ' ').replace(/\s+\n/g, '\n').trim();
  s = s.replace(/\s*([.,;:!?])\s*/g, '$1 ');
  // Ensure concise imperative phrasing
  if (/^(add|create|explain|optimize|rewrite|refactor|fix|summarize|implement)\b/i.test(s)) {
    s = `Please ${s[0].toLowerCase()}${s.slice(1)}`;
  }
  return s.trim();
}

