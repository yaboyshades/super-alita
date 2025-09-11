# DeepCode: Security Audit

Scope:
- File: {file_path}
- Snippet or area:
```
{code_snippet}
```

Checklist:
- Input validation & sanitization
- Avoid `eval`/`exec` and unsafe deserialization
- Secrets handling and logging hygiene
- AuthZ/AuthN checks
- Dependency risks

Output:
- Findings with CWE-style categories
- Severity per finding (critical/error/warning/info)
- Proof-of-concept or exploit path (if applicable)
- Remediation diffs or patches

