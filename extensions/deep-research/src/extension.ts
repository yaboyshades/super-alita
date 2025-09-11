/* eslint-disable @typescript-eslint/no-explicit-any */
import * as vscode from 'vscode';
import axios from 'axios';
import { htmlToCleanText } from './extract';
import { searxngSearch } from './providers/searxng';
import { perplexicaSearch } from './providers/perplexica';
import { StackOverflowProvider } from './providers/stackoverflow';
import type { ResearchConfig, ResearchMode, SearchResult, FetchedDoc } from './types';

function getConfig(): ResearchConfig {
  const cfg = vscode.workspace.getConfiguration('deepResearch');
  return {
    provider: (cfg.get('provider') as any) ?? 'searxng',
    maxArticles: (cfg.get('maxArticles') as number) ?? 5,
    timeoutMs: (cfg.get('timeoutMs') as number) ?? 20000,
    modelVendor: cfg.get('model.vendor') as string | undefined,
    modelFamily: cfg.get('model.family') as string | undefined,
    searxngEndpoint: cfg.get('searxng.endpoint') as string | undefined,
    searxngEngines: cfg.get('searxng.engines') as string | undefined,
    perplexicaBaseUrl: cfg.get('perplexica.baseUrl') as string | undefined,
    perplexicaApiKey: cfg.get('perplexica.apiKey') as string | undefined
  };
}

function parseMode(text: string): ResearchMode {
  if (/\/(academic|papers?)\b/i.test(text)) {
    return 'academic';
  }
  if (/\/(tech|technical)\b/i.test(text)) {
    return 'technical';
  }
  return 'web';
}

function sanitizeQuery(text: string): string {
  return text.replace(/\/(academic|papers?|tech|technical|implement(:\w+)?|apply|so)\b/gi, '').trim();
}

function categoriesForMode(mode: ResearchMode): string[] | undefined {
  if (mode === 'academic') {
    return ['science'];
  }
  if (mode === 'technical') {
    return ['it'];
  }
  return undefined;
}

async function fetchAndExtract(url: string, timeoutMs: number): Promise<string> {
  const res = await axios.get(url, {
    timeout: timeoutMs,
    headers: {
      'user-agent': 'Mozilla/5.0 (ResearchBot)'
    }
  });
  const html = res.data ?? '';
  return htmlToCleanText(html);
}

function toFetchedDoc(r: SearchResult, content: string): FetchedDoc {
  const words = content.trim().split(/\s+/);
  return {
    url: r.url,
    title: r.title || r.url,
    content,
    wordCount: words.length
  };
}

function citationsBlock(docs: FetchedDoc[]): string {
  const items = docs.map((d, i) => `[$${i + 1}] ${d.title} — ${d.url}`);
  return items.length ? `\n\nSources\n${items.join('\n')}` : '';
}

async function synthesizeWithLM(
  prompt: string,
  docs: FetchedDoc[],
  mode: ResearchMode,
  vendor?: string,
  family?: string,
  token?: vscode.CancellationToken
): Promise<AsyncIterable<string> | string> {
  try {
    const selector: any = {};
    if (vendor && vendor.trim()) {
      selector.vendor = vendor.trim();
    }
    if (family && family.trim()) {
      selector.family = family.trim();
    }
    const [model] = await (vscode as any).lm.selectChatModels(selector);
    if (!model) {
      return `No language model available for synthesis. Displaying gathered sources only.`;
    }

    const messages = [
      new vscode.LanguageModelChatMessage(
        vscode.LanguageModelChatMessageRole.Assistant,
        [
          'You are a meticulous research assistant. Produce a concise, sourced synthesis.',
          'Requirements:',
          '- Summarize key findings tailored to the request and mode.',
          '- Use inline citations like [1], [2] referring to Sources list provided.',
          '- Highlight consensus, disagreements, and actionable guidance.',
          '- Include brief caveats/limitations. Avoid hallucinations.',
          `Mode: ${mode}`
        ].join('\n')
      ),
      new vscode.LanguageModelChatMessage(
        vscode.LanguageModelChatMessageRole.User,
        [
          'User request:',
          prompt,
          '',
          'Context snippets (truncate beyond 1200 words each):',
          ...docs.map((d, i) => `[#${i + 1}] ${d.title} ${d.url}\n${d.content.slice(0, 7500)}`),
          '',
          'Please produce a structured answer with citations [n].'
        ].join('\n')
      )
    ];

    const response = await model.sendRequest(messages, {}, token);
    return response.text ?? 'Synthesis completed.';
  } catch (err: any) {
    return `Synthesis error: ${err?.message || String(err)}`;
  }
}

function detectImplementKind(text: string): string | undefined {
  const m = text.match(/\/implement(?::([\w_\-]+))?/i);
  return m ? (m[1] || 'text2backend') : undefined;
}

function detectApply(text: string): boolean {
  return /\/(apply)\b/i.test(text);
}

function detectSO(text: string): boolean {
  return /\/(so)\b/i.test(text);
}

function getPythonPath(): string {
  // Simplified version for VS Code extension
  return 'python';
}

async function runDeepcodeCLI(args: string[], token: vscode.CancellationToken): Promise<any> {
  // Simplified implementation - would need VS Code task integration
  return new Promise((resolve) => {
    resolve({ status: 'not_implemented', message: 'DeepCode CLI not available in extension mode' });
  });
}

async function runResearch(
  query: string,
  mode: ResearchMode,
  cfg: ResearchConfig,
  stream: vscode.ChatResponseStream,
  token: vscode.CancellationToken
): Promise<void> {
  stream.progress(`Searching (${mode})…`);

  let results: SearchResult[] = [];
  try {
    if (cfg.provider === 'searxng') {
      results = await searxngSearch(query, {
        endpoint: cfg.searxngEndpoint || 'http://localhost:8080',
        engines: cfg.searxngEngines,
        timeoutMs: cfg.timeoutMs
      }, categoriesForMode(mode), Math.max(5, cfg.maxArticles));
    } else {
      results = await perplexicaSearch(query, {
        baseUrl: cfg.perplexicaBaseUrl || 'http://localhost:3000',
        apiKey: cfg.perplexicaApiKey,
        timeoutMs: cfg.timeoutMs
      }, Math.max(5, cfg.maxArticles));
    }
  } catch (e: any) {
    stream.markdown(`Search error: ${e?.message || String(e)}`);
    return;
  }

  if (!results.length) {
    stream.markdown('No results found. Try refining your query.');
    return;
  }

  stream.progress(`Fetching ${Math.min(results.length, cfg.maxArticles)} pages…`);
  const selected = results.slice(0, cfg.maxArticles);
  const fetched: FetchedDoc[] = [];

  for (const r of selected) {
    if (token.isCancellationRequested) {
      break;
    }
    try {
      const text = await fetchAndExtract(r.url, cfg.timeoutMs);
      if (text && text.length > 200) {
        fetched.push(toFetchedDoc(r, text));
        stream.progress(`Fetched: ${r.title || r.url}`);
      }
    } catch {
      // ignore fetch errors per result
    }
  }

  if (!fetched.length) {
    stream.markdown('Could not fetch page contents. Showing links only:\n' + selected.map((r, i) => `[$${i + 1}] ${r.title} — ${r.url}`).join('\n'));
    return;
  }

  stream.progress('Synthesizing…');
  const synthesis = await synthesizeWithLM(query, fetched, mode, cfg.modelVendor, cfg.modelFamily, token);

  let synthesisText = typeof synthesis === 'string' ? synthesis : '';
  if (typeof synthesis !== 'string') {
    for await (const chunk of synthesis) {
      if (token.isCancellationRequested) {
        break;
      }
      synthesisText += String(chunk);
    }
  }
  stream.markdown(synthesisText + citationsBlock(fetched));

  // Optional DeepCode pipeline: /implement[:task_kind] and /apply
  const implementKind = detectImplementKind(query);
  const doApply = detectApply(query);
  if (implementKind) {
    stream.progress(`Invoking DeepCode (${implementKind})…`);
    try {
      const req = await runDeepcodeCLI(['request', '--task-kind', implementKind, '--requirements', synthesisText], token);
      stream.markdown(`DeepCode request: ${'status' in req ? req.status : 'submitted'}`);
      const latest = await runDeepcodeCLI(['latest'], token);
      const files = (latest?.diffs || latest?.files || []).length || 0;
      stream.markdown(`Proposed changes: ${files} files${doApply ? ' — applying…' : ''}`);
      if (doApply) {
        const applied = await runDeepcodeCLI(['apply'], token);
        stream.markdown(`Apply status: ${applied?.status || 'done'}`);
      }
    } catch (e: any) {
      stream.markdown(`DeepCode error: ${e?.message || String(e)}`);
    }
  }
}

export function activate(context: vscode.ExtensionContext) {
  const participant = vscode.chat.createChatParticipant('research', async (
    request: vscode.ChatRequest,
    chatContext: vscode.ChatContext,
    stream: vscode.ChatResponseStream,
    token: vscode.CancellationToken
  ) => {
    try {
      const raw = (request.prompt || '').toString();
      const mode = parseMode(raw);
      const query = sanitizeQuery(raw);
      const cfg = getConfig();

      // StackOverflow command handler
      if (detectSO(raw)) {
        const so = new StackOverflowProvider();
        const q = query || "custom github copilot mcp implementations fix not working";
        if (!q) {
          stream.markdown('Please provide a query after `/so`, e.g., `@research /so debounce in JavaScript`.');
          return;
        }
        stream.progress(`Fetching Stack Overflow snippets for MCP Copilot issues…`);
        try {
          const results = await so.search(q, Math.max(3, Math.min(6, cfg.maxArticles)));
          if (!results.length) {
            stream.markdown('No snippets found on Stack Overflow.');
            return;
          }
          const md = results.map((r, i) => `### Snippet ${i + 1}: [${r.title}](${r.url})\n\n\`\`\`\n${(r.snippet || '').trim()}\n\`\`\`\n`).join('\n');
          stream.markdown(md);
          return;
        } catch (e: any) {
          stream.markdown(`StackOverflow error: ${e?.message || String(e)}`);
          return;
        }
      }

      await runResearch(query, mode, cfg, stream, token);
    } catch (error: any) {
      stream.markdown(`Extension error: ${error?.message || String(error)}`);
    }
  });

  context.subscriptions.push(participant);

  const disposable = vscode.commands.registerCommand('deepResearch.search', async () => {
    const query = await vscode.window.showInputBox({ prompt: 'Enter research query (prefix /academic or /technical for modes)' });
    if (!query) {
      return;
    }
    const stream = new (class implements vscode.ChatResponseStream {
      progress(message: string): void {
        void vscode.window.setStatusBarMessage(`$(search) ${message}`, 2000);
      }
      markdown(value: string | vscode.MarkdownString): void {
        void vscode.window.showInformationMessage(typeof value === 'string' ? value : value.value);
      }
      anchor(): void { /* noop */ }
      button(): void { /* noop */ }
      filetree(): void { /* noop */ }
      reference(): void { /* noop */ }
      push(): void { /* noop */ }
      setThrottleDelay?(): void { /* noop */ }
    })();
    const tokenSrc = new vscode.CancellationTokenSource();
    const mode = parseMode(query);
    await runResearch(sanitizeQuery(query), mode, getConfig(), stream, tokenSrc.token);
  });
  context.subscriptions.push(disposable);
}

export function deactivate() {
  // noop
}
