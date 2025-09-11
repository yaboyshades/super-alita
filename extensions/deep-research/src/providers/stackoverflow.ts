import axios from 'axios';
import * as vscode from 'vscode';
import type { SearchResult } from '../types';

function decodeHtml(html: string): string {
  return html
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&amp;/g, '&')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'");
}

export class StackOverflowProvider {
  public readonly name = 'StackOverflow';
  private apiKey: string;

  constructor() {
    const config = vscode.workspace.getConfiguration('deepResearch');
    this.apiKey = (config.get<string>('stackOverflow.apiKey') || '').trim();
  }

  async search(query: string, count = 5): Promise<SearchResult[]> {
    try {
      const encoded = encodeURIComponent(query);
      const keyParam = this.apiKey ? `&key=${encodeURIComponent(this.apiKey)}` : '';
      const base = `https://api.stackexchange.com/2.3`;
      const url = `${base}/search/advanced?order=desc&sort=relevance&q=${encoded}&site=stackoverflow&accepted=True&filter=default${keyParam}`;

      const resp = await axios.get(url, {
        headers: { 'accept': 'application/json' },
        timeout: 10000
      });

      const items = (resp.data?.items || []).slice(0, count);

      const results: SearchResult[] = [];
      for (const item of items) {
        const qid = item.question_id;
        let snippet = '';
        try {
          const ansUrl = `${base}/questions/${qid}/answers?order=desc&sort=votes&site=stackoverflow&filter=withbody${keyParam}`;
          const ans = await axios.get(ansUrl, {
            headers: { 'accept': 'application/json' },
            timeout: 5000
          });
          const ansItems = ans.data?.items || [];
          if (ansItems.length) {
            const body: string = ansItems[0].body || '';
            const match = body.match(/<code>([\s\S]*?)<\/code>/i);
            snippet = match ? decodeHtml(match[1]) : '';
          }
        } catch {
          // ignore answer fetch errors
        }

        results.push({
          title: item.title,
          url: item.link,
          snippet,
          engine: 'StackOverflow'
        });
      }
      return results;
    } catch (error: any) {
      // Use vscode output channel instead of console
      const channel = vscode.window.createOutputChannel('Deep Research');
      channel.appendLine(`Stack Overflow search error: ${error?.message || String(error)}`);
      return [];
    }
  }
}

