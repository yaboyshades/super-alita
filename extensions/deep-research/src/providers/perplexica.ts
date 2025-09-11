import axios from 'axios';
import type { SearchResult, PerplexicaConfig } from '../types';

export async function perplexicaSearch(
  query: string,
  config: PerplexicaConfig,
  maxResults = 5
): Promise<SearchResult[]> {
  try {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json'
    };

    if (config.apiKey) {
      headers['Authorization'] = `Bearer ${config.apiKey}`;
    }

    const response = await axios.post(
      `${config.baseUrl}/api/search`,
      {
        query,
        mode: 'web',
        limit: maxResults
      },
      {
        headers,
        timeout: config.timeoutMs
      }
    );

    const results = response.data?.results || response.data || [];

    return results.slice(0, maxResults).map((result: any) => ({
      title: result.title || result.url,
      url: result.url || result.link,
      snippet: result.content || result.snippet || '',
      engine: 'perplexica'
    }));

  } catch (error: any) {
    throw new Error(`Perplexica search failed: ${error?.message || String(error)}`);
  }
}

