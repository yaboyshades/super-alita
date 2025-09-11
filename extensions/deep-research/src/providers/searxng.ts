import axios from 'axios';
import type { SearchResult, SearxngConfig } from '../types';

export async function searxngSearch(
  query: string,
  config: SearxngConfig,
  categories?: string[],
  maxResults = 5
): Promise<SearchResult[]> {
  try {
    let url = `${config.endpoint}/search?q=${encodeURIComponent(query)}&format=json`;

    if (categories && categories.length > 0) {
      url += `&categories=${encodeURIComponent(categories.join(','))}`;
    }

    if (config.engines) {
      url += `&engines=${encodeURIComponent(config.engines)}`;
    }

    const response = await axios.get(url, {
      timeout: config.timeoutMs,
      headers: {
        'User-Agent': 'Mozilla/5.0 (DeepResearch/1.0)'
      }
    });

    const results = response.data?.results || [];

    return results.slice(0, maxResults).map((result: any) => ({
      title: result.title || result.url,
      url: result.url,
      snippet: result.content || '',
      engine: 'searxng'
    }));

  } catch (error: any) {
    throw new Error(`SearXNG search failed: ${error?.message || String(error)}`);
  }
}

