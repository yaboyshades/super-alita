export interface SearchResult {
  title: string;
  url: string;
  snippet?: string;
  engine?: string;
}

export interface FetchedDoc {
  url: string;
  title: string;
  content: string;
  wordCount: number;
}

export interface ResearchConfig {
  provider: 'searxng' | 'perplexica';
  maxArticles: number;
  timeoutMs: number;
  modelVendor?: string;
  modelFamily?: string;
  searxngEndpoint?: string;
  searxngEngines?: string;
  perplexicaBaseUrl?: string;
  perplexicaApiKey?: string;
}

export type ResearchMode = 'web' | 'academic' | 'technical';

export interface SearxngConfig {
  endpoint: string;
  engines?: string;
  timeoutMs: number;
}

export interface PerplexicaConfig {
  baseUrl: string;
  apiKey?: string;
  timeoutMs: number;
}

