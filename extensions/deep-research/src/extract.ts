/**
 * Extract clean text from HTML content
 */
export function htmlToCleanText(html: string): string {
  if (!html || typeof html !== 'string') {
    return '';
  }

  // Remove script and style elements
  let text = html.replace(/<script[^>]*>[\s\S]*?<\/script>/gi, '');
  text = text.replace(/<style[^>]*>[\s\S]*?<\/style>/gi, '');

  // Remove HTML tags
  text = text.replace(/<[^>]*>/g, ' ');

  // Decode HTML entities
  text = text
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&amp;/g, '&')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&nbsp;/g, ' ')
    .replace(/&#(\d+);/g, (match, dec) => String.fromCharCode(dec))
    .replace(/&#x([a-fA-F0-9]+);/g, (match, hex) => String.fromCharCode(parseInt(hex, 16)));

  // Clean up whitespace
  text = text.replace(/\s+/g, ' ').trim();

  return text;
}/**
 * Extract text content from various formats
 */
export function extractTextContent(content: string, contentType?: string): string {
  if (!content) {
    return '';
  }

  // If it looks like HTML, extract text
  if (content.includes('<') && content.includes('>')) {
    return htmlToCleanText(content);
  }

  // If it's already plain text, just clean it up
  return content.replace(/\s+/g, ' ').trim();
}

