import type { Response } from "express";

export function initSSE(res: Response) {
  res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
  res.setHeader("Cache-Control", "no-cache, no-transform");
  res.setHeader("Connection", "keep-alive");
  // Helpful for some proxies:
  res.setHeader("X-Accel-Buffering", "no");
}

export function sendSSE(res: Response, type: string | null, data: unknown) {
  if (type) res.write(`event: ${type}\n`);
  res.write(`data: ${JSON.stringify(data)}\n\n`);
}

export function sendDefault(res: Response, chunk: unknown) {
  // Default SSE: no event name — raw data only.
  res.write(`data: ${JSON.stringify(chunk)}\n\n`);
}