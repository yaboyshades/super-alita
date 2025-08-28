/** @module Interface vscode:example/host-api **/
export function getFileInfo(path: string): FileInfo;
export function readFileSnippet(path: string, startLine: number, endLine: number): string;
export function emitDiagnostic(path: string, line: number, message: string): void;
export interface FileInfo {
  path: string,
  size: number,
  modified: bigint,
}
