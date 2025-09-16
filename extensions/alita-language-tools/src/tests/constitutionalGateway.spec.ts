import { describe, expect, it, vi } from 'vitest';

vi.mock('vscode', () => {
  class Range {
    start: { line: number; character: number };
    end: { line: number; character: number };

    constructor(
      startLine: number,
      startCharacter: number,
      endLine: number,
      endCharacter: number
    ) {
      this.start = { line: startLine, character: startCharacter };
      this.end = { line: endLine, character: endCharacter };
    }
  }

  class Diagnostic {
    source?: string;
    code?: string;

    constructor(
      public range: Range,
      public message: string,
      public severity: number
    ) {}
  }

  const DiagnosticSeverity = {
    Error: 0,
    Warning: 1,
    Information: 2,
  } as const;

  return {
    Range,
    Diagnostic,
    DiagnosticSeverity,
  };
});

import * as vscode from 'vscode';

import { toVscodeDiagnostics } from '../features/constitutionalGateway';

describe('toVscodeDiagnostics', () => {
  it('returns empty array when input is not an array', () => {
    expect(toVscodeDiagnostics(undefined)).toEqual([]);
    expect(toVscodeDiagnostics('not-an-array')).toEqual([]);
  });

  it('converts diagnostics to vscode diagnostics', () => {
    const diagnostics = toVscodeDiagnostics([
      {
        range: {
          start: { line: 1, character: 2 },
          end: { line: 1, character: 4 },
        },
        severity: 1,
        message: 'First issue',
        source: 'constitutional-security',
        code: 'SEC001',
        type: 'security',
      },
      {
        range: {
          start: { line: 2, character: 0 },
          end: { line: 2, character: 5 },
        },
        severity: 3,
        message: 'Second issue',
        source: 'constitutional-style',
        code: 'STYLE001',
        type: 'style',
      },
    ]);

    expect(diagnostics).toHaveLength(2);
    expect(diagnostics[0].message).toBe('First issue');
    expect(diagnostics[0].severity).toBe(vscode.DiagnosticSeverity.Error);
    expect(diagnostics[0].source).toBe('constitutional-security');
    expect(diagnostics[0].code).toBe('SEC001');
    expect(diagnostics[1].severity).toBe(vscode.DiagnosticSeverity.Information);
  });
});
