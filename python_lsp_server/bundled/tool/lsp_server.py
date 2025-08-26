#
# /python_lsp_server/bundled/tool/lsp_server.py
#
# Description: A Python-based Language Server using the `pygls` library.
# This provides an alternative to the Node.js server for integrating Python-based tools.
#

from pygls.server import LanguageServer
from pygls.lsp import (
    types as lsp,
    CompletionOptions,
    CompletionParams,
    CompletionList,
    CompletionItem,
    CompletionItemKind,
    TextDocumentSyncKind,
    DidChangeTextDocumentParams,
    DidOpenTextDocumentParams
)
from pygls.workspace import Document

class AlitaPythonLspServer(LanguageServer):
    """
    A pygls-based Language Server for the Alita language.
    Provides diagnostics and completions.
    """
    CMD_RESTART = "alita.restart"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

server = AlitaPythonLspServer('alita-lsp-server', 'v0.1')

def _validate(ls: AlitaPythonLspServer, params):
    """Validates the document and publishes diagnostics."""
    text_doc = ls.workspace.get_document(params.text_document.uri)
    source = text_doc.source
    diagnostics = []
    
    # Example: Find all-caps words as warnings
    lines = source.splitlines()
    for line_num, line in enumerate(lines):
        for word in line.split():
            if word.isupper() and len(word) > 1:
                start_char = line.find(word)
                end_char = start_char + len(word)
                d = lsp.Diagnostic(
                    range=lsp.Range(
                        start=lsp.Position(line=line_num, character=start_char),
                        end=lsp.Position(line=line_num, character=end_char)
                    ),
                    message=f'"{word}" is all uppercase.',
                    severity=lsp.DiagnosticSeverity.Warning,
                    source='Alita Linter'
                )
                diagnostics.append(d)

    ls.publish_diagnostics(text_doc.uri, diagnostics)

@server.feature(lsp.TEXT_DOCUMENT_DID_OPEN)
async def did_open(ls: AlitaPythonLspServer, params: DidOpenTextDocumentParams):
    """Text document did open notification."""
    ls.show_message_log('Alita document opened.')
    _validate(ls, params)

@server.feature(lsp.TEXT_DOCUMENT_DID_CHANGE)
def did_change(ls: AlitaPythonLspServer, params: DidChangeTextDocumentParams):
    """Text document did change notification."""
    _validate(ls, params)

@server.feature(
    lsp.COMPLETION,
    CompletionOptions(trigger_characters=[',', ' '])
)
def completions(params: CompletionParams) -> CompletionList:
    """Returns completion items."""
    return CompletionList(
        is_incomplete=False,
        items=[
            CompletionItem(label='ability', kind=CompletionItemKind.Keyword),
            CompletionItem(label='atom', kind=CompletionItemKind.Class),
            CompletionItem(label='bond', kind=CompletionItemKind.Struct),
            CompletionItem(label='CAUSES', kind=CompletionItemKind.Constant),
            CompletionItem(label='SUPPORTS', kind=CompletionItemKind.Constant),
        ]
    )