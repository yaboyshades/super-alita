#!/usr/bin/env python3
"""
Quick test script to validate our Python LSP server implementation.
"""
import sys
import os

# Mock the pygls module since we can't install it in this environment
class MockTypes:
    class Diagnostic:
        def __init__(self, range, message, severity=None, source=None):
            self.range = range
            self.message = message
            self.severity = severity
            self.source = source
    
    class Range:
        def __init__(self, start, end):
            self.start = start
            self.end = end
    
    class Position:
        def __init__(self, line, character):
            self.line = line
            self.character = character
    
    class CompletionList:
        def __init__(self, is_incomplete=False, items=None):
            self.is_incomplete = is_incomplete
            self.items = items or []
    
    class CompletionItem:
        def __init__(self, label, kind=None):
            self.label = label
            self.kind = kind
    
    class DiagnosticSeverity:
        Warning = 2
    
    class CompletionItemKind:
        Keyword = 14
        Class = 7
        Struct = 22
        Constant = 21
    
    TEXT_DOCUMENT_DID_OPEN = "textDocument/didOpen"
    TEXT_DOCUMENT_DID_CHANGE = "textDocument/didChange"
    COMPLETION = "textDocument/completion"

class MockLSP:
    types = MockTypes()
    CompletionOptions = lambda trigger_characters=None: None
    CompletionParams = None
    CompletionList = MockTypes.CompletionList
    CompletionItem = MockTypes.CompletionItem
    CompletionItemKind = MockTypes.CompletionItemKind
    TextDocumentSyncKind = None
    DidChangeTextDocumentParams = None
    DidOpenTextDocumentParams = None

class MockDocument:
    def __init__(self, source="", uri="file:///test.alita"):
        self.source = source
        self.uri = uri

class MockWorkspace:
    def get_document(self, uri):
        return MockDocument("EXAMPLE UPPERCASE text")

class MockLanguageServer:
    def __init__(self, name, version):
        self.name = name
        self.version = version
        self.workspace = MockWorkspace()
    
    def feature(self, *args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def show_message_log(self, message):
        print(f"LSP Log: {message}")
    
    def publish_diagnostics(self, uri, diagnostics):
        print(f"LSP Diagnostics for {uri}: {len(diagnostics)} issues")

# Mock the modules properly
import types

# Create pygls module structure
pygls_module = types.ModuleType('pygls')
server_module = types.ModuleType('pygls.server')
lsp_module = types.ModuleType('pygls.lsp')
workspace_module = types.ModuleType('pygls.workspace')

# Add server module to pygls
server_module.LanguageServer = MockLanguageServer
pygls_module.server = server_module

# Add lsp module to pygls  
lsp_module.types = MockTypes()
lsp_module.CompletionOptions = lambda trigger_characters=None: None
lsp_module.CompletionParams = None
lsp_module.CompletionList = MockTypes.CompletionList
lsp_module.CompletionItem = MockTypes.CompletionItem
lsp_module.CompletionItemKind = MockTypes.CompletionItemKind
lsp_module.TextDocumentSyncKind = None
lsp_module.DidChangeTextDocumentParams = None
lsp_module.DidOpenTextDocumentParams = None
pygls_module.lsp = lsp_module

# Add workspace module to pygls
workspace_module.Document = MockDocument
pygls_module.workspace = workspace_module

# Register in sys.modules
sys.modules['pygls'] = pygls_module
sys.modules['pygls.server'] = server_module
sys.modules['pygls.lsp'] = lsp_module
sys.modules['pygls.workspace'] = workspace_module

# Now test our LSP server implementation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python_lsp_server', 'bundled', 'tool'))

try:
    import lsp_server
    print("✓ LSP server module imports successfully")
    
    print(f"✓ Server created: {lsp_server.server.name} v{lsp_server.server.version}")
    
    # Test validation function
    from types import SimpleNamespace
    mock_params = SimpleNamespace()
    mock_params.text_document = SimpleNamespace()
    mock_params.text_document.uri = "file:///test.alita"
    
    lsp_server._validate(lsp_server.server, mock_params)
    print("✓ Validation function works")
    
    # Test completions
    completions = lsp_server.completions(None)
    print(f"✓ Completions function returns {len(completions.items)} items")
    
    print("✓ LSP server implementation looks good!")
    
except Exception as e:
    print(f"✗ Error testing LSP server: {e}")
    import traceback
    traceback.print_exc()