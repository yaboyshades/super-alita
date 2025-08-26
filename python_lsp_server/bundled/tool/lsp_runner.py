#
# /python_lsp_server/bundled/tool/lsp_runner.py
#
# Description: Entry point for launching the Python LSP server.
# This script ensures that bundled dependencies are on the path and runs the server module.
#

import pathlib
import runpy
import sys


def main():
    """Configures the path and runs the LSP server module."""
    # Add the current directory (and any bundled libs) to the Python path
    here = pathlib.Path(__file__).parent
    sys.path.insert(0, str(here))
    
    # Run the lsp_server module as the main entry point
    runpy.run_module("lsp_server", run_name="__main__")

if __name__ == "__main__":
    main()