"""
Fake servers for testing plugin integrations.
"""

from typing import Any, Dict

from aiohttp import web


class FakePuterServer:
    """Fake Puter server for testing."""

    def __init__(self) -> None:
        self.files: Dict[str, str] = {
            "/test/file.txt": "test file content",
            "/test/readme.md": "# Test README\nThis is a test file.",
        }
        self.directories: Dict[str, list[str]] = {
            "/": ["test"],
            "/test": ["file.txt", "readme.md"],
        }

    def create_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/api/health", self.health_check)
        app.router.add_get("/api/fs/read", self.read_file)
        app.router.add_post("/api/fs/write", self.write_file)
        app.router.add_get("/api/fs/list", self.list_directory)
        app.router.add_delete("/api/fs/delete", self.delete_file)
        app.router.add_post("/api/fs/mkdir", self.create_directory)
        app.router.add_get("/api/fs/stat", self.get_file_stat)
        app.router.add_post("/api/exec", self.execute_command)
        return app

    async def health_check(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def read_file(self, request: web.Request) -> web.Response:
        path = request.query.get("path")
        if not path or path not in self.files:
            return web.json_response({"error": "File not found"}, status=404)
        return web.json_response({"content": self.files[path]})

    async def write_file(self, request: web.Request) -> web.Response:
        data = await request.json()
        path = data.get("path")
        content = data.get("content", "")
        if not path:
            return web.json_response({"error": "Path required"}, status=400)
        self.files[path] = content
        return web.json_response({"success": True})

    async def list_directory(self, request: web.Request) -> web.Response:
        path = request.query.get("path", "/")
        if path not in self.directories:
            return web.json_response({"error": "Directory not found"}, status=404)
        items = []
        for item in self.directories[path]:
            full = f"{path.rstrip('/')}/{item}"
            items.append(
                {
                    "name": item,
                    "type": "file" if full in self.files else "directory",
                    "size": len(self.files.get(full, "")),
                }
            )
        return web.json_response({"items": items})

    async def delete_file(self, request: web.Request) -> web.Response:
        path = request.query.get("path")
        if path in self.files:
            del self.files[path]
        return web.json_response({"success": True})

    async def create_directory(self, request: web.Request) -> web.Response:
        data = await request.json()
        path = data.get("path")
        if path:
            self.directories[path] = []
        return web.json_response({"success": True})

    async def get_file_stat(self, request: web.Request) -> web.Response:
        path = request.query.get("path")
        if path in self.files:
            return web.json_response(
                {
                    "type": "file",
                    "size": len(self.files[path]),
                    "modified": "2024-01-01T00:00:00Z",
                }
            )
        if path in self.directories:
            return web.json_response(
                {
                    "type": "directory",
                    "size": 0,
                    "modified": "2024-01-01T00:00:00Z",
                }
            )
        return web.json_response({"error": "Path not found"}, status=404)

    async def execute_command(self, request: web.Request) -> web.Response:
        data = await request.json()
        command = data.get("command")
        args = data.get("args", [])
        if command == "echo":
            stdout = " ".join(args) + "\n"
            return web.json_response(
                {
                    "stdout": stdout,
                    "stderr": "",
                    "exit_code": 0,
                    "execution_time": 0.1,
                }
            )
        if command == "ls":
            cwd = data.get("cwd", "/")
            if cwd in self.directories:
                stdout = "\n".join(self.directories[cwd]) + "\n"
            else:
                stdout = ""
            return web.json_response(
                {
                    "stdout": stdout,
                    "stderr": "",
                    "exit_code": 0,
                    "execution_time": 0.05,
                }
            )
        return web.json_response(
            {
                "stdout": "",
                "stderr": f"Command not found: {command}\n",
                "exit_code": 127,
                "execution_time": 0.01,
            }
        )
