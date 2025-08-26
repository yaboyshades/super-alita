from fastapi.testclient import TestClient

from src.main import create_app

app = create_app()
client = TestClient(app)
resp = client.post(
    "/deepcode/request",
    json={
        "task_kind": "generic",
        "requirements": "Add a hello module",
        "repo_path": ".",
    },
)
print("Status:", resp.status_code)
print(resp.text)
