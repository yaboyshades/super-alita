import sys
from pathlib import Path

from fastapi.testclient import TestClient

# Ensure project root is on sys.path for 'src' package and local modules
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.main import create_app

app = create_app()
client = TestClient(app)

payload = {
    "user_input": (
        "Add SDD pipeline with constitutional validation gates and Mangle "
        "reasoning integration for planning and tasks."
    ),
    "context": {"priority": "high", "owner": "platform"},
    "constitutional_gates": True,
}

resp = client.post("/sdd/specify", json=payload)
print("status:", resp.status_code)
try:
    data = resp.json()
    print("keys:", list(data.keys()))
    print("success:", data.get("success"))
    print("feature_id:", data.get("feature_id"))
    print("overall_compliance_score:", data.get("overall_compliance_score"))
    print("compliance_threshold_met:", data.get("compliance_threshold_met"))
    # Optional: show a snippet of the spec
    spec = data.get("specification") or ""
    print("spec snippet:", spec[:200].replace("\n", " "))
except Exception:
    print("raw:", resp.text[:500])
