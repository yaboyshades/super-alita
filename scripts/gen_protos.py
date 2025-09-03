"""Generate gRPC Python stubs for consensus service.
Run: python scripts/gen_protos.py
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PROTO_DIR = ROOT / "src" / "protos"
OUT_DIR = ROOT / "src" / "protos"


def main() -> int:
    proto = PROTO_DIR / "consensus.proto"
    if not proto.exists():
        print(f"Proto file not found: {proto}")
        return 1
    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"--proto_path={PROTO_DIR}",
        f"--python_out={OUT_DIR}",
        f"--grpc_python_out={OUT_DIR}",
        str(proto),
    ]
    print("Generating stubs:", " ".join(cmd))
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:  # pragma: no cover
        print("Stub generation failed", e)
        return e.returncode
    print("Stubs generated.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
