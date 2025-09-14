#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import queue
import re
import signal
import threading
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

# ---------------- Config (tune for RTX 3060: 12GB) ----------------
MODEL_NAME = os.getenv("MEM_MODEL", "all-MiniLM-L6-v2")  # 384-d
DIM = 384
N_LIST = int(os.getenv("MEM_NLIST", "1024"))              # IVF cells
PQ_M = int(os.getenv("MEM_PQ_M", "32"))                   # PQ subvectors
PQ_BITS = int(os.getenv("MEM_PQ_BITS", "8"))
TRAIN_THRESHOLD = int(
    os.getenv("MEM_TRAIN_THRESHOLD", "4096")
)  # vectors to train
BATCH_SIZE = int(os.getenv("MEM_BATCH_SIZE", "128"))
CHUNK_CHARS = int(os.getenv("MEM_CHUNK_CHARS", "600"))
CHUNK_OVERLAP = int(os.getenv("MEM_CHUNK_OVERLAP", "120"))
WATCH_GLOBS = tuple(os.getenv("MEM_GLOBS", ".py,.md,.yaml,.yml").split(","))
SAVE_EVERY = int(
    os.getenv("MEM_SAVE_EVERY", "5000")
)  # save index every N adds
DATA_DIR = Path(os.getenv("MEM_DATA_DIR", "memory/data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH = DATA_DIR / "index.faiss"
SNIPPETS_PATH = DATA_DIR / "snippets.jsonl"
STATE_PATH = DATA_DIR / "state.json"

# ---------------- Globals ----------------
MODEL = SentenceTransformer(
    MODEL_NAME, device="cuda" if torch.cuda.is_available() else "cpu"
)
_FAISS_INDEX_CPU_TO_GPU = getattr(faiss, "index_cpu_to_gpu", None)
_FAISS_INDEX_GPU_TO_CPU = getattr(faiss, "index_gpu_to_cpu", None)
_FAISS_StandardGpuResources = getattr(faiss, "StandardGpuResources", None)
HAS_FAISS_GPU = bool(
    torch.cuda.is_available()
    and _FAISS_INDEX_CPU_TO_GPU
    and _FAISS_StandardGpuResources
)
GPU_RES = (
    _FAISS_StandardGpuResources()
    if HAS_FAISS_GPU and callable(_FAISS_StandardGpuResources)
    else None
)
INDEX: Any = None
TRAIN_BUF: list[np.ndarray] = []
ADDED = 0
SNIPPETS: list[dict[str, Any]] = []  # [{id, path, off, text}]
LOCK = threading.Lock()
Q: queue.Queue[tuple[str, str]] = queue.Queue()

# ---------------- Helpers ----------------


def cosine_embed(texts: list[str]) -> np.ndarray:
    emb = MODEL.encode(
        texts,
        batch_size=BATCH_SIZE,
        convert_to_tensor=True,
        show_progress_bar=False,
    )
    emb = emb.detach().cpu().numpy().astype("float32")
    faiss.normalize_L2(emb)
    return emb


def build_index_cpu() -> Any:
    quant = faiss.IndexFlatIP(
        DIM
    )  # inner-product with normalized vectors == cosine
    index = faiss.IndexIVFPQ(quant, DIM, N_LIST, PQ_M, PQ_BITS)
    index.metric_type = faiss.METRIC_INNER_PRODUCT
    return index


def to_gpu(cpu_index: Any) -> Any:
    if HAS_FAISS_GPU and GPU_RES is not None and _FAISS_INDEX_CPU_TO_GPU:
        return _FAISS_INDEX_CPU_TO_GPU(  # type: ignore[misc]
            GPU_RES, 0, cpu_index
        )
    return cpu_index


def to_cpu(idx: Any) -> Any:
    # If GPU index, convert to CPU; else it's already CPU
    if HAS_FAISS_GPU and _FAISS_INDEX_GPU_TO_CPU and hasattr(idx, "getDevice"):
        try:
            return _FAISS_INDEX_GPU_TO_CPU(idx)  # type: ignore[misc]
        except Exception:
            return idx
    return idx


def get_cpu_index() -> Any | None:
    if INDEX is None:
        return None
    try:
        return to_cpu(INDEX)
    except Exception:
        return INDEX


def save_all():
    with LOCK:
        if INDEX is None:
            return
        cpu = get_cpu_index()
        if cpu is None:
            return
        faiss.write_index(cpu, str(INDEX_PATH))
        with open(SNIPPETS_PATH, "w", encoding="utf-8") as f:
            for rec in SNIPPETS:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        state = {
            "added": ADDED,
            "dim": DIM,
            "nlist": N_LIST,
            "pq_m": PQ_M,
            "pq_bits": PQ_BITS,
            "model": MODEL_NAME,
        }
        STATE_PATH.write_text(json.dumps(state, indent=2))
    print(f"[mem] Saved index: {INDEX_PATH}, snippets: {SNIPPETS_PATH}")


def load_all():
    global INDEX, SNIPPETS, ADDED
    if INDEX_PATH.exists():
        cpu = faiss.read_index(str(INDEX_PATH))
        cpu.metric_type = faiss.METRIC_INNER_PRODUCT
        INDEX = to_gpu(cpu)
        print(f"[mem] Loaded index: {INDEX_PATH}")
    else:
        INDEX = to_gpu(build_index_cpu())
        print("[mem] New blank index (untrained)")

    SNIPPETS = []
    if SNIPPETS_PATH.exists():
        with open(SNIPPETS_PATH, encoding="utf-8") as f:
            SNIPPETS.extend(json.loads(ln) for ln in f)
    ADDED = len(SNIPPETS)
    print(f"[mem] Loaded {ADDED} snippets")


def chunk_text(text: str) -> list[str]:
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i + CHUNK_CHARS]
        if len(chunk) < 50:
            break
        chunks.append(chunk)
        i += max(1, CHUNK_CHARS - CHUNK_OVERLAP)
    return chunks or ([text] if len(text) > 50 else [])


class ChangeHandler(FileSystemEventHandler):
    def __init__(self, q: queue.Queue[tuple[str, str]]):
        self.q = q

    def on_modified(self, event):
        if event.is_directory:
            return
        path_str = str(event.src_path)
        if not any(path_str.endswith(g) for g in WATCH_GLOBS):
            return
        try:
            txt = Path(path_str).read_text(
                encoding="utf-8", errors="ignore"
            )
        except OSError:
            return
        self.q.put((path_str, txt))


def ensure_trained(vecs: np.ndarray):
    """Accumulate vectors until threshold, then train IVF-PQ."""
    global INDEX, TRAIN_BUF
    with LOCK:
        cpu = get_cpu_index()
        # already trained
        if cpu is not None and getattr(cpu, "is_trained", False):
            INDEX = to_gpu(cpu)
            return
        TRAIN_BUF.append(vecs)
        total = sum(arr.shape[0] for arr in TRAIN_BUF)
        if total >= TRAIN_THRESHOLD:
            train = np.vstack(TRAIN_BUF)
            print(f"[mem] Training IVF-PQ on {train.shape[0]} vectors...")
            if cpu is not None and hasattr(cpu, "train"):
                cpu.train(train)
            INDEX = to_gpu(cpu)
            TRAIN_BUF = []


def add_embeddings(emb: np.ndarray, recs: list[dict[str, Any]]):
    global INDEX, ADDED, SNIPPETS
    with LOCK:
        cpu = get_cpu_index()
        if cpu is None or not getattr(cpu, "is_trained", False):
            raise RuntimeError("Index not trained yet; cannot add")
        local_index = to_gpu(cpu)
        local_index.add(emb)
        # sync global
        global INDEX
        INDEX = local_index
        SNIPPETS.extend(recs)
        ADDED += emb.shape[0]
        if ADDED % SAVE_EVERY == 0:
            save_all()


def worker_loop():
    """Batches changes → chunks → embeddings → train/add."""
    batch_texts: list[str] = []
    batch_meta: list[dict[str, Any]] = []
    while True:
        path, text = Q.get()
        try:
            chunks = chunk_text(text)
            if not chunks:
                continue
            batch_texts.extend(chunks)
            for idx, ch in enumerate(chunks):
                batch_meta.append(
                    {
                        "id": len(SNIPPETS) + len(batch_meta),
                        "path": path,
                        "off": idx,
                        "text": ch,
                    }
                )
            if len(batch_texts) >= BATCH_SIZE:
                emb = cosine_embed(batch_texts)
                cpu = get_cpu_index()
                if cpu is None or not getattr(cpu, "is_trained", False):
                    ensure_trained(emb)
                else:
                    add_embeddings(emb, batch_meta)
                batch_texts, batch_meta = [], []
        finally:
            Q.task_done()


# ---------------- FastAPI for search & control ----------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    cpu = get_cpu_index()
    return {
        "ok": True,
        "trained": bool(cpu and cpu.is_trained),
        "added": ADDED,
        "snippets": len(SNIPPETS),
    }


@app.post("/ingest")
def ingest(payload: dict[str, Any]):
    """Optional: push arbitrary text records {path, text}."""
    path = payload.get("path", "mem://push")
    txt = payload.get("text", "")
    Q.put((path, txt))
    return {"queued": True}


@app.post("/search")
def search(payload: dict[str, Any]):
    q = payload.get("q", "").strip()
    k = int(payload.get("k", 3))
    if not q:
        return {"hits": []}
    emb = cosine_embed([q])
    cpu = get_cpu_index()
    if not cpu or not cpu.is_trained or ADDED == 0:
        return {"hits": []}
    assert INDEX is not None
    D, indices = INDEX.search(emb, k)
    idxs = [idx for idx in indices[0].tolist() if idx != -1]
    hits = []
    for idx, score in zip(idxs, D[0][: len(idxs)].tolist(), strict=True):
        if 0 <= idx < len(SNIPPETS):
            rec = SNIPPETS[idx]
            hits.append(
                {
                    "path": rec["path"],
                    "off": rec["off"],
                    "text": rec["text"],
                    "score": float(score),
                }
            )
    return {"hits": hits}


def start_watch_thread():
    handler = ChangeHandler(Q)
    obs = Observer()
    obs.schedule(handler, ".", recursive=True)
    obs.daemon = True
    obs.start()


def start_worker_thread():
    t = threading.Thread(target=worker_loop, daemon=True)
    t.start()


def main():
    load_all()
    start_watch_thread()
    start_worker_thread()

    def _sig(*_):
        save_all()
        os._exit(0)

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="warning")


if __name__ == "__main__":
    main()
