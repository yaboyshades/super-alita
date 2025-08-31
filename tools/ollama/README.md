# Ollama CPU+GPU Split (gpt-oss:20b)

This folder includes a ready-to-run Modelfile for splitting `gpt-oss:20b` across your RTX 3060 12 GB (GPU) and CPU.

## Files
- `Modelfile.gpt-oss-20b-split`: conservative defaults for 12 GB VRAM.
  - `num_ctx 2048`, `num_batch 1` to limit VRAM
  - `num_gpu 12` to offload ~12 layers to GPU
  - `num_thread 12` to utilize CPU cores

## Build and Run
- Create the model:
  - `ollama create gpt-oss-20b-split -f tools/ollama/Modelfile.gpt-oss-20b-split`
- Run it:
  - `ollama run gpt-oss-20b-split`

## Tune for Best Throughput
- Use the tuner (already in this repo):
  - `python tools/ollama_gpu_cpu_tuner.py --model gpt-oss-20b-split --layers 8-16 --num-thread 12 --prompt "Hello!"`
- If you see errors or OOM:
  - Reduce `num_gpu` by 2–4
  - Keep `num_ctx` at 2048 and `num_batch` at 1; increase later once stable

## One-shot via API (no Modelfile rebuild)
```
curl http://localhost:11434/api/generate -s -d '{
  "model":"gpt-oss:20b",
  "prompt":"Hello!",
  "options":{"num_gpu":12, "num_thread":12, "num_ctx":2048, "num_batch":1}
}'
```

## Tips
- Monitor VRAM with `nvidia-smi` while generating; adjust `gpu_layers` to avoid OOM.
- Raise `num_gpu` gradually for more GPU work; raise `num_ctx` or `num_batch` only after it’s stable.
- If CPU is pegged and GPU idle → increase `num_gpu`.
- If GPU is pegged but VRAM underused → try higher `num_gpu` or reduce `num_thread` slightly to lower CPU contention.
