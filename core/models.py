from llama_cpp import Llama
from pathlib import Path
import json
import psutil

N_THREADS = psutil.cpu_count(logical=False) or 4
N_CTX = 8192

models = json.loads(
    (Path(__file__).resolve().parent / "models.json").read_text(
        encoding="utf-8"
    )
)

def load_model(name: str, n_ctx: int=N_CTX, n_gpu_layers: int=-1, n_threads: int=N_CTX) -> Llama:
    return Llama.from_pretrained(
        repo_id=models[name]["repo_id"],
        filename=models[name]["filename"],
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        n_threads=n_threads,
        verbose=False,
    )