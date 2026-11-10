from llama_cpp import Llama

qwen3_14b = Llama.from_pretrained(
    repo_id="Qwen/Qwen3-14B-GGUF",
    filename="Qwen3-14B-Q4_K_M.gguf",
    n_ctx=2048,
    n_gpu_layers=25,
    n_threads=8,
    verbose=False,
)
