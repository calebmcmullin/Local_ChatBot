from core.completions import stream_no_think
from core.models import qwen3_14b

history = [
    {"role": "system", "content": "You are a helpful assistant. /no_think"}
]

while True:
    message = input("> ").strip()

    if message.lower() in {"exit", "quit"}:
        break

    if not message:
        continue

    history.append({"role": "user", "content": message})

    content = stream_no_think(qwen3_14b, history)

    history.append({"role": "assistant", "content": content})