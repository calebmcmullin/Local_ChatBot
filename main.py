from core.completions import completion_no_think
from core.models import load_model

model = load_model("qwen3-4b")

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

    content = completion_no_think(
        model=model,
        message=history,
        max_tokens=4080
    )

    history.append({"role": "assistant", "content": content})