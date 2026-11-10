from llama_cpp.llama import Llama

def stream_no_think(model: Llama, message: str) -> str:
    stream = model.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": f"/no_think\n{message}"
            }
        ],
        max_tokens=512,
        stream=True
    )

    content = ""
    thinking_finished = False

    for chunk in stream:
        token = chunk["choices"][0]["delta"].get("content", "")

        if not isinstance(token, str):
            continue

        content += token

        if not thinking_finished:
            if "</think>" not in content:
                continue

            content = content.split("</think>", 1)[1]
            thinking_finished = True

            if content:
                print(content, end="", flush=True)
        else:
            print(token, end="", flush=True)

    print()
    return content.strip()