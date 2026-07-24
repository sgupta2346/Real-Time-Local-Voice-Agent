import json

import requests

from config import OLLAMA_HOST, LLM_MODEL, LLM_SYSTEM_PROMPT


def reply(user_text: str) -> str:
    resp = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
            "stream": False,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def reply_stream(user_text: str):
    """Yield reply text as it's generated, instead of waiting for the full response."""
    resp = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
            "stream": True,
        },
        timeout=60,
        stream=True,
    )
    resp.raise_for_status()
    for line in resp.iter_lines():
        if not line:
            continue
        chunk = json.loads(line)
        delta = chunk.get("message", {}).get("content", "")
        if delta:
            yield delta
        if chunk.get("done"):
            break
