"""
qwen.py — thin OpenAI-compatible client for the local Qwen endpoint.

Same pattern as stage2_reasoner.py — thinking is enabled, budget is
generous, we prefer `content` over `reasoning_content` so callers get the
actual answer rather than the chain-of-thought.
"""

import json
import urllib.error
import urllib.request


def discover_model(base_url, timeout=5):
    url = base_url.rstrip("/") + "/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        data = payload.get("data") or []
        return data[0].get("id") if data else None
    except Exception:
        return None


def chat(messages, base_url, model, temperature=0.2,
         max_tokens=32000, timeout=3600):
    """
    Free-form chat — NOT JSON-constrained, because the copilot renders
    Markdown from the model. We keep thinking on so explanations are
    well-reasoned.
    """
    endpoint = base_url.rstrip("/") + "/v1/chat/completions"
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    msg = payload["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    reasoning = (msg.get("reasoning_content") or "").strip()
    return content or reasoning
