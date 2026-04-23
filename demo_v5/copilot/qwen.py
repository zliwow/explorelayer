"""
qwen.py — thin OpenAI-compatible client for the local Qwen endpoint.

Returns a (answer, debug) tuple so the caller can surface diagnostics
(finish_reason, raw lengths, whether <think> tags were stripped) when
things come back empty.
"""

import json
import re
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


THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
UNCLOSED_THINK_RE = re.compile(r"<think>.*", flags=re.DOTALL | re.IGNORECASE)


def _strip_think(text):
    """Remove closed <think>...</think>. If an unclosed <think> remains,
    keep only the part BEFORE it (the post-think answer never came)."""
    if not text:
        return ""
    stripped = THINK_RE.sub("", text)
    stripped = UNCLOSED_THINK_RE.sub("", stripped)
    return stripped.strip()


def _one_call(messages, base_url, model, temperature, max_tokens,
              timeout, no_think=False):
    """Make a single API call and return a rich debug dict."""
    msgs = list(messages)
    if no_think and msgs and msgs[0].get("role") == "system":
        msgs[0] = dict(msgs[0])
        msgs[0]["content"] = msgs[0]["content"].rstrip() + "\n\n/no_think"
    body = {
        "model": model,
        "messages": msgs,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    endpoint = base_url.rstrip("/") + "/v1/chat/completions"
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    choice = payload["choices"][0]
    msg = choice["message"]
    raw_content = msg.get("content") or ""
    raw_reasoning = msg.get("reasoning_content") or ""
    finish = choice.get("finish_reason")

    stripped_content = _strip_think(raw_content)
    stripped_reasoning = _strip_think(raw_reasoning)
    answer = stripped_content or stripped_reasoning
    debug = {
        "finish_reason": finish,
        "raw_content_len": len(raw_content),
        "raw_reasoning_len": len(raw_reasoning),
        "stripped_content_len": len(stripped_content),
        "stripped_reasoning_len": len(stripped_reasoning),
        "had_think_tag": "<think>" in raw_content.lower(),
        "no_think_used": no_think,
        "usage": payload.get("usage"),
    }
    return answer, debug, raw_content, raw_reasoning


def chat(messages, base_url, model, temperature=0.2,
         max_tokens=32000, timeout=3600):
    """
    Free-form chat. Returns (answer, debug_dict).

    Strategy:
      1. First try with thinking enabled so answers are well-reasoned.
      2. If the answer is empty (model burned its budget thinking and
         never emitted a closing </think> + answer, or server routed
         everything to reasoning), retry once with /no_think so the
         model goes straight to the answer.
    """
    answer, debug, rc, rr = _one_call(
        messages, base_url, model, temperature, max_tokens, timeout,
        no_think=False,
    )
    debug["attempts"] = [dict(debug)]

    if not answer:
        # Retry without thinking
        answer2, debug2, rc2, rr2 = _one_call(
            messages, base_url, model, temperature, max_tokens, timeout,
            no_think=True,
        )
        debug["attempts"].append(dict(debug2))
        if answer2:
            debug.update(debug2)
            answer = answer2
            rc, rr = rc2, rr2

    if not answer:
        # Surface the raw output so the user can see what the server said
        debug["raw_content_preview"] = (rc or "")[:1500]
        debug["raw_reasoning_preview"] = (rr or "")[:1500]

    return answer, debug
