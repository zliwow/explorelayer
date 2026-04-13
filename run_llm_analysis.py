#!/usr/bin/env python3
"""
Send the overlap report to a local vLLM server for risk assessment.

Usage:
    python run_llm_analysis.py prompt.txt --output analysis.md
    python run_llm_analysis.py prompt.txt --url http://localhost:8000
    python run_llm_analysis.py prompt.txt --model Qwen/Qwen3.5-27B
"""

import argparse
import json
import sys
import urllib.request
import urllib.error


def query_vllm(prompt, url, model, max_tokens=4096):
    """Send a prompt to a vLLM OpenAI-compatible endpoint."""
    endpoint = f"{url}/v1/chat/completions"

    payload = json.dumps({
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are an expert IC packaging and layout reliability engineer. "
                           "Provide detailed, actionable risk assessments."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }).encode("utf-8")

    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
    except urllib.error.URLError as e:
        print(f"Error connecting to vLLM at {url}: {e}")
        print("Make sure vLLM is running: vllm serve <model> --dtype bfloat16")
        sys.exit(1)
    except (KeyError, IndexError) as e:
        print(f"Unexpected response format: {e}")
        print(f"Raw response: {data}")
        sys.exit(1)


def detect_model(url):
    """Query the vLLM server to see which model is loaded."""
    try:
        req = urllib.request.Request(f"{url}/v1/models")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = data.get("data", [])
            if models:
                model_id = models[0]["id"]
                print(f"  Detected model: {model_id}")
                return model_id
    except Exception:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Send overlap analysis prompt to a local vLLM server."
    )
    parser.add_argument("prompt_file", help="Path to prompt.txt from report_generator.py")
    parser.add_argument("--url", default="http://localhost:8000",
                        help="vLLM server URL (default: http://localhost:8000)")
    parser.add_argument("--model", default=None,
                        help="Model name (default: auto-detect from server)")
    parser.add_argument("--output", default=None,
                        help="Save analysis to file (default: print to stdout)")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Max tokens for response (default: 4096)")
    args = parser.parse_args()

    # Read prompt
    with open(args.prompt_file) as f:
        prompt = f.read()
    print(f"Loaded prompt: {len(prompt)} characters")

    # Detect or use specified model
    model = args.model
    if not model:
        print(f"Checking vLLM server at {args.url} ...")
        model = detect_model(args.url)
        if not model:
            print("Could not detect model. Specify with --model")
            sys.exit(1)

    print(f"Sending to {model} ...")
    response = query_vllm(prompt, args.url, model, args.max_tokens)
    print(f"Got response: {len(response)} characters")

    if args.output:
        with open(args.output, "w") as f:
            f.write(f"# LLM Risk Assessment\n\n")
            f.write(f"**Model:** {model}\n\n")
            f.write(response)
        print(f"Analysis saved to {args.output}")
    else:
        print("\n" + "=" * 60)
        print(response)


if __name__ == "__main__":
    main()
