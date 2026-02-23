#!/usr/bin/env python3
"""Minimal OpenAI-compatible client for SGLang server."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional

import requests


def list_models(base_url: str = "http://127.0.0.1:30000", timeout: int = 20) -> List[str]:
    """Return model ids exposed by SGLang."""
    endpoint = f"{base_url.rstrip('/')}/v1/models"
    response = requests.get(endpoint, timeout=timeout)
    response.raise_for_status()

    payload = response.json()
    models = payload.get("data", [])
    return [m.get("id", "") for m in models if m.get("id")]


def _stream_sse_chunks(response: requests.Response):
    """Yield parsed JSON payloads from an SSE stream."""
    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        if not raw_line.startswith("data: "):
            continue

        chunk = raw_line[len("data: ") :].strip()
        if chunk == "[DONE]":
            break

        try:
            yield json.loads(chunk)
        except json.JSONDecodeError:
            # Ignore malformed lines and continue reading stream.
            continue


def _extract_text_from_chunk(chunk: Dict[str, Any]) -> str:
    """Extract textual content from OpenAI-compatible chunk payloads."""
    choices = chunk.get("choices", [])
    if not choices:
        return ""

    choice = choices[0]
    delta = choice.get("delta", {})
    if isinstance(delta, dict) and delta.get("content"):
        return str(delta["content"])

    message = choice.get("message", {})
    if isinstance(message, dict) and message.get("content"):
        return str(message["content"])

    text = choice.get("text")
    if isinstance(text, str):
        return text

    return ""


def sglang_chat_completion(
    prompt: str,
    model: Optional[str] = None,
    base_url: str = "http://127.0.0.1:30000",
    system_prompt: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    stream: bool = True,
    timeout: int = 120,
    extra_body: Optional[Dict[str, Any]] = None,
    print_stream: bool = True,
) -> str:
    """Call SGLang `/v1/chat/completions` and return generated text."""
    model_id = model
    if not model_id:
        available = list_models(base_url=base_url, timeout=timeout)
        if not available:
            raise RuntimeError(f"No models found at {base_url}/v1/models")
        model_id = available[0]

    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})

    payload: Dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    if extra_body:
        payload["extra_body"] = extra_body

    endpoint = f"{base_url.rstrip('/')}/v1/chat/completions"
    with requests.post(endpoint, json=payload, stream=True, timeout=timeout) as response:
        response.raise_for_status()

        text_parts: List[str] = []
        content_type = response.headers.get("Content-Type", "")
        if "text/event-stream" in content_type:
            for chunk in _stream_sse_chunks(response):
                content = _extract_text_from_chunk(chunk)
                if content:
                    if print_stream:
                        print(content, end="", flush=True)
                    text_parts.append(content)

                choices = chunk.get("choices", [])
                if choices and choices[0].get("finish_reason") is not None:
                    break
        else:
            chunk = response.json()
            content = _extract_text_from_chunk(chunk)
            if content:
                if print_stream:
                    print(content, end="", flush=True)
                text_parts.append(content)

    if print_stream:
        print()

    return "".join(text_parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test SGLang OpenAI-compatible chat API")
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default=None, help="Model id. If omitted, first server model is used.")
    parser.add_argument("--prompt", default="Reply with exactly: sglang test ok")
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--no-stream", action="store_true", help="Request stream=false in payload.")
    parser.add_argument("--quiet", action="store_true", help="Do not print token stream.")
    parser.add_argument("--chat", action="store_true", help="Run interactive multi-turn chat mode.")
    parser.add_argument("--list-models", action="store_true", help="List available models and exit.")
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Set extra_body.chat_template_kwargs.enable_thinking=false (model-dependent).",
    )
    args = parser.parse_args()

    if args.list_models:
        models = list_models(base_url=args.base_url, timeout=args.timeout)
        for model_id in models:
            print(model_id)
        return

    extra_body: Optional[Dict[str, Any]] = None
    if args.disable_thinking:
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

    if args.chat:
        history: List[Dict[str, str]] = []
        print("Chat mode enabled. Commands: /bye, /quit, /exit, /reset")
        while True:
            prompt = input("User: ").strip()
            if prompt in {"/bye", "/quit", "/exit"}:
                print("See you again!")
                break
            if prompt == "/reset":
                history = []
                print("Chat history cleared.")
                continue
            if not prompt:
                continue

            if not args.quiet:
                print("Assistant: ", end="", flush=True)

            output = sglang_chat_completion(
                prompt=prompt,
                model=args.model,
                base_url=args.base_url,
                system_prompt=args.system_prompt,
                history=history,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                stream=not args.no_stream,
                timeout=args.timeout,
                extra_body=extra_body,
                print_stream=not args.quiet,
            )
            if args.quiet:
                print(f"Assistant: {output}")

            history.append({"role": "user", "content": prompt})
            history.append({"role": "assistant", "content": output})
        return

    output = sglang_chat_completion(
        prompt=args.prompt,
        model=args.model,
        base_url=args.base_url,
        system_prompt=args.system_prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stream=not args.no_stream,
        timeout=args.timeout,
        extra_body=extra_body,
        print_stream=not args.quiet,
    )

    if args.quiet:
        print(output)


if __name__ == "__main__":
    main()
