#!/usr/bin/env python3
"""Deterministic mock LLM provider server for tests (stdlib only).

Run: uv run --no-project scripts/mock_llm_server.py [port]

Responses echo the request so tests can assert row/param association:
  openai/local: "openai|<model>|<prompt>|t=<temperature>|mt=<max_tokens>|sys=<system>|fmt=<response_format type>|auth=<bearer>"
  gemini:       "gemini|<model>|<prompt>|t=<temperature>|sys=<system>|key=<key>"
  cloudflare:   "cloudflare|<account>|<model>|<prompt>|mt=<max_tokens>|sys=<system>"
Prompt triggers:
  contains FAIL      -> HTTP 500
  contains MALFORMED -> 200 with malformed JSON body
  contains NONSTRING -> 200 with null content
"""

import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse


def fmt_num(value):
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def find_messages(messages):
    prompt, system = "", ""
    for msg in messages or []:
        if msg.get("role") == "system":
            system = msg.get("content", "")
        elif msg.get("role") == "user":
            prompt = msg.get("content", "")
    return prompt, system


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("mock: " + fmt % args + "\n")

    def send(self, status, body):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self.send(200, '{"ok":true}')
        else:
            self.send(404, '{"error":{"message":"not found"}}')

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            req = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self.send(400, '{"error":{"message":"bad request json"}}')
            return
        url = urlparse(self.path)

        if url.path == "/v1/chat/completions":
            prompt, system = find_messages(req.get("messages"))
            if self.trigger(prompt):
                return
            if "NONSTRING" in prompt:
                self.send(200, json.dumps({"choices": [{"message": {"role": "assistant", "content": None}}]}))
                return
            auth = self.headers.get("Authorization", "")
            auth = auth[len("Bearer ") :] if auth.startswith("Bearer ") else auth
            fmt = (req.get("response_format") or {}).get("type", "")
            content = "|".join(
                [
                    "openai",
                    req.get("model", ""),
                    prompt,
                    "t=" + fmt_num(req.get("temperature")),
                    "mt=" + fmt_num(req.get("max_tokens")),
                    "sys=" + system,
                    "fmt=" + fmt,
                    "auth=" + auth,
                ]
            )
            self.send(200, json.dumps({"choices": [{"message": {"role": "assistant", "content": content}}]}))
            return

        m = re.match(r"^/v1beta/models/([^:]+):generateContent$", url.path)
        if m:
            parts = req.get("contents", [{}])[0].get("parts", [{}])
            prompt = parts[0].get("text", "") if parts else ""
            sys_parts = (req.get("systemInstruction") or {}).get("parts", [])
            system = sys_parts[0].get("text", "") if sys_parts else ""
            if self.trigger(prompt):
                return
            if "NONSTRING" in prompt:
                self.send(200, json.dumps({"candidates": [{"content": {"parts": [{"text": None}]}}]}))
                return
            key = parse_qs(url.query).get("key", [""])[0]
            temp = (req.get("generationConfig") or {}).get("temperature")
            text = "|".join(["gemini", m.group(1), prompt, "t=" + fmt_num(temp), "sys=" + system, "key=" + key])
            self.send(200, json.dumps({"candidates": [{"content": {"parts": [{"text": text}]}}]}))
            return

        m = re.match(r"^/client/v4/accounts/([^/]+)/ai/run/(.+)$", url.path)
        if m:
            prompt, system = find_messages(req.get("messages"))
            if self.trigger(prompt):
                return
            if "NONSTRING" in prompt:
                self.send(200, json.dumps({"success": True, "result": {"response": None}}))
                return
            text = "|".join(
                ["cloudflare", m.group(1), m.group(2), prompt, "mt=" + fmt_num(req.get("max_tokens")), "sys=" + system]
            )
            self.send(200, json.dumps({"success": True, "result": {"response": text}}))
            return

        self.send(404, '{"error":{"message":"unknown route"}}')

    def trigger(self, prompt):
        if "FAIL" in prompt:
            self.send(500, '{"error":{"message":"mock failure"}}')
            return True
        if "MALFORMED" in prompt:
            self.send(200, '{"choices": [ this is not json')
            return True
        return False


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 18080
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    sys.stderr.write(f"mock LLM server on http://127.0.0.1:{port}\n")
    server.serve_forever()


if __name__ == "__main__":
    main()
