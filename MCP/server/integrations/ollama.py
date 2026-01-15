"""
Ollama SDK integration for LLM and Embedding services
"""

import json
import re
from typing import List, Dict, Any, Optional


class OllamaClient:
    """
    Ollama API client for LLM and Embedding operations.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        llm_model: str = "llama3.1:8b-instruct",
        embedding_model: str = "nomic-embed-text",
    ):
        self.base_url = base_url.rstrip("/")
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self._client = None

    def _get_client(self):
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={"Content-Type": "application/json"},
                timeout=120.0,
            )
        return self._client

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict] = None,
        stream: bool = False,
    ) -> str:
        client = self._get_client()
        request_messages = list(messages)

        if response_format and response_format.get("type") == "json_object":
            request_messages = self._enforce_json_output(request_messages)

        payload = {
            "model": self.llm_model,
            "messages": request_messages,
            "stream": stream,
            "options": {
                "temperature": temperature,
            },
        }

        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens

        if stream:
            return await self._stream_completion(payload)

        response = await client.post("/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "").strip()

    async def _stream_completion(self, payload: Dict) -> str:
        client = self._get_client()
        content_parts = []

        async with client.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                message = data.get("message", {})
                if message.get("content"):
                    content_parts.append(message["content"])
                if data.get("done"):
                    break

        return "".join(content_parts)

    async def create_embedding(self, texts: List[str]) -> List[List[float]]:
        client = self._get_client()
        embeddings: List[List[float]] = []

        for text in texts:
            payload = {
                "model": self.embedding_model,
                "prompt": text,
            }
            response = await client.post("/api/embeddings", json=payload)
            response.raise_for_status()
            data = response.json()
            embedding = data.get("embedding", [])
            embeddings.append(embedding)

        return embeddings

    async def create_single_embedding(self, text: str) -> List[float]:
        embeddings = await self.create_embedding([text])
        return embeddings[0]

    def extract_json(self, text: str) -> Any:
        if not text:
            return None

        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        json_str = self._extract_json_substring(text)
        if json_str:
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                cleaned = self._clean_json_string(json_str)
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    return None

        return None

    def _enforce_json_output(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if not messages:
            return messages

        json_hint = "Respond with ONLY valid JSON. Do not include markdown or extra text."
        updated = list(messages)
        last = updated[-1]
        updated[-1] = {
            "role": last.get("role", "user"),
            "content": f"{last.get('content', '').strip()}\n\n{json_hint}",
        }
        return updated

    def _extract_json_substring(self, text: str) -> Optional[str]:
        start = text.find("{")
        if start == -1:
            start = text.find("[")
        if start == -1:
            return None

        stack = []
        for i, ch in enumerate(text[start:], start):
            if ch in "{[":
                stack.append(ch)
            elif ch in "}]":
                if stack:
                    stack.pop()
                if not stack:
                    return text[start : i + 1]

        return None

    def _clean_json_string(self, text: str) -> str:
        prefixes = [
            "Here's the JSON:",
            "Here is the JSON:",
            "JSON output:",
            "Output:",
            "Result:",
        ]
        cleaned = text.strip()
        for prefix in prefixes:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()

        cleaned = re.sub(r",\s*([\}\]])", r"\1", cleaned)
        cleaned = re.sub(r"//.*$", "", cleaned, flags=re.MULTILINE)
        return cleaned


class OllamaClientManager:
    """
    Manages a shared Ollama client instance.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        llm_model: str = "llama3.1:8b-instruct",
        embedding_model: str = "nomic-embed-text",
    ):
        self.base_url = base_url
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self._client: Optional[OllamaClient] = None

    def get_client(self, api_key: str = "") -> OllamaClient:
        if self._client is None:
            self._client = OllamaClient(
                base_url=self.base_url,
                llm_model=self.llm_model,
                embedding_model=self.embedding_model,
            )
        return self._client

    async def close_all(self):
        if self._client:
            await self._client.close()
            self._client = None
