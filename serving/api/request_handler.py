"""
HTTP API request parsing, validation, and execution handler.
"""

import time
import uuid
import math
from typing import Dict, Any, Optional, List
from utils.logging_utils import get_logger

logger = get_logger("request_handler")


class RequestHandler:
    """
    Handles endpoint request validation, inference invocation, and response formatting.
    """

    def __init__(self, engine: Optional[Any] = None, tokenizer: Optional[Any] = None):
        self.engine = engine
        self.tokenizer = tokenizer

    def handle_completion(self, request_body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process OpenAI-compatible /v1/completions request.

        Args:
            request_body: Dictionary containing 'prompt', optional 'max_tokens', 'temperature', etc.

        Returns:
            Formatted response dictionary.
        """
        if not isinstance(request_body, dict):
            raise ValueError("request body must be a JSON object")

        prompt = request_body.get("prompt", "")
        if not isinstance(prompt, str):
            raise ValueError("prompt must be a string")
        max_tokens = request_body.get("max_tokens", 16)
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 0:
            raise ValueError("max_tokens must be a non-negative integer")
        temperature = request_body.get("temperature", 0.8)
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or temperature < 0
        ):
            raise ValueError("temperature must be a finite non-negative number")
        top_k = request_body.get("top_k")
        if top_k is not None and (
            not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 0
        ):
            raise ValueError("top_k must be a non-negative integer")
        top_p = request_body.get("top_p")
        if top_p is not None and (
            isinstance(top_p, bool)
            or not isinstance(top_p, (int, float))
            or not math.isfinite(top_p)
            or not 0.0 <= top_p <= 1.0
        ):
            raise ValueError("top_p must be a finite number in [0, 1]")

        # Tokenize prompt
        if self.tokenizer is not None and hasattr(self.tokenizer, "encode"):
            prompt_tokens = self.tokenizer.encode(prompt)
        else:
            prompt_tokens = [ord(c) for c in prompt]

        # Generate continuation
        if self.engine is not None and hasattr(self.engine, "generate_tokens"):
            generated_tokens = self.engine.generate_tokens(
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_tokens,
                temperature=temperature
            )
            response_text = (
                self.tokenizer.decode(generated_tokens)
                if self.tokenizer is not None and hasattr(self.tokenizer, "decode")
                else " ".join(str(token) for token in generated_tokens)
            )
        elif self.engine is not None and hasattr(self.engine, "generate"):
            raw_gen = self.engine.generate(prompt, tokenizer=self.tokenizer, max_new_tokens=max_tokens)
            response_text = raw_gen
            try:
                generated_tokens = self.tokenizer.encode(raw_gen) if self.tokenizer is not None else prompt_tokens
            except Exception:
                generated_tokens = prompt_tokens
        else:
            generated_tokens = prompt_tokens + [prompt_tokens[-1] + 1 if prompt_tokens else 42]
            response_text = (
                self.tokenizer.decode(generated_tokens)
                if self.tokenizer is not None and hasattr(self.tokenizer, "decode")
                else " ".join(str(token) for token in generated_tokens)
            )

        return {
            "id": f"cmpl-{uuid.uuid4().hex[:12]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": "broken-transformer",
            "text": response_text,
            "choices": [
                {
                    "text": response_text,
                    "index": 0,
                    "finish_reason": "length"
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_tokens),
                "completion_tokens": max(0, len(generated_tokens) - len(prompt_tokens)),
                "total_tokens": len(generated_tokens)
            }
        }

    def handle_health(self) -> Dict[str, Any]:
        """Process /health endpoint check."""
        return {
            "status": "ok",
            "service": "broken-transformer-server",
            "timestamp": time.time()
        }

    def handle_models(self) -> Dict[str, Any]:
        """Process /v1/models endpoint check."""
        return {
            "object": "list",
            "data": [
                {
                    "id": "broken-transformer",
                    "object": "model",
                    "owned_by": "openai-dsg-iitr",
                    "ready": True
                }
            ]
        }
