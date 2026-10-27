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
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be a non-negative number")
        top_k = request_body.get("top_k")
        if top_k is not None and (not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 0):
            raise ValueError("top_k must be a non-negative integer")
        top_p = request_body.get("top_p")
        if top_p is not None and (
            isinstance(top_p, bool)
            or not isinstance(top_p, (int, float))
            or not math.isfinite(top_p)
            or not 0.0 <= top_p <= 1.0
        ):
            raise ValueError("top_p must be a number in [0, 1]")

        # Tokenize prompt
        if self.tokenizer is not None and hasattr(self.tokenizer, "encode"):
            prompt_tokens = self.tokenizer.encode(prompt)
        else:
            vocab_size = getattr(getattr(self.engine, "model_config", None), "vocab_size", 50257)
            prompt_tokens = [ord(c) if ord(c) < vocab_size else 1 for c in prompt]

        # Generate continuation
        if self.engine is not None and hasattr(self.engine, "generate_tokens"):
            generated_tokens = self.engine.generate_tokens(
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_tokens,
                temperature=float(temperature),
                top_k=top_k,
                top_p=top_p,
            )
            generated_tokens = list(generated_tokens)
            completion_tokens = generated_tokens[len(prompt_tokens):]
            completion_count = len(completion_tokens)
            if self.tokenizer is not None and hasattr(self.tokenizer, "decode"):
                completion_text = self.tokenizer.decode(completion_tokens)
            else:
                completion_text = "".join(chr(token) for token in completion_tokens if 0 <= token <= 0x10FFFF)
        elif self.engine is not None and hasattr(self.engine, "generate"):
            raw_gen = self.engine.generate(
                prompt,
                tokenizer=self.tokenizer,
                max_new_tokens=max_tokens,
                temperature=float(temperature),
                top_k=top_k,
                top_p=top_p,
            )
            completion_text = raw_gen[len(prompt):] if isinstance(raw_gen, str) and raw_gen.startswith(prompt) else str(raw_gen)
            if self.tokenizer is not None and hasattr(self.tokenizer, "encode"):
                completion_count = max(0, len(self.tokenizer.encode(completion_text)) - 1)
            else:
                completion_count = len(completion_text)
            generated_tokens = []
        else:
            raise RuntimeError("No inference engine is configured")

        stop_ids = getattr(getattr(self.engine, "inference_config", None), "stop_token_ids", [])
        finish_reason = "stop" if generated_tokens and generated_tokens[-1] in stop_ids else "length"
        return {
            "id": f"cmpl-{uuid.uuid4().hex[:12]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": "broken-transformer",
            "text": completion_text,
            "choices": [
                {
                    "text": completion_text,
                    "index": 0,
                    "finish_reason": finish_reason
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_tokens),
                "completion_tokens": completion_count,
                "total_tokens": len(prompt_tokens) + completion_count
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
