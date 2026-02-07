"""LLM Manager - unified interface for multiple LLM providers via LiteLLM."""

import os
import time
from itertools import cycle
from typing import Any, Dict, List, Optional

import litellm
import nest_asyncio
from litellm import completion

from utils.logger import SESSION_ID, get_logger

nest_asyncio.apply()

logger = get_logger(__name__)

if os.getenv("OPIK_API_KEY"):
    os.environ["OPIK_PROJECT_NAME"] = os.getenv("OPIK_PROJECT_NAME", "IASCIS")
    litellm.callbacks = ["opik"]
    logger.info(
        f"Opik callback registered for project: {os.environ['OPIK_PROJECT_NAME']}"
    )
else:
    logger.warning("OPIK_API_KEY not found in environment. Opik tracing disabled.")


class LLMManager:
    """Unified LLM interface with key rotation and retry logic."""

    PROVIDER_CONFIG = {
        "groq": {
            "env_prefix": "GROQ_API_KEY",
            "default_model": "llama-3.3-70b-versatile",
            "litellm_prefix": "groq",
        },
        "gemini": {
            "env_prefix": "GEMINI_API_KEY",
            "default_model": "gemini-3-flash-preview",
            "litellm_prefix": "gemini",
        },
        "openai": {
            "env_prefix": "OPENAI_API_KEY",
            "default_model": "gpt-4o",
            "litellm_prefix": "openai",
        },
        "anthropic": {
            "env_prefix": "ANTHROPIC_API_KEY",
            "default_model": "claude-3-sonnet-20240229",
            "litellm_prefix": "anthropic",
        },
        "cerebras": {
            "env_prefix": "CEREBRAS_API_KEY",
            "default_model": "llama3.1-70b",
            "litellm_prefix": "cerebras",
        },
    }

    def __init__(self, provider: str = "groq", model: Optional[str] = None):
        if provider not in self.PROVIDER_CONFIG:
            raise ValueError(f"Unknown provider: {provider}")

        self.provider = provider
        config = self.PROVIDER_CONFIG[provider]
        env_prefix = config["env_prefix"]
        model_name = model or config["default_model"]
        self.model = f"{config['litellm_prefix']}/{model_name}"

        self.api_keys = [os.getenv(f"{env_prefix}_{i}") for i in range(1, 16)]
        self.api_keys = [k for k in self.api_keys if k]

        if not self.api_keys:
            if single_key := os.getenv(env_prefix):
                self.api_keys = [single_key]

        if not self.api_keys:
            logger.warning(f"No {provider.upper()} API keys found")
            self.key_pool = None
        else:
            self.key_pool = cycle(self.api_keys)
            logger.info(f"Loaded {len(self.api_keys)} {provider.upper()} keys")

    def get_next_key(self) -> Optional[str]:
        """Get next API key from rotation pool."""
        return next(self.key_pool) if self.key_pool else None

    def generate(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, str]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate completion with automatic key rotation and retries."""
        if not self.api_keys:
            return {"content": None, "error": f"No {self.provider} keys"}

        last_error = None
        start_time = time.perf_counter()

        metadata = kwargs.get("metadata", {})
        if "opik" not in metadata:
            metadata["opik"] = {}

        tags = metadata["opik"].get("tags", [])
        if isinstance(tags, str):
            tags = [tags]

        session_tag = f"session:{SESSION_ID}"
        if session_tag not in tags:
            tags.append(session_tag)

        metadata["opik"]["tags"] = tags
        kwargs["metadata"] = metadata

        logger.info(
            f"Generating with {self.model} | Tokens: {max_tokens} | Temp: {temperature} | Session: {SESSION_ID}"
        )

        for key_idx in range(len(self.api_keys)):
            api_key = self.get_next_key()

            for attempt in range(max_retries):
                try:
                    params = {
                        "model": self.model,
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "api_key": api_key,
                        **kwargs,
                    }
                    if response_format:
                        params["response_format"] = response_format

                    response = completion(**params)
                    content = response.choices[0].message.content

                    duration = (time.perf_counter() - start_time) * 1000
                    logger.info(f"Generation success in {duration:.2f}ms")

                    return {"content": content}

                except Exception as e:
                    last_error = e
                    error_msg = str(e)
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed using key {key_idx + 1}: {error_msg}"
                    )

                    if any(x in error_msg for x in ["429", "rate_limit", "quota"]):
                        if attempt < max_retries - 1:
                            time.sleep(initial_delay * (2**attempt))
                        else:
                            break
                    elif "json_validate_failed" in error_msg:
                        if attempt < max_retries - 1:
                            time.sleep(0.5)
                        else:
                            break
                    else:
                        logger.error(f"Non-retryable error: {e}")
                        return {"content": None, "error": str(e)}

        logger.error(f"All keys failed. Last error: {last_error}")
        return {"content": None, "error": str(last_error)}

    def generate_json(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Generate completion with JSON response format, with fallback."""
        result = self.generate(
            messages, response_format={"type": "json_object"}, **kwargs
        )

        if result.get("error") and "json_validate_failed" in str(result["error"]):
            logger.warning("JSON mode failed, retrying without JSON format constraint")
            result = self.generate(messages, **kwargs)

            if result.get("content"):
                import re

                content = result["content"]
                json_match = re.search(r"\{[\s\S]*\}", content)
                if json_match:
                    result["content"] = json_match.group(0)

        return result

    def generate_text(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs
    ) -> Dict[str, Any]:
        """Generate completion from simple text prompt."""
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": prompt})
        return self.generate(messages=msgs, **kwargs)


_llm_managers: Dict[str, LLMManager] = {}


def get_llm_manager(provider: str = "groq", model: Optional[str] = None) -> LLMManager:
    """Get or create a cached LLMManager instance."""
    global _llm_managers
    key = f"{provider}:{model or 'default'}"
    if key not in _llm_managers:
        _llm_managers[key] = LLMManager(provider, model)
    return _llm_managers[key]


def get_groq_manager(model: str = "llama-3.3-70b-versatile") -> LLMManager:
    """Get Groq LLMManager instance."""
    return get_llm_manager("groq", model)


def get_gemini_manager(model: str = "gemini-2.5-flash") -> LLMManager:
    """Get Gemini LLMManager instance."""
    return get_llm_manager("gemini", model)


def get_cerebras_manager(model: str = "llama3.1-70b") -> LLMManager:
    """Get Cerebras LLMManager instance."""
    return get_llm_manager("cerebras", model)


def reset_llm_managers():
    """Clear all cached LLMManager instances."""
    global _llm_managers
    _llm_managers = {}
