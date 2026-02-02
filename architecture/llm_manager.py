"""Centralized LiteLLM client manager with multi-key rotation and retry logic.

Supports multiple providers: Groq, Gemini, OpenAI, etc.
"""

import os
import time
from itertools import cycle
from typing import Optional, Dict, Any, List
from litellm import completion
from utils.logger import get_logger

logger = get_logger(__name__)


class LLMManager:
    """
    Manages LLM API calls with:
    - Multiple API key rotation (e.g., GROQ_API_KEY_1, GEMINI_API_KEY_1, ...)
    - Automatic retry with exponential backoff
    - Rate limit handling
    - Multiple provider support
    """

    # Provider configurations
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
    }

    def __init__(self, provider: str = "groq", model: Optional[str] = None):
        if provider not in self.PROVIDER_CONFIG:
            raise ValueError(f"Unknown provider: {provider}. Supported: {list(self.PROVIDER_CONFIG.keys())}")
        
        self.provider = provider
        config = self.PROVIDER_CONFIG[provider]
        env_prefix = config["env_prefix"]
        
        # Set model with provider prefix
        model_name = model or config["default_model"]
        self.model = f"{config['litellm_prefix']}/{model_name}"
        
        # Try numbered keys first (e.g., GROQ_API_KEY_1, GROQ_API_KEY_2, ...)
        self.api_keys = [os.getenv(f"{env_prefix}_{i}") for i in range(1, 16)]
        self.api_keys = [key for key in self.api_keys if key]
        
        # Fallback to single key (e.g., GROQ_API_KEY)
        if not self.api_keys:
            single_key = os.getenv(env_prefix)
            if single_key:
                self.api_keys = [single_key]
        
        if not self.api_keys:
            logger.warning(f"No {provider.upper()} API keys found!")
            self.key_pool = None
        else:
            self.key_pool = cycle(self.api_keys)
            logger.info(f"Loaded {len(self.api_keys)} {provider.upper()} API key(s)")

    def get_next_key(self) -> Optional[str]:
        if not self.key_pool:
            return None
        return next(self.key_pool)

    def generate(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, str]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        max_retries: int = 3,
        initial_delay: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Generate completion with automatic retry and key rotation.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            response_format: Optional format spec (e.g., {"type": "json_object"})
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            max_retries: Retries per key before switching
            initial_delay: Initial backoff delay in seconds
            
        Returns:
            Dict with 'content' (str) and optional 'error' or 'rate_limit_warning'
        """
        if not self.api_keys:
            return {
                "content": None,
                "error": f"No {self.provider.upper()} API keys configured",
            }

        last_error = None

        for key_attempt in range(len(self.api_keys)):
            api_key = self.get_next_key()

            for retry_attempt in range(max_retries):
                try:
                    kwargs = {
                        "model": self.model,
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "api_key": api_key,
                    }
                    
                    if response_format:
                        kwargs["response_format"] = response_format

                    response = completion(**kwargs)
                    content = response.choices[0].message.content

                    return {"content": content}

                except Exception as e:
                    error_str = str(e)
                    
                    # Rate limit errors
                    if any(x in error_str for x in ["429", "rate_limit", "ResourceExhausted", "quota"]):
                        last_error = e
                        logger.warning(
                            f"Rate limit hit on key {key_attempt + 1}, attempt {retry_attempt + 1}/{max_retries}"
                        )

                        if retry_attempt < max_retries - 1:
                            delay = initial_delay * (2 ** retry_attempt)
                            logger.info(f"Retrying in {delay}s...")
                            time.sleep(delay)
                        else:
                            logger.warning(
                                f"Max retries reached for key {key_attempt + 1}, trying next key..."
                            )
                            break
                    
                    # JSON generation failure - retry with same key
                    elif "json_validate_failed" in error_str:
                        last_error = e
                        logger.warning(f"JSON generation failed, retrying... ({retry_attempt + 1}/{max_retries})")
                        if retry_attempt < max_retries - 1:
                            time.sleep(0.5)
                        else:
                            break
                    
                    # Other errors - don't retry
                    else:
                        last_error = e
                        logger.error(f"LLM API error: {e}")
                        return {"content": None, "error": str(e)}

        logger.error("All API keys exhausted or rate-limited.")
        return {
            "content": None,
            "error": str(last_error),
            "rate_limit_warning": "All API keys exhausted. Please try again later.",
        }

    def generate_json(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 4096,
        **kwargs
    ) -> Dict[str, Any]:
        """Convenience method for JSON mode generation."""
        return self.generate(
            messages=messages,
            response_format={"type": "json_object"},
            max_tokens=max_tokens,
            **kwargs
        )

    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Convenience method for simple text generation."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        return self.generate(messages=messages, **kwargs)


# Singleton instances per provider
_llm_managers: Dict[str, LLMManager] = {}


def get_llm_manager(
    provider: str = "groq",
    model: Optional[str] = None
) -> LLMManager:
    """
    Get or create the LLM manager instance for a provider.
    
    Args:
        provider: "groq", "gemini", "openai", or "anthropic"
        model: Optional model override (uses provider default if not specified)
    """
    global _llm_managers
    
    # Create unique key based on provider and model
    cache_key = f"{provider}:{model or 'default'}"
    
    if cache_key not in _llm_managers:
        _llm_managers[cache_key] = LLMManager(provider=provider, model=model)
    
    return _llm_managers[cache_key]


def get_groq_manager(model: str = "llama-3.3-70b-versatile") -> LLMManager:
    """Convenience function for Groq."""
    return get_llm_manager(provider="groq", model=model)


def get_gemini_manager(model: str = "gemini-2.5-flash") -> LLMManager:
    """Convenience function for Gemini."""
    return get_llm_manager(provider="gemini", model=model)


def reset_llm_managers():
    """Reset all singleton instances (useful for testing)."""
    global _llm_managers
    _llm_managers = {}
