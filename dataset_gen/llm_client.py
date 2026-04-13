import os
import time
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
    """Standalone OpenAI client for persona generation pipeline"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize LLM client

        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

        self.client = OpenAI(api_key=self.api_key)

    def call(
        self,
        prompt: str,
        model: str = "gpt-4.1",
        temperature: float = 0.7,
        max_tokens: int = 8192,
        retry_attempts: int = 3
    ) -> str:
        return self._call_with_retry(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            retry_attempts=retry_attempts
        )

    # Keep backwards-compatible alias
    def call_gpt41(self, prompt: str, temperature: float = 0.7,
                   max_tokens: int = 8192, retry_attempts: int = 3) -> str:
        return self.call(prompt, model="gpt-4.1", temperature=temperature,
                         max_tokens=max_tokens, retry_attempts=retry_attempts)

    def _call_with_retry(
        self,
        model: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
        retry_attempts: int
    ) -> str:
        """Internal method to call OpenAI API with retry logic"""
        last_exception = None

        # Models >= gpt-5 use max_completion_tokens instead of max_tokens
        newer_models = {"gpt-5", "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano",
                        "gpt-5.4-pro", "gpt-5.3", "gpt-5.2", "gpt-5.1"}
        use_completion_tokens = any(model.startswith(m) for m in newer_models)

        for attempt in range(retry_attempts):
            try:
                kwargs = dict(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                )
                if use_completion_tokens:
                    kwargs["max_completion_tokens"] = max_tokens
                else:
                    kwargs["max_tokens"] = max_tokens
                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content

            except Exception as e:
                last_exception = e
                if attempt < retry_attempts - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"API call failed (attempt {attempt + 1}/{retry_attempts}). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"API call failed after {retry_attempts} attempts.")

        raise Exception(f"Failed to call {model} after {retry_attempts} attempts: {last_exception}")
