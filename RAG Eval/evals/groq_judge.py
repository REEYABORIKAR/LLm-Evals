import os
import asyncio
import time

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from deepeval.models import DeepEvalBaseLLM


load_dotenv()


class GroqJudge(DeepEvalBaseLLM):

    def __init__(self, model="openai/gpt-oss-120b"):
        self.model = model

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set in the .env file."
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )

    def load_model(self):
        return self.client

    def generate(
        self,
        prompt: str,
        schema: BaseModel | None = None,
    ):
        max_retries = 5

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    temperature=0,
                )

                output = response.choices[0].message.content

                if output is None:
                    raise ValueError(
                        "Groq returned an empty response."
                    )

                if schema is not None:
                    return schema.model_validate_json(output)

                return output

            except Exception as e:
                error_text = str(e)

                # Retry temporary Groq errors:
                # 429 = rate limit
                # 503 = model temporarily overloaded
                is_retryable = (
                    "429" in error_text
                    or "503" in error_text
                    or "rate_limit" in error_text.lower()
                    or "over capacity" in error_text.lower()
                )

                if not is_retryable:
                    raise

                if attempt == max_retries - 1:
                    raise

                # Exponential backoff:
                # 1s, 2s, 4s, 8s
                wait_time = 2 ** attempt

                print(
                    f"\nGroq temporary error "
                    f"(attempt {attempt + 1}/{max_retries}). "
                    f"Retrying in {wait_time}s..."
                )

                time.sleep(wait_time)

        raise RuntimeError(
            "Groq request failed after maximum retries."
        )

    async def a_generate(
        self,
        prompt: str,
        schema: BaseModel | None = None,
    ):
        return await asyncio.to_thread(
            self.generate,
            prompt,
            schema,
        )

    def get_model_name(self):
        return f"Groq {self.model}"