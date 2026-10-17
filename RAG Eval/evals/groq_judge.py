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
                    max_tokens=4096,
                )

                output = response.choices[0].message.content

                if not output or not output.strip():
                    raise ValueError("Groq returned an empty response.")

                output = output.strip()

                # Remove markdown JSON fences
                if output.startswith("```"):
                    output = output.replace("```json", "")
                    output = output.replace("```", "")
                    output = output.strip()

                if schema is not None:
                    try:
                        return schema.model_validate_json(output)
                    except Exception as json_error:
                        # Invalid/truncated JSON. Retry the Groq request.
                        print(f"\nGroq error: {type(e).__name__}: {error_text}")
                        print(
                                f"Attempt {attempt + 1}/{max_retries}. "
                                f"Retrying in {wait_time}s..."
                            )

                        if attempt == max_retries - 1:
                            print("\nLast Groq output:")
                            print(output[:2000])
                            raise json_error

                        time.sleep(2 ** attempt)
                        continue

                return output

            except Exception as e:
                error_text = str(e)

                is_retryable = (
                    "429" in error_text
                    or "503" in error_text
                    or "rate_limit" in error_text.lower()
                    or "over capacity" in error_text.lower()
                    or "temporary" in error_text.lower()
                    or "empty response" in error_text.lower()
                )

                if not is_retryable:
                    raise

                if attempt == max_retries - 1:
                    raise

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