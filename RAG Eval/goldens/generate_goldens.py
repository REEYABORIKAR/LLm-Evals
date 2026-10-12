import os
import glob
import json
import random
import time

from dotenv import load_dotenv
from groq import Groq
from deepeval.synthesizer import Synthesizer
from deepeval.models import DeepEvalBaseLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


def load_chunks():
    texts = []

    BASE_DIR = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    DATA_DIR = os.path.join(BASE_DIR, "data")

    vtt_files = glob.glob(
        os.path.join(DATA_DIR, "*.vtt")
    )

    print(f"Found {len(vtt_files)} VTT files")

    for path in vtt_files:
        print(f"Loading: {path}")

        with open(path, "r", encoding="utf-8") as f:
            lines = [
                ln.strip()
                for ln in f
                if ln.strip()
                and ln.strip() != "WEBVTT"
                and "-->" not in ln
            ]

        texts.append(" ".join(lines))

    if not texts:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    return splitter.split_text("\n\n".join(texts))


chunks = load_chunks()

if not chunks:
    raise ValueError("No chunks found. Check your data/*.vtt files.")

sample = random.sample(
    chunks,
    min(15, len(chunks))
)

contexts = [[c] for c in sample]


groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError(
        "GROQ_API_KEY is not set in your .env file."
    )

MAX_RETRIES = 6

PREFERRED_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "qwen/qwen3-32b",
]

EXCLUDED_KEYWORDS = [
    "whisper", "tts", "guard", "orpheus",
    "safeguard", "embed", "playai", "compound",
]


def pick_working_model(client):
    available = [m.id for m in client.models.list().data]
    print(f"Models listed for your key: {available}")

    chat_models = [
        m for m in available
        if not any(k in m.lower() for k in EXCLUDED_KEYWORDS)
    ]

    ordered = [m for m in PREFERRED_MODELS if m in chat_models]
    ordered += [m for m in chat_models if m not in ordered]

    env_model = os.getenv("GROQ_MODEL")
    if env_model:
        ordered = [env_model] + [m for m in ordered if m != env_model]

    for model_id in ordered:
        try:
            client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": "Say OK"}],
                max_tokens=5,
            )
            print(f"Using Groq model: {model_id}")
            return model_id
        except Exception as e:
            print(f"Model {model_id} not usable: {str(e)[:120]}")

    raise RuntimeError(
        "No usable Groq chat model found for this API key. "
        "Check model access in the Groq console (console.groq.com)."
    )


def _clean_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    return text.strip()


class GroqModel(DeepEvalBaseLLM):
    def __init__(self, model_name, client):
        self.model_name = model_name
        self.client = client

    def load_model(self):
        return self.client

    def _messages(self, prompt, schema):
        if schema is not None:
            prompt = (
                prompt
                + "\n\nReturn ONLY valid JSON that matches this JSON schema, "
                + "with no extra text:\n"
                + json.dumps(schema.model_json_schema())
            )
        return [{"role": "user", "content": prompt}]

    def _parse(self, text, schema):
        if schema is None:
            return text
        return schema(**json.loads(_clean_json(text)))

    def generate(self, prompt, schema=None):
        for attempt in range(MAX_RETRIES):
            try:
                kwargs = {}
                if schema is not None:
                    kwargs["response_format"] = {"type": "json_object"}

                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=self._messages(prompt, schema),
                    temperature=0,
                    **kwargs,
                )
                return self._parse(resp.choices[0].message.content, schema)
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    wait = 10 * (attempt + 1)
                    print(f"Rate limited, waiting {wait}s (attempt {attempt + 1}/{MAX_RETRIES})")
                    time.sleep(wait)
                    continue
                raise
        raise RuntimeError("Groq rate limit retries exhausted.")

    async def a_generate(self, prompt, schema=None):
        return self.generate(prompt, schema)

    def get_model_name(self):
        return self.model_name


groq_client = Groq(api_key=groq_api_key)
groq_model = GroqModel(pick_working_model(groq_client), groq_client)


synthesizer = Synthesizer(
    model=groq_model,
    async_mode=False,
)


goldens = synthesizer.generate_goldens_from_contexts(
    contexts=contexts,
    include_expected_output=True,
    max_goldens_per_context=1,
)


rows = []

for i, g in enumerate(goldens, 1):
    rows.append({
        "id": f"g{i:03d}",
        "query": g.input,
        "ideal_answer": g.expected_output,
        "source": "TODO-verify",
    })


os.makedirs("golden", exist_ok=True)

output_file = "golden/retriever_deepeval_goldens.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(
        rows,
        f,
        indent=2,
        ensure_ascii=False
    )


print(f"Written {len(rows)} DRAFT goldens -> {output_file}")

print(
    "!! REVIEW EVERY ONE before using: "
    "check grounding, trim padding, fix leading questions."
)