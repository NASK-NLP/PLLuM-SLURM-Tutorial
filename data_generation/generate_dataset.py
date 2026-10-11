import json
import asyncio
import random
import inspect
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from openai import APIStatusError, AsyncOpenAI
from tqdm.asyncio import tqdm
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tools as calculator

# Configuration
BASE_URL = "https://llmlab.plgrid.pl/api/v1"
MODEL_NAME = "Qwen/Qwen3.5-122B-A10B"

NUM_EXAMPLES = 200
CONCURRENCY_LIMIT = 15
OUTPUT_FILE = "data/tools_dataset.jsonl"

TOOL_FUNCTIONS = {
    name: function
    for name, function in vars(calculator).items()
    if inspect.isfunction(function) and function.__module__ == calculator.__name__
}
TOOL_DEFINITIONS = {
    name: (tuple(inspect.signature(function).parameters), inspect.getdoc(function))
    for name, function in TOOL_FUNCTIONS.items()
}
TOOLS = list(TOOL_DEFINITIONS)
TOOL_DESCRIPTIONS = {name: description for name, (_, description) in TOOL_DEFINITIONS.items()}
WEIGHTS = [0.25, 0.25, 0.15, 0.15, 0.05, 0.05, 0.05, 0.05]
API_ERRORS = []

EXAMPLE_JSON = """{
  "messages": [
    {"role": "user", "content": "Zebraliśmy 125.5 kg jabłek w sadzie. Chcemy je spakować do skrzynek, z których każda mieści dokładnie 12 kg. Ile kilogramów jabłek zostanie nam luzem po zapełnieniu maksymalnej liczby skrzynek?"},
    {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "modulo", "arguments": "{\\"a\\": 125.5, \\"b\\": 12}"}}]},
    {"role": "tool", "tool_call_id": "call_1", "name": "modulo", "content": "5.5"},
    {"role": "assistant", "content": "5.5"}
  ]
}"""

def get_system_prompt(target_tool: str) -> str:
    return f"""
Jesteś ekspertem tworzącym wysokiej jakości dane treningowe dla modeli LLM.
Wygeneruj jedną konwersację zawierającą naturalne zadanie tekstowe z matematyki.

Wymogi:
1. Styl: Pytanie musi brzmieć jak autentyczny, życiowy problem.
2. Trudność: Zadania mają zawierać 1-3 kroki obliczeniowe.
3. Cel główny: Zadanie MUSI w naturalny sposób wymagać użycia funkcji: {target_tool}. Możesz użyć też innych operacji, jeśli są potrzebne.
4. Wartości: Używaj liczb całkowitych oraz ułamków dziesiętnych.
5. Asystent musi rozwiązać zadanie krok po kroku, wywołując odpowiednie funkcje. Wyniki narzędzi muszą być poprawne.
6. Odpowiedzią jest tylko liczba.

Dostępne funkcje:
{chr(10).join(f"- {name}: {TOOL_DESCRIPTIONS[name]}" for name in TOOLS)}

Zwróć wyłącznie JSON z kluczem "messages".

Przykład:
{EXAMPLE_JSON}
"""

async def fetch_single_example(client: AsyncOpenAI, sem: asyncio.Semaphore) -> str | None:
    max_retries = 3
    target_tool = random.choices(TOOLS, weights=WEIGHTS, k=1)[0]
    prompt = get_system_prompt(target_tool)
    
    async with sem:
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": "Wygeneruj nowe, unikalne zadanie matematyczne."}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7
                )
                return response.choices[0].message.content
            except APIStatusError as error:
                if error.status_code in (400, 401, 403, 404):
                    raise RuntimeError(
                        f"API configuration error (HTTP {error.status_code}). "
                        "Check BASE_URL, MODEL_NAME, LLMLAB_API_KEY, and supported request options."
                    ) from error
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    API_ERRORS.append(f"HTTP {error.status_code}: {error}")
                    return None
            except Exception as error:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    API_ERRORS.append(f"{type(error).__name__}: {error}")
                    return None

async def generate_data() -> None:
    load_dotenv('.env')
    api_key = os.environ.get('LLMLAB_API_KEY', '').strip()
    if not api_key:
        raise RuntimeError('Set LLMLAB_API_KEY in .env or in the environment before generating data.')
    client = AsyncOpenAI(api_key=api_key, base_url=BASE_URL)
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
    
    print(f"Starting generation of {NUM_EXAMPLES} examples...")
    
    tasks = [fetch_single_example(client, sem) for _ in range(NUM_EXAMPLES)]
    valid_examples = 0
    rows = []

    with tqdm(total=NUM_EXAMPLES, desc="Generating dataset") as pbar:
        for task in asyncio.as_completed(tasks):
            res = await task
            pbar.update(1)

            if res:
                try:
                    rows.append(json.loads(res))
                    valid_examples += 1
                except json.JSONDecodeError:
                    continue

    if not rows:
        detail = f" First request error: {API_ERRORS[0]}" if API_ERRORS else ""
        raise RuntimeError(
            "No examples were saved. Check the API key, BASE_URL, MODEL_NAME, and API availability." + detail
        )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Successfully saved {valid_examples} examples to {OUTPUT_FILE}.")


if __name__ == "__main__":
    asyncio.run(generate_data())
