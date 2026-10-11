import argparse
import json
import math
from itertools import islice
from pathlib import Path

from tqdm import tqdm
from rich.console import Console
from rich.table import Table

from .tools import TOOLS, execute_tool, finite_number, parse_number, parse_response

SYSTEM_PROMPT = """
Rozwiąż zadanie matematyczne, korzystając z kalkulatora.

Każda twoja odpowiedź musi zawierać dokładnie jedno wywołanie narzędzia:
<tool_call>{"name": "add", "arguments": {"a": 3, "b": 5}}</tool_call>
albo końcową odpowiedź będącą samą liczbą.
Po wywołaniu narzędzia poczekaj na jego wynik.
Używaj tylko skończonych liczb rzeczywistych.

Dostępne narzędzia:
""" + "\n".join(
    f"{name}({', '.join(arguments)}): {description}"
    for name, (_, arguments, description) in TOOLS.items()
)


def load_dataset(path):
    from datasets import load_dataset as hf_load_dataset

    rows = hf_load_dataset("json", data_files=str(path), split="train")
    dataset = []
    for row in rows:
        sample = {
            "prompt": row.get("prompt", row.get("task")),
            "category": row["category"],
        }
        if row.get("id") is not None:
            sample["id"] = row["id"]
        if row.get("answer") is not None:
            sample["answer"] = (parse_number(row["answer"])
                                if isinstance(row["answer"], str)
                                else finite_number(row["answer"]))
        dataset.append(sample)
    return dataset


def load_model(chat_template_path, *, model_id=None, adapter_path=None):
    import torch
    from peft import PeftConfig
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if adapter_path is not None:
        adapter_path = Path(adapter_path)
        adapter_config = PeftConfig.from_pretrained(adapter_path)
        base_model = adapter_config.base_model_name_or_path
        tokenizer_source = adapter_path
    else:
        base_model = str(model_id)
        tokenizer_source = base_model

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    tokenizer.chat_template = Path(chat_template_path).read_text(encoding="utf-8")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map={"": 0},
        torch_dtype=dtype,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        ),
    )
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer, base_model


def generate_replies(model, tokenizer, conversations, *, max_new_tokens, max_model_len):
    import torch

    inputs = tokenizer.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        padding=True,
    )
    inputs.pop("token_type_ids", None)
    lengths = inputs["attention_mask"].sum(dim=1).tolist()
    valid = [i for i, length in enumerate(lengths) if length + max_new_tokens <= max_model_len]
    replies: list[str | Exception] = [ValueError("Conversation exceeds max-model-len; use a shorter prompt.")
               for _ in conversations]
    if not valid:
        return replies
    prompt_length = max(lengths[i] for i in valid)
    inputs = {key: value[valid, -prompt_length:].to(model.device) for key, value in inputs.items()}

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    for index, tokens in zip(valid, output[:, prompt_length:].tolist()):
        if tokenizer.eos_token_id in tokens:
            tokens = tokens[:tokens.index(tokenizer.eos_token_id)]
        replies[index] = tokenizer.decode(tokens, skip_special_tokens=False).strip()
    return replies


class Evaluator:
    def __init__(
        self,
        model,
        tokenizer,
        *,
        batch_size=4,
        call_budget=16,
        max_new_tokens=256,
        max_model_len=8192,
        tolerance=1e-6,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if call_budget < 0:
            raise ValueError("call_budget must be nonnegative")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if max_model_len < 1:
            raise ValueError("max_model_len must be positive")
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("tolerance must be finite and nonnegative")

        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.call_budget = call_budget
        self.max_new_tokens = max_new_tokens
        self.max_model_len = max_model_len
        self.tolerance = tolerance

    def evaluate(self, dataset):
        if not dataset:
            raise ValueError("Dataset must not be empty")
        has_ground_truth = all("answer" in row for row in dataset)
        samples = []
        for row in dataset:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row["prompt"]},
            ]
            sample = {
                **row,
                "prediction": None,
                "error": None,
                "messages": messages,
                "outputs": [],
                "calls": 0,
                "turns": 0,
            }
            if has_ground_truth:
                sample["correct"] = False

            samples.append(sample)

        pending, active = iter(samples), []
        with tqdm(total=len(samples), desc="Evaluating", unit="sample") as progress:
            while True:
                active.extend(islice(pending, self.batch_size - len(active)))
                if not active:
                    break
                try:
                    outputs = generate_replies(
                        self.model,
                        self.tokenizer,
                        [sample["messages"] for sample in active],
                        max_new_tokens=self.max_new_tokens,
                        max_model_len=self.max_model_len,
                    )
                except (TypeError, ValueError, RuntimeError) as error:
                    outputs = [error] * len(active)
                remaining = []
                for sample, output in zip(active, outputs):
                    try:
                        if isinstance(output, Exception):
                            raise output
                        sample["outputs"].append(output)
                        sample["turns"] = len(sample["outputs"])
                        sample["messages"].append({"role": "assistant", "content": output})
                        parsed = parse_response(output)
                        if not isinstance(parsed, dict):
                            sample["prediction"] = parsed
                            if has_ground_truth:
                                sample["correct"] = abs(parsed - sample["answer"]) <= self.tolerance
                            continue
                        if sample["calls"] >= self.call_budget:
                            raise ValueError(f"Tool call budget exhausted ({self.call_budget})")
                        result = execute_tool(parsed)
                        sample["calls"] += 1
                        sample["messages"].append({
                            "role": "tool",
                            "tool_call_id": f"call_{sample['calls']}",
                            "content": "WYNIK_NARZEDZIA " + json.dumps(
                                {"tool": parsed["name"], "result": result}, ensure_ascii=False
                            ),
                        })
                        remaining.append(sample)
                    except (TypeError, ValueError, RuntimeError) as error:
                        sample["error"] = str(error)
                progress.update(len(active) - len(remaining))
                active = remaining
        return summarize(samples) if has_ground_truth else None, samples


def summarize(samples):
    def metrics(group):
        correct = sum(sample["correct"] for sample in group)
        return {"total": len(group), "correct": correct, "accuracy": correct / len(group)}

    categories = sorted({sample["category"] for sample in samples})
    return {
        **metrics(samples),
        "by_category": {
            category: metrics([sample for sample in samples if sample["category"] == category])
            for category in categories
        },
    }


def save_outputs(output_dir, results, samples):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [("samples.json", samples)]
    if results is not None:
        outputs.append(("results.json", results))
    else:
        # Do not leave metrics from an earlier labeled run next to these predictions.
        (output_dir / "results.json").unlink(missing_ok=True)
    for name, payload in outputs:
        (output_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )


def print_results(results):
    table = Table()
    table.add_column("Category")
    table.add_column("Samples", justify="right")
    table.add_column("Correct", justify="right")
    table.add_column("Accuracy", justify="right")
    for category, metrics in results['by_category'].items():
        table.add_row(category, str(metrics['total']), str(metrics['correct']),
                      f"{metrics['accuracy']:.2%}")
    table.add_section()
    table.add_row("Total", str(results['total']), str(results['correct']),
                  f"{results['accuracy']:.2%}", style="bold cyan")
    Console().print(table)


def main():
    from dotenv import load_dotenv

    load_dotenv('.env')
    parser = argparse.ArgumentParser(description=__doc__)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", help="Hugging Face model ID or local model directory")
    model_group.add_argument("--adapter", type=Path, help="PEFT LoRA adapter directory")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4, help="Maximum active conversations per batch")
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--call-budget", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--chat-template", type=Path, default=Path("chat_template.jinja"))
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    print(f"Loading dataset from {args.dataset}")
    dataset = load_dataset(args.dataset)
    print(f"Loaded {len(dataset)} samples from {args.dataset}")
    if any("answer" not in row for row in dataset):
        print("Dataset has missing reference answers. Saving predictions without metrics.")
    if args.adapter is not None:
        print(f"Loading adapter from {args.adapter}")
        model, tokenizer, base_model = load_model(
            args.chat_template,
            adapter_path=args.adapter,
        )
    else:
        print(f"Loading full model {args.model}")
        model, tokenizer, base_model = load_model(
            args.chat_template,
            model_id=args.model,
        )

    evaluator = Evaluator(
        model,
        tokenizer,
        batch_size=args.batch_size,
        call_budget=args.call_budget,
        max_new_tokens=args.max_new_tokens,
        max_model_len=args.max_model_len,
        tolerance=args.tolerance,
    )
    metrics, samples = evaluator.evaluate(dataset)
    results = None if metrics is None else {
        "model": base_model,
        "adapter": str(args.adapter) if args.adapter is not None else None,
        "dataset": str(args.dataset),
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key not in {"model", "adapter", "dataset", "output_dir"}
        },
        **metrics,
    }
    save_outputs(args.output_dir, results, samples)
    if results is not None:
        print_results(results)
    print(f"Outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
