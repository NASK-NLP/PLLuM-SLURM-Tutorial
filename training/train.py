import argparse
import inspect
import json
import math
import os
import random
from pathlib import Path
REPO_DIR = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO_DIR))
import tools as calculator
from evaluation.tools import parse_number

CALCULATOR = {
    name: (function, tuple(inspect.signature(function).parameters))
    for name, function in vars(calculator).items()
    if inspect.isfunction(function) and function.__module__ == calculator.__name__
}

TOOL_LIST = '\n'.join(
    f"{name}({', '.join(arguments)}): {inspect.getdoc(function)}"
    for name, (function, arguments) in CALCULATOR.items()
)
SYSTEM = f'''Jesteś polskim asystentem używającym kalkulatora.
Każde działanie wykonuj narzędziem. Zwracaj jedno wywołanie narzędzia albo odpowiedź końcową na turę.
Wywołanie: <tool_call>{{"name": "add", "arguments": {{"a": 3, "b": 4}}}}</tool_call>
Odpowiedź końcowa: 7
Dostępne funkcje:
{TOOL_LIST}
W zadaniach wieloetapowych wywołuj funkcje po kolei i czekaj na <tool_response> przed dalszymi obliczeniami.
Nie zgaduj brakujących danych.
'''


def execute(call):
    try:
        if not isinstance(call, dict) or set(call) != {'tool', 'arguments'}:
            raise ValueError('Expected tool and arguments fields.')
        name, arguments = call['tool'], call['arguments']
        if name not in CALCULATOR:
            raise ValueError('Unknown tool.')
        function, expected_args = CALCULATOR[name]
        if not isinstance(arguments, dict) or set(arguments) != set(expected_args):
            raise ValueError('Invalid arguments.')
        values = {}
        for key in expected_args:
            value = arguments[key]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError('Arguments must be finite numbers.')
            values[key] = value
        result = function(**values)
        if isinstance(result, complex) or not math.isfinite(result):
            raise ValueError('Result is not a finite real number.')
        result = round(result, 8)
        if result == int(result):
            result = int(result)
        return {'result': result}
    except (ArithmeticError, TypeError, ValueError, OverflowError) as error:
        return {'error': str(error)}


def tool_message(name, result):
    payload = json.dumps({'tool': name, **result}, ensure_ascii=False, allow_nan=False)
    return {'role': 'tool', 'name': name, 'content': 'WYNIK_NARZEDZIA ' + payload}

def convert_conversation(row):
    source = row.get('messages')
    if not isinstance(source, list) or not source or source[0].get('role') != 'user':
        raise ValueError('Expected a messages list starting with a user message.')

    messages = [{'role': 'system', 'content': SYSTEM}]
    pending = {}
    for message in source:
        role = message.get('role')
        if role == 'user':
            messages.append({'role': 'user', 'content': message['content']})
        elif role == 'assistant' and message.get('tool_calls'):
            calls = message['tool_calls']
            if not isinstance(calls, list) or len(calls) != 1:
                raise ValueError('Expected exactly one tool call per assistant message.')
            call = calls[0]
            function = call['function']
            arguments = function['arguments']
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            action = {'tool': function['name'], 'arguments': arguments}
            if not isinstance(call.get('id'), str):
                raise ValueError('Tool call is missing its id.')
            pending[call['id']] = action
            messages.append({
                'role': 'assistant',
                'content': '',
                'tool_calls': [{
                    'id': call['id'],
                    'type': 'function',
                    'function': {'name': function['name'], 'arguments': arguments},
                }],
            })
        elif role == 'tool':
            call_id = message.get('tool_call_id')
            if call_id not in pending:
                raise ValueError('Tool result has no matching tool call.')
            action = pending.pop(call_id)
            if message.get('name') != action['tool']:
                raise ValueError('Tool result name does not match its call.')
            generated_result = json.loads(message['content'])
            if not isinstance(generated_result, dict):
                generated_result = {'result': generated_result}
            actual_result = execute(action)
            if set(actual_result) == {'result'} and set(generated_result) == {'result'}:
                if not math.isclose(float(actual_result['result']), float(generated_result['result']),
                                    rel_tol=1e-6, abs_tol=1e-6):
                    raise ValueError('Generated tool result does not match the calculator.')
            elif actual_result != generated_result:
                raise ValueError('Generated tool result does not match the calculator.')
            messages.append(tool_message(action['tool'], actual_result))
        elif role == 'assistant' and isinstance(message.get('content'), str):
            parse_number(message['content'])
            messages.append({'role': 'assistant', 'content': message['content'].strip()})
        else:
            raise ValueError(f'Unsupported message role or format: {role!r}')

    if pending:
        raise ValueError('Conversation has tool calls without results.')
    if messages[-1]['role'] != 'assistant' or messages[-1].get('tool_calls'):
        raise ValueError('Conversation must end with a numerical assistant answer.')
    return {'messages': messages}


def prepare_examples(rows, tokenizer, max_length):
    examples = []
    for row in rows:
        encoded = tokenizer.apply_chat_template(
            row['messages'], tokenize=True, return_dict=True,
            return_assistant_tokens_mask=True,
        )
        if len(encoded['input_ids']) > max_length:
            raise ValueError(f"Example has {len(encoded['input_ids'])} tokens; increase --max-length.")
        if not any(encoded['assistant_masks']):
            raise ValueError('Chat template produced no assistant tokens to train on.')
        examples.append({
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'labels': [token if assistant else -100 for token, assistant in
                       zip(encoded['input_ids'], encoded['assistant_masks'])],
        })
    return examples


def main():
    from dotenv import load_dotenv

    load_dotenv('.env')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--data-file', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--adapter-dir', type=Path, required=True)
    parser.add_argument('--validation-ratio', type=float, default=0.1)
    parser.add_argument('--epochs', type=float, default=2)
    parser.add_argument('--max-steps', type=int, default=-1)
    parser.add_argument('--max-length', type=int, default=2048)
    parser.add_argument('--limit', type=int, default=0, help='Number of conversations to use; 0 = all')
    args = parser.parse_args()
    os.environ.setdefault('HF_HOME', str(args.run_dir / 'huggingface'))

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if not torch.cuda.is_available():
        raise SystemExit('Training requires an NVIDIA GPU with CUDA.')
    if not 0 < args.validation_ratio < 1:
        raise SystemExit('--validation-ratio must be between 0 and 1.')
    rows = []
    skipped = 0
    if not args.data_file.is_file():
        raise SystemExit(f'Dataset not found: {args.data_file}')
    with args.data_file.open(encoding='utf-8') as file:
        for line_number, line in enumerate(file, 1):
            try:
                rows.append(convert_conversation(json.loads(line)))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                skipped += 1
                print(f'Skipping data row {line_number}: {error}')
    if args.limit:
        rows = rows[:args.limit]
    if len(rows) < 2:
        raise SystemExit(f'Need at least two valid conversations in {args.data_file}; found {len(rows)}.')
    random.Random(42).shuffle(rows)
    validation_count = max(1, round(len(rows) * args.validation_ratio))
    validation_rows, train_rows = rows[:validation_count], rows[validation_count:]
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.chat_template = (REPO_DIR / 'chat_template.jinja').read_text(encoding='utf-8')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    datasets = {
        'train': prepare_examples(train_rows, tokenizer, args.max_length),
        'validation': prepare_examples(validation_rows, tokenizer, args.max_length),
    }
    print(f'Data: {args.data_file} ({len(rows)} conversations, {skipped} skipped)')
    for split, examples in datasets.items():
        print(split, len(examples), 'assistant decisions')

    args.adapter_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map={'': 0}, torch_dtype=dtype,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype),
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], task_type='CAUSAL_LM'))
    model.print_trainable_parameters()

    def collate(batch):
        length = max(len(row['input_ids']) for row in batch)
        return {key: torch.tensor([row[key] + [pad] * (length - len(row[key])) for row in batch])
                for key, pad in [('input_ids', tokenizer.pad_token_id), ('attention_mask', 0), ('labels', -100)]}

    trainer = Trainer(
        model=model, train_dataset=datasets['train'], eval_dataset=datasets['validation'],
        data_collator=collate,
        args=TrainingArguments(
            output_dir=str(args.adapter_dir), num_train_epochs=args.epochs, max_steps=args.max_steps,
            per_device_train_batch_size=4, per_device_eval_batch_size=4,
            gradient_accumulation_steps=4, learning_rate=2e-4, warmup_ratio=0.03,
            bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
            optim='paged_adamw_8bit', logging_steps=10, eval_strategy='epoch',
            save_strategy='no', report_to='none', seed=42,
        ),
    )
    trainer.train()
    args.adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.adapter_dir))
    tokenizer.save_pretrained(args.adapter_dir)


if __name__ == '__main__':
    main()
