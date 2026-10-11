# Evaluating LLMs on Athena: a practical introduction to Slurm

This tutorial takes you through a complete experiment on an HPC cluster: prepare data, evaluate a base language model, fine-tune it, and evaluate the resulting adapter. The case study is teaching PLLuM to use a calculator to solve Polish word problems. Along the way, you will learn to request a GPU, submit jobs, follow their progress, diagnose failures, and keep your results.

## 1. Before you start

This is an introductory tutorial and no previous HPC experience is required. You should be comfortable navigating directories, editing files, and running scripts in a Linux terminal. You should also have a good command of Python. A general understanding of a language model and a benchmark is sufficient. Bring a laptop with an SSH client installed.

On Linux and macOS, use your usual terminal. On Windows, use PowerShell with OpenSSH or a terminal in WSL. Verify locally:

```bash
ssh -V
```

The instructor will provide your tutorial SSH username, authentication instructions, the reservation's active time window, and LLMLab credentials if data generation is included.

Replace `YOUR_TUTORIAL_LOGIN` in the commands below with the supplied username. Passwords and API keys should stay out of commits and screenshots.

The tutorial follows these steps:

1. Connect to Athena through VS Code Remote SSH and clone the repository.
2. Run an interactive GPU check and submit a short batch job.
3. Review the model task and repository structure.
4. Create the Python environment in a batch job.
5. Prepare training data and inspect the benchmark.
6. Measure the base model's performance.
7. Train a LoRA adapter and evaluate it under the same conditions.
8. Inspect failures, record what changed, and submit predictions to the leaderboard.


## 2. Set up VS Code Remote SSH

We recommend you to use Visual Studio Code to perform the tutorial.

Install [Visual Studio Code](https://code.visualstudio.com/) on your laptop and Microsoft's **Remote - SSH** extension (`ms-vscode-remote.remote-ssh`). It is also included in the Remote Development extension pack. The editor will open files directly on Athena, without a separate local copy. See the [Remote SSH guide](https://code.visualstudio.com/docs/remote/ssh).

First, check that authentication works from your laptop's terminal:

```bash
ssh YOUR_TUTORIAL_LOGIN@athena.cyfronet.pl
exit
```

In VS Code, open the Command Palette with Ctrl+Shift+P on Windows/Linux or Cmd+Shift+P on macOS. Select `Remote-SSH: Open SSH Configuration File...` and choose your local SSH configuration. Add this entry, replacing the username:

```sshconfig
Host athena-tutorial
    HostName athena.cyfronet.pl
    User YOUR_TUTORIAL_LOGIN
```

Select `Remote-SSH: Connect to Host...`, choose `athena-tutorial`, and enter your password when prompted. Choose **Linux** if asked for the remote platform. The first connection installs VS Code Server on Athena. Open **Terminal → New Terminal** to run the workshop commands. Connection and installation details are covered in the [Remote SSH guide](https://code.visualstudio.com/docs/remote/ssh).

If authentication appears to hang, enable `remote.SSH.showLoginTerminal` in VS Code settings to reveal password or other authentication prompts. Check **View → Output → Remote - SSH** for connection errors. See the [troubleshooting guide](https://code.visualstudio.com/docs/remote/troubleshooting#troubleshooting-hanging-or-failing-connections).

Use the remote editor to modify scripts and the terminal to submit Slurm jobs. VS Code's Run and Debug actions do not request a Slurm allocation, so run GPU workloads through the commands in this guide.

## 3. Where your code will run

Athena is Cyfronet's GPU supercomputer. Its documented GPU partition, `plgrid-gpu-a100`, has 48 nodes, each with 128 CPU cores, about 1 TB of host RAM, and eight NVIDIA A100 GPUs with 40 GB of device memory each. The partition's documented maximum job duration is 48 hours. This lab uses one GPU per job. See the [Athena documentation](https://docs.hpc.cyfronet.pl/supercomputers/athena/) for current hardware and policies.

When you connect to Athena through SSH, your terminal session starts on a **login node**. This shared environment is used to prepare scripts, manage files, submit jobs, and inspect results. Computational workloads run on separate **compute nodes**, where Slurm allocates the CPU, memory, and GPU resources requested for each job. In this tutorial, model loading, training, inference, and environment setup run on compute nodes within Slurm allocations. Athena is dedicated to GPU workloads.


For regular access after the tutorial, you need a PLGrid computing grant and access to the Athena service. During the event, use the credentials and allocation supplied by the organizers. The [accounts and grants guide](https://docs.hpc.cyfronet.pl/environment/batch-system/accounts-and-grants/) explains account names, including Athena's `-gpu-a100` suffix.

## 4. Prepare your remote workspace

Use the remote terminal opened in VS Code. If you prefer a standalone terminal, connect from your laptop with:

```bash
ssh YOUR_TUTORIAL_LOGIN@athena.cyfronet.pl
```

Enter the supplied password if prompted. Follow any additional authentication instructions from the instructor. Once connected, check your identity and storage:

```bash
whoami
hostname
printf 'Home: %s\nScratch: %s\n' "$HOME" "$SCRATCH"
hpc-fs
hpc-grants
```

`hpc-fs` reports filesystem usage and quotas. `hpc-grants` lists computing allocations. If `$SCRATCH` is empty or the event allocation is unavailable, ask the instructor before submitting jobs.

Keep source code in your home directory and large temporary artifacts in scratch:

```bash
git clone https://github.com/NASK-NLP/PLLuM-SLURM-Tutorial.git
cd PLLuM-SLURM-Tutorial
pwd
```

In the remote VS Code window, select **File → Open Folder...** and choose the repository path printed by `pwd`. Open a terminal in that folder for the remaining exercises. Remote folders and terminals are described in the [Remote SSH guide](https://code.visualstudio.com/docs/remote/ssh).

Scratch is temporary. Cyfronet documents automatic deletion of scratch data older than 30 days and job directories older than seven days. Prefer supplied storage variables over hard-coded filesystem paths. See [Athena storage](https://docs.hpc.cyfronet.pl/supercomputers/athena/) and [environment variables](https://docs.hpc.cyfronet.pl/environment/variables/).

## 5. Understand how Slurm runs your work

Athena's compute nodes are shared by many users. Slurm coordinates access to these resources so that each program runs with an assigned set of CPUs, memory, and GPUs. From the login node, you submit a **job** that specifies both the command to run and the resources it needs. For example, a training job in this tutorial requests one GPU, eight CPUs, and a maximum running time.

Submitting a job gives you a job ID that you can use to track it. The job may first enter a queue while Slurm waits for suitable resources and checks scheduling priorities and access rules. Once the resources are available, Slurm creates an **allocation** and starts your program on a compute node. When the job finishes, those resources become available to other jobs. Acceptance of a submission therefore does not mean the program has started or completed.

Several settings determine where and under which conditions a job can run:

- A **partition** is a group of compute nodes with scheduling rules, such as allowed job durations. This tutorial uses Athena's `plgrid-gpu-a100` partition.
- An **account** identifies the computing grant charged for the job. It is separate from the username you use to log in through SSH.
- A **reservation** makes resources available to authorized users during a specified period. The workshop reservation gives participants access to the event's reserved resources. Each job still needs to request its CPUs, memory, GPUs, and time.
- The **wall-time limit** is the maximum time a job can run after it starts. Waiting in the queue does not count toward this limit. Slurm terminates a job that exceeds it, so choose a limit appropriate for the experiment and the event's reservation window.

You will use Slurm in two ways. A **batch job** runs a script without requiring an open terminal, which makes it suitable for installation, training, and evaluation. An **interactive job** gives you a terminal within an allocation, which is useful for checking GPU access and debugging. Both use allocated compute resources and are subject to scheduling and time limits.

The main commands you will encounter are:

| Command | Purpose |
| --- | --- |
| `sbatch` | Submit a batch script and obtain a job ID |
| `srun` | Launch a task, including an interactive shell |
| `squeue` | Check active jobs and why pending jobs are waiting |
| `scontrol show job JOB_ID` | Inspect a job's request, allocation, and paths |
| `scancel JOB_ID` | Cancel a specific job |

Slurm provides the resources and controls execution. Your Python code determines how those resources are used. Requesting several GPUs does not automatically make a single-GPU program run in parallel. The next exercises cover interactive GPU access, batch submission, and monitoring. See the official [`sbatch`](https://slurm.schedmd.com/sbatch.html), [`srun`](https://slurm.schedmd.com/srun.html), and [`squeue`](https://slurm.schedmd.com/squeue.html) references for details.

## 6. Try an interactive GPU allocation

An interactive job helps you check the environment and investigate problems. From the login node:

```bash
srun --account=tutorial --partition=tutorial --reservation=tutorial-llm \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
  --time=00:10:00 --pty bash
```

Once the compute-node shell opens:

```bash
hostname
printf 'Job: %s\nVisible GPUs: %s\n' "$SLURM_JOB_ID" "${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi
exit
```

Compare this hostname with the login node's hostname. CUDA can expose the allocated device as GPU `0` even when its physical index differs. `exit` ends the interactive shell and releases the allocation. You return to the login node.

## 7. Submit a short batch job

Run the same GPU check as a batch job. From the login node, submit:

```bash
sbatch --job-name=gpu-check --output=gpu-check-%j.out \
  --account=tutorial --partition=tutorial --reservation=tutorial-llm \
  --nodes=1 --ntasks=1 --cpus-per-task=1 --gres=gpu:1 \
  --time=00:01:00 --wrap='nvidia-smi'
```

`--wrap` submits a command without a separate job script. Slurm prints `Submitted batch job JOB_ID`. Replace `JOB_ID` below with that number:

```bash
squeue -j JOB_ID
scontrol show job JOB_ID
```

Inspect the requested resources and output path. In `squeue`, `PD` means pending and `R` means running. This check finishes quickly, so it may already have left the queue when you inspect it. Read its output from the directory where you submitted it:

```bash
less gpu-check-JOB_ID.out
```

Compare the GPU information with the interactive check. Both commands run within a GPU allocation, but the batch job saves its output to a file and does not require an interactive session.

## 8. Your task

Our experiment focuses on solving Polish mathematical word problems with calculator tools. Given a problem description, PLLuM should identify the required operations, choose their arguments, and produce a sequence of tool calls leading to the correct numerical answer. We will prepare examples of this behavior, fine-tune the model, and evaluate its solutions on a separate set of problems.

Tool calling separates selecting an operation from executing it. The model decides what to calculate, while the application runs the corresponding Python function and returns its result. For multistep problems, the model must use intermediate results to choose the next operation. A calculator ensures that the selected operation is executed correctly, but choosing the right sequence remains the model's task.

### Tools and response format

The calculator functions are defined in `tools.py`:

| Tool | Calculation |
| --- | --- |
| `add(a, b)` | `a + b` |
| `subtract(a, b)` | `a - b` |
| `multiply(a, b)` | `a * b` |
| `divide(a, b)` | `a / b` |
| `modulo(a, b)` | `a % b` |
| `power(a, b)` | `a ** b` |
| `absolute_value(a)` | `abs(a)` |
| `square_root(a)` | `math.sqrt(a)` |

Each assistant turn contains one tool call in the following format, with no additional text:

```text
<tool_call>{"name": "<tool_name>", "arguments": {"<arg1>": <value1>, "<arg2>": <value2>}}</tool_call>
```

Use the exact function and argument names listed above. Arguments must be finite numbers, and calls must produce valid real-valued results. After each call, the application adds a tool response to the conversation before asking the model for its next step.

The last assistant message supplies only a finite number, without an explanation, units, or a JSON wrapper. In the source dataset, this is stored as a string such as `"4"`. Both training and evaluation accept a decimal point or comma and scientific notation, for example `3.5`, `3,5`, or `1e-3`. Text such as `4 albo 5` and JSON such as `{"answer": 4}` are rejected.

### Example: a problem requiring two operations

Consider the question: “Jaś ma 3 jabłka, a Ania ma 5 jabłek. Ile średnio przypada jabłek na osobę?” To calculate the average number of apples per person, the model first requests `add(3, 5)`. After receiving `8`, it requests `divide(8, 2)` and receives `4`.

The corresponding training conversation is stored as follows. The function names are `add` and `divide`, matching the current implementation:

```json
[
  {
    "role": "user",
    "content": "Jaś ma 3 jabłka, a Ania ma 5 jabłek. Ile średnio przypada jabłek na osobę?"
  },
  {
    "role": "assistant",
    "content": null,
    "tool_calls": [
      {
        "id": "call_1",
        "type": "function",
        "function": {
          "name": "add",
          "arguments": "{\"a\": 3, \"b\": 5}"
        }
      }
    ]
  },
  {
    "role": "tool",
    "tool_call_id": "call_1",
    "name": "add",
    "content": "8"
  },
  {
    "role": "assistant",
    "content": null,
    "tool_calls": [
      {
        "id": "call_2",
        "type": "function",
        "function": {
          "name": "divide",
          "arguments": "{\"a\": 8, \"b\": 2}"
        }
      }
    ]
  },
  {
    "role": "tool",
    "tool_call_id": "call_2",
    "name": "divide",
    "content": "4"
  },
  {
    "role": "assistant",
    "content": "4"
  }
]
```

The `tool_call_id` links each result to the assistant's request. Before training, `training/train.py` validates the calls and recalculates their results. The chat template in `chat_template.jinja` then renders the structured calls as `<tool_call>` blocks and the tool results as `<tool_response>` blocks. This keeps the conversation format consistent between training and evaluation.

## 9. Repository overview

The repository implements the experiment in three stages: preparing tool-calling conversations, training a PLLuM adapter, and evaluating its answers. Python files implement each stage, while the accompanying shell scripts submit the GPU workloads to Slurm.

| File or directory | Role in the experiment |
| --- | --- |
| `setup_env.sh`, `requirements.txt` | Create the Python environment and install dependencies |
| `.env.example` | Template for LLMLab and Hugging Face credentials in your local `.env` |
| `tools.py` | Define the calculator functions used for data validation and tool execution |
| `data_generation/` | Generate training conversations through LLMLab |
| `data/` | Store prepared training conversations and evaluation problems |
| `chat_template.jinja` | Render conversations in the model's input/output format |
| `training/train.py`, `training/train.sh` | Train the adapter and submit training to Slurm |
| `evaluation/evaluate.py`, `evaluation/evaluate.sh` | Evaluate answers and submit evaluation to Slurm |

Submit the shell scripts from the repository root. Slurm logs are written to `out/`, while the environment and model artifacts are stored in scratch:

| Location | Contents |
| --- | --- |
| `out/` | Slurm stdout/stderr logs |
| `$SCRATCH/pllum-slurm-lab/.venv` | Python environment |
| `$SCRATCH/pllum-slurm-lab/huggingface` | Model and dataset cache |
| `$SCRATCH/pllum-slurm-lab/adapter` | Trained adapter and tokenizer |
| `$SCRATCH/pllum-slurm-lab/evaluation*` | Evaluation metrics and conversation traces |

## 10. Create the Python environment

From the repository root, submit the environment setup job:

```bash
mkdir -p out
sbatch setup_env.sh
```

The script creates a Python 3.12 virtual environment in `$SCRATCH/pllum-slurm-lab/.venv` and installs `requirements.txt`, including PyTorch, Transformers, PEFT, and bitsandbytes. Use the job ID printed by Slurm to check its status:

```bash
squeue -j JOB_ID
```

Once the job starts, follow its log:

```bash
tail -f out/setup_env_slurm-pllum-setup-JOB_ID.out
```

Wait for the `Environment ready:` message before using the environment. Press Ctrl+C to stop following the log.

## 11. Generate training data

We will use a larger model through LLMLab to generate examples for PLLuM. Each example contains a Polish word problem, the tool calls needed to solve it, their results, and the final answer. These complete conversations provide the supervision for both choosing operations and responding to tool results.

Copy the credential template from the repository root:

```bash
cp .env.example .env
```

Open `.env` in your editor and set `LLMLAB_API_KEY` to the credential supplied for the tutorial. You can also set `HF_TOKEN` for authenticated Hugging Face model downloads; otherwise, leave it empty. The generator, training script, and evaluator load `.env` from the repository root.

The generator's current configuration is:

| Setting | Value |
| --- | --- |
| `BASE_URL` | `https://llmlab.plgrid.pl/api/v1` |
| `MODEL_NAME` | `Qwen/Qwen3.5-122B-A10B` |
| `NUM_EXAMPLES` | 200 |
| `CONCURRENCY_LIMIT` | 15 |
| `OUTPUT_FILE` | `data/tools_dataset.jsonl` |

Read `get_system_prompt()` before running the generator. It asks for problems requiring one to three operations and selects a target tool for each example. You can change the contexts or wording to explore how the generated data affects the model, while preserving the conversation format and function names.

After environment setup finishes, run from the repository root:

```bash
source "$SCRATCH/pllum-slurm-lab/.venv/bin/activate"
python data_generation/generate_dataset.py
```

Generation uses remote inference through LLMLab and does not require a local GPU. The script saves one conversation per line in a JSONL file. Failed requests or invalid JSON can reduce the number of saved examples. Repeating generation replaces the output file, so preserve any dataset you want to keep.

Inspect the number of examples and one complete conversation:

```bash
wc -l data/tools_dataset.jsonl
head -n 1 data/tools_dataset.jsonl | python -m json.tool
```

Check whether the question supplies enough information, the tool sequence solves the stated problem, and the final answer agrees with the tool results. The generator checks JSON syntax, but it does not validate arithmetic. Training later recalculates the tool results and skips conversations with invalid calls, inconsistent results, or final answers that are not finite numbers. This still leaves the interpretation of the question and the correctness of the final answer for you to inspect.

## 12. Train the model

Supervised fine-tuning teaches the model to produce the assistant responses shown in the training conversations. Here, the targets include tool calls and final answers. User messages and tool results provide context but are excluded from the training loss.

We use LoRA to train small adapter matrices while keeping the base model weights frozen. The base model is loaded in four-bit precision to reduce memory use, a combination commonly called QLoRA. The resulting adapter must be loaded together with the original base model. See the [PEFT quantization guide](https://huggingface.co/docs/peft/developer_guides/quantization) for details.

`training/train.sh` uses `CYFRAGOVPL/PLLuM-4B-instruct-2512` and reads the included example dataset, `data/train.jsonl`, by default. To train on the conversations generated in section 11, select that file explicitly:

```bash
sbatch training/train.sh --data-file data/tools_dataset.jsonl
```

The training configuration in `training/train.py` is:

| Setting | Value |
| --- | --- |
| Epochs | 2 passes over the training split |
| Validation split | 10% of valid conversations |
| Maximum sequence length | 2048 tokens |
| Batch size | 4 conversations per GPU |
| Gradient accumulation | 4 microbatches per optimizer update (16 conversations total) |
| Learning rate | `2e-4` |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |

Gradient accumulation combines several small batches before updating the adapter, increasing the effective batch size without holding all examples in memory at once. The validation split is used to monitor performance on conversations excluded from training. It is separate from the benchmark used in the next section.

Submit training from the repository root:

```bash
sbatch training/train.sh
```

For a short run that checks the pipeline, use this command instead:

```bash
sbatch training/train.sh --limit 128 --max-steps 10
```

`--limit` caps the number of valid conversations before splitting. `--max-steps` limits optimizer updates and takes precedence over the epoch count. This short run checks that training and saving work, but it is too small to draw conclusions about model quality.

Check the job and follow its log using the ID printed by Slurm:

```bash
squeue -j JOB_ID
tail -f out/training_slurm-pllum-train-JOB_ID.out
```

The log reports skipped conversations, split sizes, trainable parameters, and training and validation losses. Lower training loss means the model fits the examples better. Benchmark evaluation will show whether this translates into better answers on separate problems.

After training finishes without errors, inspect the adapter:

```bash
ls -lh "$SCRATCH/pllum-slurm-lab/adapter"
```

Expect `adapter_config.json`, adapter weights, and tokenizer files. The script saves at the end without intermediate checkpoints. Repeated runs write to the same directory, so preserve an adapter before replacing it and avoid concurrent training jobs that share this output path.

To vary the experiment, pass options such as `--epochs 1` or `--limit 500` after `training/train.sh`. Changing the learning rate or LoRA rank requires editing `training/train.py`.

## 13. Evaluate the model

Evaluation tests whether the trained model can solve problems outside its training conversations. For each problem, the evaluator generates a response, executes any tool call, and feeds the result back into the conversation. It repeats this process until the model returns an answer or the interaction fails.

Start with the ten problems in `data/example.test.json`:

```bash
sbatch evaluation/evaluate.sh
```

The script loads the saved adapter and its base model. Check progress in the evaluation log:

```bash
squeue -j JOB_ID
tail -f out/eval_slurm-pllum-eval-JOB_ID.out
```

The evaluator uses greedy generation, allows up to 16 tool calls per problem, and compares the final answer with the reference using an absolute tolerance of `1e-6`. Invalid calls, arithmetic errors, and exhausted call budgets count as unsuccessful samples. The log prints overall and per-category accuracy.

Two files are saved in `$SCRATCH/pllum-slurm-lab/evaluation`:

| File | Contents |
| --- | --- |
| `results.json` | Model and dataset identifiers, evaluation settings, sample counts, and accuracy |
| `samples.json` | Questions, reference answers, predictions, generated responses, tool interactions, and errors |

Read the results and inspect individual conversations:

```bash
python -m json.tool "$SCRATCH/pllum-slurm-lab/evaluation/results.json"
less "$SCRATCH/pllum-slurm-lab/evaluation/samples.json"
```

For incorrect answers, determine whether the model chose the wrong operation, supplied incorrect arguments, produced malformed output, or stopped before completing the calculation. A correctly executed tool can still be part of an incorrect solution. Accuracy measures the final numerical answer and does not require a particular tool sequence or even a tool call.

The ten-problem set is a quick pipeline check. The larger public benchmark, `data/test.samples.json`, contains 295 problems without reference answers and is used for leaderboard submissions (section 14). Its categories are `easy` (99), `medium` (80), `hard` (57), and `challanging` (59); `challanging` is the category label stored in the dataset. The reference answers remain private, so local accuracy is available only for `data/example.test.json`.

To measure the effect of fine-tuning, evaluate the base model on the same problems. Open a GPU allocation using the interactive command from section 6, then run from the repository root:

```bash
source "$SCRATCH/pllum-slurm-lab/.venv/bin/activate"
export HF_HOME="$SCRATCH/pllum-slurm-lab/huggingface"
python -m evaluation.evaluate \
  --model CYFRAGOVPL/PLLuM-4B-instruct-2512 \
  --dataset data/example.test.json \
  --output-dir "$SCRATCH/pllum-slurm-lab/evaluation-base"
exit
```

Choose an interactive time limit long enough for this evaluation. The base-model command uses the same default evaluation settings as the adapter job. The wrapper `evaluation/evaluate.sh` always selects an adapter, so do not append `--model` to it.

Compare overall accuracy, category scores, and several failed conversations. Keep the benchmark and generation settings fixed across runs. Record the dataset, training parameters, and job IDs with your results so that any observed change can be traced to the experiment you ran.


## 14. Submit predictions to the leaderboard

Run the trained adapter on the public benchmark from the repository root:

```bash
sbatch evaluation/evaluate.sh \
  --dataset data/test.samples.json \
  --output-dir "$SCRATCH/pllum-slurm-lab/evaluation-lb"
```

The benchmark contains 295 problems with unique IDs and no reference answers. The evaluator saves predictions, IDs, conversation traces, and any errors to `$SCRATCH/pllum-slurm-lab/evaluation-lb/samples.json`. It does not produce accuracy metrics or a `results.json` file for this dataset.

After the job finishes, submit **only `samples.json`** to the leaderboard using the submission instructions supplied by the instructor.
