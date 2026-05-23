# ICWSM 2026 — Continual Learning Tutorial

Companion code for the ICWSM 2026 tutorial on **Continual Learning for NLP**. It demonstrates Domain-Incremental Learning (Domain-IL) using LoRA adapters on the [SemEval-2016 Task 6](http://alt.qcri.org/semeval2016/task6/) Stance Detection dataset — a shared classification head is trained sequentially across various stance targets (HC, FM, CC, etc).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/human-info-lab/ICWSM-2026-Continual-Learning-Tutorial-Code/blob/main/notebooks/colab_setup.ipynb)

## Google Colab

Click the badge above to open the tutorial notebook in Colab. It walks through cloning the repo, installing dependencies, downloading data, training, and evaluation — no local setup required. Enable a GPU runtime first (**Runtime → Change runtime type → T4 GPU**).

## Quick start

```bash
uv run -m src.download          # download & preprocess data
uv run -m src.train             # sequential training (sft, default)
uv run -m src.evaluate          # build performance matrix & CL metrics
```

## Module overview

### `src/download.py`

Downloads `stancedataset.zip` from the SemEval-2016 Task 6 server, extracts the raw CSVs to `data/raw/StanceDataset/`, and writes the unified preprocessed file to `data/processed/stance_dataset.csv`.

- Train rows are 80/20 split into `train`/`valid` (stratified by target × stance).
- Original test rows form the `test` split.

### `src/data.py`

Reads the processed CSV, tokenizes tweets, and organises data into a `TaskList` — one `Task` per stance target in the fixed order `[HC, FM, CC, AT, AB, DT]`.

- `Task` extends `DatasetDict` with `.name` (e.g. `"Task_HC"`) and `.label2id` (`{"AGAINST": 0, "FAVOR": 1, "NONE": 2}`).
- Train rows from `data.py` are further 90/10 split into train/valid (stratified by `task_stance`); the original test split is kept as-is.
- Entry point: `load_data_dil(tokenizer) -> TaskList`.

### `src/metrics.py`

Two metric classes used by training and evaluation:

- **`MulticlassClassificationMetrics`** — wraps HuggingFace `evaluate` for accuracy, F1 (macro/micro), precision, and recall. Accepts either an `EvalPrediction` or raw `predictions`/`references`.
- **`ContinualLearningMetrics`** — takes a `TaskList` and a performance matrix and returns mean forgetting, final average accuracy/F1, per-task forgetting, snapshot accuracies, and best-ever accuracies. Also pretty-prints a summary table.

### `src/methods.py`

Defines the continual learning methods:

- **`CLMethod`** (base class) — interface with `setup`, `train_task`, and `after_task` hooks.
- **`SequentialFineTuning`** (`--method sft`) — naive Domain-IL baseline; trains on each task's data with no memory.
- **`ExperienceReplay`** (`--method er`) — replay with a fixed-size buffer built by reservoir sampling (Algorithm R). Buffer never exceeds `buffer_size` examples; at each task it concatenates the buffer with the current task's training data before calling the trainer.

The shared `train()` helper (also in this module) wraps `AdapterTrainer` with early stopping (patience 3, threshold 0.001) on `f1_macro`.

### `src/train.py`

Orchestrates sequential training across all tasks.

- Builds a single `AutoAdapterModel` with one LoRA adapter (`task-lora`) and one classification head (`task-head`) shared across all tasks.
- For each task: calls `method.train_task`, then `method.after_task`, then saves adapter + head to `output/{benchmark}/{method}/adapter-{task_name}/`.
- CLI options: `--model-name`, `--benchmark`, `--method` (`sft` or `er`), `--output-dir`, `--buffer-size`.

```bash
uv run -m src.train --method er --buffer-size 200
```

### `src/evaluate.py`

Builds the triangular **performance matrix** and computes CL metrics.

- For each saved adapter (task `t`), loads the model and evaluates it on all tasks `j ≤ t`, filling `perf_matrix[t][j]`.
- Passes the matrix to `ContinualLearningMetrics` for forgetting / final accuracy / F1 summary.
- Saves full results to `output/{benchmark}/{method}/eval_results.json`.
- CLI options: `--model-name`, `--benchmark`, `--method`, `--output-dir`.

```bash
uv run -m src.evaluate --method er
```

## Output layout

```
output/
└── {benchmark}/
    └── {method}/
        ├── adapter-Task_HC/        # LoRA adapter + head saved after task HC
        ├── adapter-Task_FM/
        ├── ...
        └── eval_results.json       # performance matrix + CL metrics
```

## Data paths

| Path | Contents |
|------|----------|
| `data/raw/StanceDataset/` | Raw CSVs from the zip |
| `data/processed/stance_dataset.csv` | Unified preprocessed dataset |
| `output/{benchmark}/{method}/adapter-{task}/` | Saved LoRA adapter + head |
| `output/{benchmark}/{method}/eval_results.json` | Final evaluation metrics |
