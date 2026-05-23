from __future__ import annotations

import random
from pathlib import Path

from adapters import AdapterTrainer, ModelWithFlexibleHeadsAdaptersMixin
from datasets import Dataset, concatenate_datasets
from transformers import (
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.training_args import TrainingArguments

from src.data import Task, TaskList
from src.metrics import MulticlassClassificationMetrics

AdaptersModel = ModelWithFlexibleHeadsAdaptersMixin | PreTrainedModel

_compute_metrics = MulticlassClassificationMetrics()


def train(
    model: AdaptersModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset,
    valid_dataset,
    output_dir: Path | str | None = None,
    num_train_epochs: int = 20,
):
    if output_dir is not None:
        output_dir = Path(output_dir)
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=1e-4,
        num_train_epochs=num_train_epochs,
        report_to="none",
        eval_strategy="epoch",
        save_strategy="best",
        load_best_model_at_end=True,
        push_to_hub=False,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
    )
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="longest",
        return_tensors="pt",
    )
    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3, early_stopping_threshold=0.001
            )
        ],
        compute_metrics=_compute_metrics,
    )
    trainer.train()


class CLMethod:
    """Base class for continual learning methods."""

    @property
    def name(self) -> str:
        raise NotImplementedError

    def setup(self, model: AdaptersModel, tasks: TaskList) -> None:
        """One-time setup called before the training loop."""
        pass

    def train_task(
        self,
        model: AdaptersModel,
        tokenizer: PreTrainedTokenizerBase,
        task: Task,
        task_idx: int,
        output_dir: Path,
    ) -> None:
        raise NotImplementedError

    def after_task(
        self,
        model: AdaptersModel,
        task: Task,
        task_idx: int,
    ) -> None:
        """Hook called after training completes for a task."""
        pass


class SequentialFineTuning(CLMethod):
    """Naive sequential fine-tuning — the Domain-IL baseline."""

    @property
    def name(self) -> str:
        return "sft"

    def train_task(self, model, tokenizer, task, task_idx, output_dir):
        train(
            model=model,
            tokenizer=tokenizer,
            train_dataset=task["train"],
            valid_dataset=task["valid"],
            output_dir=output_dir,
        )


class ExperienceReplay(CLMethod):
    """Experience replay with a fixed memory budget via reservoir sampling (Algorithm R).

    Each incoming example is accepted with probability M/k (k = total seen so far),
    replacing a random existing entry when the buffer is full. The buffer never exceeds
    buffer_size examples and never retains references to past full datasets.
    """

    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self._buffer: list[dict] = []
        self._total_seen: int = 0

    @property
    def name(self) -> str:
        return f"er-{self.buffer_size}"

    def after_task(self, model, task, task_idx):
        # Algorithm for reservoir sampling
        for i in range(len(task["train"])):
            self._total_seen += 1
            example = task["train"][i]
            if len(self._buffer) < self.buffer_size:
                self._buffer.append(example)
            else:
                j = random.randrange(self._total_seen)
                if j < self.buffer_size:
                    self._buffer[j] = example

    def train_task(self, model, tokenizer, task, task_idx, output_dir):
        if self._buffer:
            keys = self._buffer[0].keys()
            buffer_dataset = Dataset.from_dict(
                {k: [ex[k] for ex in self._buffer] for k in keys}
            )
            for col in buffer_dataset.column_names:
                buffer_dataset = buffer_dataset.cast_column(
                    col, task["train"].features[col]
                )
            train_data = concatenate_datasets([task["train"], buffer_dataset])
        else:
            train_data = task["train"]
        train(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_data,
            valid_dataset=task["valid"],
            output_dir=output_dir,
        )
