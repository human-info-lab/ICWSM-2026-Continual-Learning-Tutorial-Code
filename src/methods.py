from __future__ import annotations

from pathlib import Path

from adapters import AdapterTrainer, ModelWithFlexibleHeadsAdaptersMixin
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
