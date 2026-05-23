from pathlib import Path

from adapters import (
    AdapterTrainer,
    AutoAdapterModel,
    ModelWithFlexibleHeadsAdaptersMixin,
)
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.training_args import TrainingArguments

from src.data import load_data_dil
from src.metrics import MulticlassClassificationMetrics

AdaptersModel = ModelWithFlexibleHeadsAdaptersMixin | PreTrainedModel

compute_metrics = MulticlassClassificationMetrics()


def train(
    model: AdaptersModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset,
    valid_dataset,
    output_dir: Path | str | None = None,
    num_train_epochs: int = 20,
):
    if output_dir:
        output_dir = Path(output_dir)
    else:
        output_dir = None
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
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.001,
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
            early_stopping_callback,
        ],
        compute_metrics=compute_metrics,
    )
    trainer.train()


def print_active_parameters(model: AdaptersModel):
    print("Active parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)


def main(
    model_name: str = "roberta-base",
    benchmark: str = "stance",
    output_dir: str | Path = "output",
):
    output_dir: Path = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name)

    tasks = load_data_dil(tokenizer)
    num_labels = len(tasks[0].label2id)

    model: AdaptersModel = AutoAdapterModel.from_pretrained(model_name)

    task_head = f"{benchmark}-head"
    model.add_classification_head(task_head, num_labels=num_labels)

    lora_adapter_name = f"{benchmark}-lora"
    model.add_adapter(lora_adapter_name, "lora")
    model.train_adapter(lora_adapter_name)

    model.set_active_adapters(lora_adapter_name)
    model.active_head = task_head

    for task in tasks:
        # same model is trained sequentially on all tasks,
        #     Domain-IL - the head is shared across all tasks
        task_output_dir = output_dir / f"adapter-{benchmark}-{task.name}"
        train_dataset = task["train"]
        eval_dataset = task["valid"]
        train(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            valid_dataset=eval_dataset,
            output_dir=task_output_dir / "checkpoints",
        )
        model.save_adapter(task_output_dir, lora_adapter_name, with_head=False)
        model.save_head(str(task_output_dir), task_head)


if __name__ == "__main__":
    main()
