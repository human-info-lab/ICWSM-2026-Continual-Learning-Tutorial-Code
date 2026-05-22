from pathlib import Path

from adapters import (
    AdapterTrainer,
    AutoAdapterModel,
    ModelWithFlexibleHeadsAdaptersMixin,
)
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.training_args import TrainingArguments

from src.data import load_data_dil

AdaptersModel = ModelWithFlexibleHeadsAdaptersMixin | PreTrainedModel


def train(
    model: AdaptersModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset,
    valid_dataset,
):
    training_args = TrainingArguments(
        learning_rate=1e-4,
        num_train_epochs=6,
        report_to="none",
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
    )
    trainer.train()


def print_active_parameters(model: AdaptersModel):
    print("Active parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)


def main(
    model_name: str = "roberta-base",
    base_name: str = "stance",
    output_dir: str | Path = "output",
):
    output_dir: Path = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name)

    tasks = load_data_dil(tokenizer)
    num_labels = len(tasks[0].label2id)

    model: AdaptersModel = AutoAdapterModel.from_pretrained(model_name)

    task_head = f"{base_name}-head"
    model.add_classification_head(task_head, num_labels=num_labels)

    lora_adapter_name = f"{base_name}-lora"
    model.add_adapter(lora_adapter_name, "lora")
    model.train_adapter(lora_adapter_name)

    model.set_active_adapters(lora_adapter_name)
    model.active_head = task_head

    for task in tasks:
        train_dataset = task["train"]
        eval_dataset = task["valid"]
        train(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            valid_dataset=eval_dataset,
        )
        adapter_save_path = output_dir / f"adapter-{base_name}-{task.name}"
        model.save_adapter(adapter_save_path, lora_adapter_name, with_head=False)
        model.save_head(str(adapter_save_path), task_head)


if __name__ == "__main__":
    main()
