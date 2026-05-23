from pathlib import Path

from adapters import (
    AutoAdapterModel,
    ModelWithFlexibleHeadsAdaptersMixin,
    PredictionHead,
)
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from src.data import load_data_dil
from src.methods import SequentialFineTuning

AdaptersModel = ModelWithFlexibleHeadsAdaptersMixin | PreTrainedModel


def print_active_parameters(model: AdaptersModel):
    print("Active parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)


class ClassifierBuilder:
    def __init__(
        self,
        model_name: str,
        num_labels: int | dict[str, int],
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.task_head_name = "task-head"
        self.adapter_name = "task-lora"

    def _get_heads(self, model: AdaptersModel) -> dict[str, PredictionHead]:
        return model.heads

    def build(self) -> AdaptersModel:
        model: AdaptersModel = AutoAdapterModel.from_pretrained(self.model_name)
        # Add a classification head for the current task
        model.add_classification_head(self.task_head_name, num_labels=self.num_labels)
        # Add a LoRA adapter for the current task
        model.add_adapter(self.adapter_name, "lora")
        # enable training for the adapter
        model.train_adapter(self.adapter_name)
        # enable training/inference for the adapter (forward pass)
        model.set_active_adapters(self.adapter_name)
        # Set the active head to the current task head
        model.active_head = self.task_head_name
        return model


def main(
    model_name: str = "roberta-base",
    benchmark: str = "stance",
    method_name: str = "sft",
    output_dir: str | Path = "output",
):
    if method_name is None:
        method_name = "sft"

    if method_name == "sft":
        method = SequentialFineTuning()
    else:
        raise ValueError(f"Method {method_name} not recognized.")

    output_dir: Path = Path(output_dir) / benchmark / method.name
    output_dir.mkdir(exist_ok=True, parents=True)

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name)

    tasks = load_data_dil(tokenizer)
    num_labels = len(tasks[0].label2id)

    builder = ClassifierBuilder(model_name=model_name, num_labels=num_labels)
    model = builder.build()

    method.setup(model, tasks)

    for task_idx, task in enumerate(tasks):
        task_output_dir = output_dir / f"adapter-{task.name}"

        method.train_task(
            model=model,
            tokenizer=tokenizer,
            task=task,
            task_idx=task_idx,
            output_dir=task_output_dir / "checkpoints",
        )

        method.after_task(model, task, task_idx)

        # Save the adapter and head for the current task
        model.save_adapter(task_output_dir, builder.adapter_name, with_head=False)
        model.save_head(str(task_output_dir), builder.task_head_name)


if __name__ == "__main__":
    main()
