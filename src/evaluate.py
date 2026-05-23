import json
import tempfile
from pathlib import Path

from adapters import (
    AdapterTrainer,
    AutoAdapterModel,
    ModelWithFlexibleHeadsAdaptersMixin,
)
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.training_args import TrainingArguments

from src.data import load_data_dil
from src.metrics import MulticlassClassificationMetrics, ContinualLearningMetrics

AdaptersModel = ModelWithFlexibleHeadsAdaptersMixin | PreTrainedModel

compute_clf_metrics = MulticlassClassificationMetrics()
compute_cl_metrics = ContinualLearningMetrics()


def evaluate(
    model: AdaptersModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset,
    task_name: str,
    label2id: dict[str, int],
) -> dict[str, float | str]:
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, padding=True, return_tensors="pt"
    )

    with tempfile.TemporaryDirectory() as tmp:
        trainer = AdapterTrainer(
            model=model,
            args=TrainingArguments(
                output_dir=tmp,
                do_train=False,
                do_eval=True,
                report_to="none",
                per_device_eval_batch_size=32,
            ),
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        pred_output = trainer.predict(dataset)

    id2label = {v: k for k, v in label2id.items()}
    target_names = [id2label[i] for i in sorted(id2label)]

    metrics: dict[str, float | str] = compute_clf_metrics(
        predictions=pred_output.predictions,
        references=pred_output.label_ids,
        target_names=target_names,
    )

    print(f"\n── {task_name} ──────────────────────────")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  F1 macro : {metrics['f1_macro']:.4f}")

    return metrics


def _adapter_path(output_dir: Path, task_name: str) -> Path:
    return output_dir / f"adapter-{task_name}"


def _load_model(model_name: str, adapter_path: Path) -> AdaptersModel:
    model: AdaptersModel = AutoAdapterModel.from_pretrained(model_name)
    lora_name = "task-lora"
    model.load_adapter(str(adapter_path), load_as=lora_name, with_head=False)
    model.set_active_adapters(lora_name)
    model.load_head(str(adapter_path))
    model.active_head = "task-head"
    model.eval()
    return model


def main(
    model_name: str = "roberta-base",
    benchmark: str = "stance",
    method_name: str = "sft",
    output_dir: str | Path = "output",
) -> dict[str, dict]:
    output_dir: Path = Path(output_dir) / benchmark / method_name

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name)
    tasks = load_data_dil(tokenizer)

    available_tasks = [t for t in tasks if _adapter_path(output_dir, t.name).exists()]
    if not available_tasks:
        print("[WARN] No saved adapters found.")
        return {}

    T = len(available_tasks)

    perf_matrix: list[list[dict | None]] = [[None] * T for _ in range(T)]

    for t_idx, adapter_task in enumerate(available_tasks):
        print(f"\n══ Adapter {t_idx + 1}/{T}: '{adapter_task.name}' ══")
        model = _load_model(
            model_name,
            _adapter_path(output_dir, adapter_task.name),
        )

        for j_idx in range(t_idx + 1):
            eval_task = available_tasks[j_idx]
            test_split = eval_task.get("test") or eval_task.get("valid")
            perf_matrix[t_idx][j_idx] = evaluate(
                model=model,
                tokenizer=tokenizer,
                dataset=test_split,
                task_name=eval_task.name,
                label2id=eval_task.label2id,
            )

    cl_metrics = compute_cl_metrics(available_tasks, perf_matrix) if T > 1 else {}

    out_path = output_dir / "eval_results.json"
    out_path.write_text(json.dumps(cl_metrics, indent=2))
    print(f"\nResults saved → {out_path}")
    return cl_metrics


if __name__ == "__main__":
    main()
