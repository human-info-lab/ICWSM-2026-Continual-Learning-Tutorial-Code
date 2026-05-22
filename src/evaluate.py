import json
import tempfile
from pathlib import Path

import numpy as np
from adapters import (
    AdapterTrainer,
    AutoAdapterModel,
    ModelWithFlexibleHeadsAdaptersMixin,
)
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.training_args import TrainingArguments

from datasets import Dataset
from src.data import load_data_dil

AdaptersModel = ModelWithFlexibleHeadsAdaptersMixin | PreTrainedModel


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

    predictions = np.argmax(pred_output.predictions, axis=-1)
    labels = pred_output.label_ids
    id2label = {v: k for k, v in label2id.items()}
    target_names = [id2label[i] for i in sorted(id2label)]

    metrics: dict[str, float | str] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_macro": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "classification_report": classification_report(
            labels, predictions, target_names=target_names, zero_division=0
        ),
    }

    print(f"\n── {task_name} ──────────────────────────")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  F1 macro : {metrics['f1_macro']:.4f}")
    print(metrics["classification_report"])

    return metrics


def _adapter_path(output_dir: Path, base_name: str, task_name: str) -> Path:
    return output_dir / f"adapter-{base_name}-{task_name}"


def _load_model(model_name: str, adapter_path: Path, base_name: str) -> AdaptersModel:
    model: AdaptersModel = AutoAdapterModel.from_pretrained(model_name)
    lora_name = f"{base_name}-lora"
    model.load_adapter(str(adapter_path), load_as=lora_name, with_head=False)
    model.set_active_adapters(lora_name)
    model.load_head(str(adapter_path))
    model.active_head = f"{base_name}-head"
    model.eval()
    return model


def _compute_cl_metrics(
    available_tasks,
    acc_matrix: list[list[dict | None]],
    final_results: dict[str, dict],
) -> dict:
    T = len(available_tasks)

    best_per_task = {
        available_tasks[j].name: float(
            max(acc_matrix[t][j]["accuracy"] for t in range(j, T))
        )
        for j in range(T)
    }
    forgetting_per_task = {
        available_tasks[j].name: float(
            best_per_task[available_tasks[j].name] - acc_matrix[T - 1][j]["accuracy"]
        )
        for j in range(T - 1)
    }
    mean_forgetting = float(np.mean(list(forgetting_per_task.values())))
    final_avg_acc = float(np.mean([v["accuracy"] for v in final_results.values()]))
    final_avg_f1 = float(np.mean([v["f1_macro"] for v in final_results.values()]))

    col = 12
    print(f"\n{'Task':<20} {'Best':>{col}} {'Final':>{col}} {'Forgetting':>{col}}")
    print("─" * (20 + col * 3 + 3))
    for j, task in enumerate(available_tasks):
        best = best_per_task[task.name]
        final = final_results[task.name]["accuracy"]
        forgetting = (
            f"{'(last task)':>{col}}"
            if j == T - 1
            else f"{forgetting_per_task[task.name]:>{col}.4f}"
        )
        print(f"{task.name:<20} {best:>{col}.4f} {final:>{col}.4f} {forgetting}")
    print("─" * (20 + col * 3 + 3))
    print(f"{'Final avg accuracy':<20} {'':>{col}} {final_avg_acc:>{col}.4f}")
    print(
        f"\n  Mean Forgetting : {mean_forgetting:.4f}  "
        f"({'forgetting detected' if mean_forgetting > 0.01 else 'no significant forgetting'})"
    )

    return {
        "mean_forgetting": mean_forgetting,
        "final_avg_accuracy": final_avg_acc,
        "final_avg_f1_macro": final_avg_f1,
        "per_task_forgetting": forgetting_per_task,
    }


def main(
    model_name: str = "roberta-base",
    base_name: str = "stance",
    output_dir: str | Path = "output",
) -> dict[str, dict]:
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name)
    tasks = load_data_dil(tokenizer)

    available_tasks = [
        t for t in tasks if _adapter_path(output_dir, base_name, t.name).exists()
    ]
    if not available_tasks:
        print("[WARN] No saved adapters found.")
        return {}

    T = len(available_tasks)
    acc_matrix: list[list[dict | None]] = [[None] * T for _ in range(T)]

    for t_idx, adapter_task in enumerate(available_tasks):
        print(f"\n══ Adapter {t_idx + 1}/{T}: '{adapter_task.name}' ══")
        model = _load_model(
            model_name,
            _adapter_path(output_dir, base_name, adapter_task.name),
            base_name,
        )

        for j_idx in range(t_idx + 1):
            eval_task = available_tasks[j_idx]
            test_split = eval_task.get("test") or eval_task.get("valid")
            metrics = evaluate(
                model=model,
                tokenizer=tokenizer,
                dataset=test_split,
                task_name=eval_task.name,
                label2id=eval_task.label2id,
            )
            acc_matrix[t_idx][j_idx] = {
                k: v for k, v in metrics.items() if k != "classification_report"
            }

    snapshot_results = {available_tasks[i].name: acc_matrix[i][i] for i in range(T)}
    final_results = {available_tasks[j].name: acc_matrix[T - 1][j] for j in range(T)}
    cl_metrics = (
        _compute_cl_metrics(available_tasks, acc_matrix, final_results) if T > 1 else {}
    )

    all_results = {
        "snapshot": snapshot_results,
        "final": final_results,
        "cl_metrics": cl_metrics,
    }
    out_path = output_dir / "eval_results.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved → {out_path}")
    return all_results


if __name__ == "__main__":
    main()
