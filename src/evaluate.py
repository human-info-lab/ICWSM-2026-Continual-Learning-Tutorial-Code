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

# ── Core evaluate ──────────────────────────────────────────────────────────────


def evaluate(
    model: AdaptersModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset,
    task_name: str,
    label2id: dict[str, int],
) -> dict[str, float | str]:
    """
    Run evaluation for a single task.

    Returns a dict with accuracy, macro-F1, and a full
    sklearn classification_report string.
    """
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    # Fix #3: tmp dir is cleaned up automatically on context exit
    with tempfile.TemporaryDirectory() as tmp:
        eval_args = TrainingArguments(
            output_dir=tmp,  # use a temporary directory for any intermediate files
            do_train=False,
            do_eval=True,
            report_to="none",
            per_device_eval_batch_size=32,
        )
        # we'll pass the dataset directly to predict(...), so no need to set eval_dataset here
        trainer = AdapterTrainer(
            model=model,
            args=eval_args,
            eval_dataset=None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=None,
        )
        pred_output = trainer.predict(dataset)

    predictions = np.argmax(pred_output.predictions, axis=-1)
    labels = pred_output.label_ids

    metrics: dict[str, float | str] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_macro": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
    }

    # ── detailed per-class report ──────────────────────────────────────────────
    id2label = {v: k for k, v in label2id.items()}
    target_names = [id2label[i] for i in sorted(id2label)]
    metrics["classification_report"] = classification_report(
        labels,
        predictions,
        target_names=target_names,
        zero_division=0,
    )

    print(f"\n── {task_name} ──────────────────────────")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  F1 macro : {metrics['f1_macro']:.4f}")
    print(metrics["classification_report"])

    return metrics


# ── Evaluate across all saved adapters ────────────────────────────────────────


def main(
    model_name: str = "roberta-base",
    base_name: str = "stance",
    output_dir: str | Path = "output",
) -> dict[str, dict]:
    """
    Load each saved adapter in turn and evaluate on its test split.
    Saves a JSON summary to output_dir/eval_results.json.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name)
    tasks = load_data_dil(tokenizer)

    all_results: dict[str, dict] = {}

    for task in tasks:
        adapter_path = output_dir / f"adapter-{base_name}-{task.name}"
        if not adapter_path.exists():
            print(f"[WARN] adapter not found for task '{task.name}', skipping.")
            continue

        # Fresh model per task so adapters don't accumulate
        model: AdaptersModel = AutoAdapterModel.from_pretrained(model_name)
        lora_adapter_name = f"{base_name}-lora"
        model.load_adapter(str(adapter_path), load_as=lora_adapter_name, with_head=True)
        model.set_active_adapters(lora_adapter_name)
        model.eval()

        test_split = task.get("test", None) or task.get("valid", None)

        metrics = evaluate(
            model=model,
            tokenizer=tokenizer,
            dataset=test_split,
            task_name=task.name,
            label2id=task.label2id,
        )
        all_results[task.name] = {
            k: v for k, v in metrics.items() if k != "classification_report"
        }

    # ── aggregate summary ──────────────────────────────────────────────────────
    if all_results:
        # Fix #1: explicitly exclude sentinel keys before computing the mean
        task_results = {k: v for k, v in all_results.items() if not k.startswith("__")}
        avg_acc = np.mean([v["accuracy"] for v in task_results.values()])
        avg_f1 = np.mean([v["f1_macro"] for v in task_results.values()])
        all_results["__average__"] = {
            "accuracy": float(avg_acc),
            "f1_macro": float(avg_f1),
        }
        print(f"\n── Average over {len(task_results)} tasks ──────────────────")
        print(f"  Accuracy : {avg_acc:.4f}")
        print(f"  F1 macro : {avg_f1:.4f}")

    out_path = output_dir / "eval_results.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved → {out_path}")
    return all_results


if __name__ == "__main__":
    main()
