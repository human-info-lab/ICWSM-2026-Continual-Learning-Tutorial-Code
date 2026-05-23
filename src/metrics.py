from __future__ import annotations

import numpy as np
from transformers import EvalPrediction

import evaluate

from src.data import Task

__all__ = [
    "MulticlassClassificationMetrics",
]


class MulticlassClassificationMetrics:
    def __init__(self):
        self.metrics = {
            "accuracy": evaluate.load("accuracy"),
            "f1": evaluate.load("f1"),
            "precision": evaluate.load("precision"),
            "recall": evaluate.load("recall"),
        }

    def __call__(self, *args, **kwargs) -> dict[str, float]:
        if len(args) == 1 and isinstance(args[0], EvalPrediction):
            predictions = args[0].predictions
            labels = args[0].label_ids
        else:
            predictions = kwargs.get("predictions")
            if predictions is None:
                predictions = args[0] if len(args) > 0 else None
            labels = kwargs.get("references")
            if labels is None:
                labels = args[1] if len(args) > 1 else None
        if labels is None or predictions is None:
            raise ValueError(
                "Both predictions and references (labels) must be provided."
            )
        target_names = kwargs.get("target_names")
        predictions = np.argmax(predictions, axis=1)
        results = {}
        for metric_name, metric in self.metrics.items():
            if metric_name.strip().lower().startswith("acc"):
                results.update(
                    metric.compute(predictions=predictions, references=labels)
                )
            else:
                for average in ["macro", "micro"]:
                    scores = metric.compute(
                        predictions=predictions,
                        references=labels,
                        average=average,
                    )
                    scores = {f"{k}_{average}": v for k, v in scores.items()}
                    results.update(scores)
        return results


class ContinualLearningMetrics:
    """

    perf_matrix:
    [
        [perf(task1 after training on task1), None, None, ...],
        [perf(task1 after training on task2), perf(task2 after training on task2), None, ...],
        ...
        [perf(task1 after training on taskT), perf(task2 after training on taskT), ..., perf(taskT after training on taskT)],
    ]
    """

    def __call__(
        self,
        tasks: list[str | Task],
        perf_matrix: list[list[dict | None]],  # [train][test]
    ) -> dict:
        task_names = [task.name if isinstance(task, Task) else task for task in tasks]

        T = len(task_names)

        final_results = {task_names[j]: perf_matrix[T - 1][j] for j in range(T)}
        snapshot_results = {task_names[i]: perf_matrix[i][i] for i in range(T)}
        best_per_task = {
            task_names[j]: float(
                max(perf_matrix[t][j]["accuracy"] for t in range(j, T))
            )
            for j in range(T)
        }
        forgetting_per_task = {
            task_names[j]: float(
                best_per_task[task_names[j]] - perf_matrix[T - 1][j]["accuracy"]
            )
            for j in range(T - 1)
        }
        mean_forgetting = float(np.mean(list(forgetting_per_task.values())))
        final_avg_acc = float(np.mean([v["accuracy"] for v in final_results.values()]))
        final_avg_f1 = float(np.mean([v["f1_macro"] for v in final_results.values()]))

        col = 12
        print(f"\n{'Task':<20} {'Best':>{col}} {'Final':>{col}} {'Forgetting':>{col}}")
        print("─" * (20 + col * 3 + 3))
        for j, task_name in enumerate(task_names):
            best = best_per_task[task_name]
            final = final_results[task_name]["accuracy"]
            forgetting = (
                f"{'(last task)':>{col}}"
                if j == T - 1
                else f"{forgetting_per_task[task_name]:>{col}.4f}"
            )
            print(f"{task_name:<20} {best:>{col}.4f} {final:>{col}.4f} {forgetting}")
        print("─" * (20 + col * 3 + 3))
        print(f"{'Final avg accuracy':<20} {'':>{col}} {final_avg_acc:>{col}.4f}")
        print(
            f"\n  Mean Forgetting : {mean_forgetting:.4f}  "
            f"({'forgetting detected' if mean_forgetting > 0.01 else 'no significant forgetting'})"
        )

        return {
            "final": final_results,
            "snapshot": snapshot_results,
            "best": best_per_task,
            #
            "mean_forgetting": mean_forgetting,
            "final_avg_accuracy": final_avg_acc,
            "final_avg_f1_macro": final_avg_f1,
            "per_task_forgetting": forgetting_per_task,
        }
