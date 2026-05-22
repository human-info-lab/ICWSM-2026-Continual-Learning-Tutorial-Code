from __future__ import annotations

from collections import UserList
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from datasets import ClassLabel, Dataset, DatasetDict, Value

DATA_PATH = Path(__file__).parents[1] / "data" / "processed" / "stance_dataset.csv"


class Task(DatasetDict[str, Dataset]):
    def __init__(
        self,
        datasets: DatasetDict,
        name: str,
        label2id: dict[str, int],
    ):
        super().__init__(datasets)
        self.name = name
        self.label2id = label2id

    def get(self, key: str, default=None) -> Dataset | None:
        return super().get(key, default)


class TaskList(UserList[Task]):
    def __init__(self, tasks: list[Task] | None = None):
        super().__init__(tasks)


def load_data_dil(tokenizer: PreTrainedTokenizerBase) -> TaskList:
    # Tweet,Target,Stance,Opinion Towards,Sentiment,Split
    df = pd.read_csv(DATA_PATH)
    targets = {
        "Hillary Clinton": "HC",
        "Feminist Movement": "FM",
        "Climate Change is a Real Concern": "CC",
        "Atheism": "AT",
        "Legalization of Abortion": "AB",
        "Donald Trump": "DT",
    }
    print("Targets:", df["Target"].unique())
    df["Target"] = df["Target"].map(targets)
    df["TaskStance"] = "_" + df["Target"] + ":" + df["Stance"]
    classes = ["AGAINST", "FAVOR", "NONE"]
    label2id = {label: idx for idx, label in enumerate(classes)}
    df["Label"] = df["Stance"].map(label2id)
    dataset = Dataset.from_pandas(df)
    # get the unique values of TaskStance and sort them
    task_stance_labels = dataset.unique("TaskStance")
    task_stance_labels = sorted(task_stance_labels)
    dataset = dataset.cast_column("TaskStance", Value("string"))
    dataset = dataset.cast_column("TaskStance", ClassLabel(names=task_stance_labels))
    dataset = dataset.rename_columns(
        {
            "Tweet": "text",
            "Target": "targets",
            "Label": "labels",
            "Split": "split",
            "TaskStance": "task_stance",
            "Stance": "stance",
            "Opinion Towards": "opinion_towards",
            "Sentiment": "sentiment",
        }
    )
    dataset = dataset.map(
        lambda x: tokenizer(
            x["text"],
            # padding="max_length",
            truncation=True,
            # max_length=None,
        ),
        batched=True,
    )
    train_ds = dataset.filter(lambda x: x["split"] == "train")
    # split the train_ds into train and valid
    train_ds = train_ds.train_test_split(
        test_size=0.1, seed=42, stratify_by_column="task_stance"
    )
    datasets = (
        DatasetDict(
            {
                "train": train_ds["train"],
                "valid": train_ds["test"],
                "test": dataset.filter(lambda x: x["split"] == "test"),
            }
        ).remove_columns(["task_stance"])
        # .with_format(
        #     "torch",
        #     columns=list(
        #         set(dataset.column_names).intersection(
        #             set(
        #                 [
        #                     "labels",
        #                     "input_ids",
        #                     "token_type_ids",
        #                     "attention_mask",
        #                 ]
        #             )
        #         )
        #     ),
        # )
    )
    tasks = TaskList()
    for terget in targets.values():
        task_name = f"Task_{terget}"
        task_ds = datasets.filter(lambda x: x["targets"] == terget)
        task_ds = task_ds.remove_columns(["targets"])
        task_ds = Task(task_ds, name=task_name, label2id=label2id)
        if not task_ds["train"].num_rows:
            continue
        tasks.append(task_ds)
    return tasks


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # Print the tasks information in console (use ncurses for better formatting)
    for task in load_data_dil(tokenizer):
        print("-" * 50)
        print(f"Task Name: {task.name}")
        print(f"Columns: {task['train'].column_names}")
        print(f"Number of classes: {len(task.label2id)}")
        print(f"Number of training samples: {task['train'].num_rows}")
        print(f"Number of validation samples: {task['valid'].num_rows}")
        print(f"Number of test samples: {task['test'].num_rows}")
    print("-" * 50)
