from pathlib import Path

import hydra
import pandas as pd
from datasets import Dataset, load_dataset
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def download_data(cfg: DictConfig) -> None:
    """
    Download and prepare Toxic Russian Comments dataset from Hugging Face.
    """
    remote_data_dir = Path(__file__).parent.parent / "data"
    train_path = remote_data_dir / "train_toxic_dataset_clean"
    val_path = remote_data_dir / "val_toxic_dataset_clean"
    test_path = remote_data_dir / "test_toxic_dataset_clean"

    if train_path.exists() and val_path.exists() and test_path.exists():
        return

    remote_data_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(cfg.data.dataset_name)

    texts = dataset["train"]["text"]
    labels = dataset["train"]["label"]

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts,
        labels,
        test_size=cfg.data.val_size,
        stratify=labels,
        random_state=cfg.data.random_state,
    )

    train_dataset = Dataset.from_pandas(pd.DataFrame({"text": train_texts, "label": train_labels}))
    val_dataset = Dataset.from_pandas(pd.DataFrame({"text": val_texts, "label": val_labels}))

    test_dataset = Dataset.from_dict(
        {
            "text": dataset["test"]["text"],
            "labels": dataset["test"]["label"],
        }
    )

    train_dataset.save_to_disk(train_path)
    val_dataset.save_to_disk(val_path)
    test_dataset.save_to_disk(test_path)


if __name__ == "__main__":
    download_data()
