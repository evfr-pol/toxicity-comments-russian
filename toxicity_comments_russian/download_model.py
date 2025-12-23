from pathlib import Path

import hydra
from omegaconf import DictConfig
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def download_model(cfg: DictConfig):
    tokenizer = AutoTokenizer.from_pretrained("ilyiniv1755/rubert_tiny2_toxic")

    model = AutoModelForSequenceClassification.from_pretrained("ilyiniv1755/rubert_tiny2_toxic")

    model_save_path = Path(cfg.training.output_dir) / "best_model"
    model_save_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)


if __name__ == "__main__":
    download_model()
