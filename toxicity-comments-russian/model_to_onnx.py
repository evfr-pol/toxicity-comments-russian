from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def to_onnx(cfg: DictConfig):
    model_save_path = Path(cfg.training.output_dir) / "best_model"
    model = AutoModelForSequenceClassification.from_pretrained(model_save_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_save_path)
    text = "Пример текста"
    inputs = tokenizer(text, return_tensors="pt")

    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        model_save_path / f"{cfg.model.save_onnx_name}.onnx",
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence"},
            "attention_mask": {0: "batch_size", 1: "sequence"},
            "logits": {0: "batch_size"},
        },
        external_data=False,
    )


if __name__ == "__main__":
    to_onnx()
