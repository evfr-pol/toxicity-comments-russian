from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def to_onnx(cfg: DictConfig):
    model_save_path = Path(__file__).parent.parent / cfg.model.model_name / "best_model"
    model = AutoModelForSequenceClassification.from_pretrained(model_save_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_save_path)
    texts = ["Пример текста", "Ещё пример"]
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    torch.onnx.export(
        model,
        (
            inputs["input_ids"],
            inputs["attention_mask"],
        ),
        model_save_path / "model.onnx",
        input_names=[
            "input_ids",
            "attention_mask",
        ],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        external_data=False,
    )


if __name__ == "__main__":
    to_onnx()
