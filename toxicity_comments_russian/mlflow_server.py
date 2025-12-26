from pathlib import Path

import mlflow
import mlflow.pyfunc
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from transformers import AutoTokenizer


class ToxicONNXModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        model_dir = Path(context.artifacts["onnx_model"])
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.onnx_path = model_dir / "model.onnx"
        self.session = ort.InferenceSession(str(self.onnx_path))

    def predict(self, context, model_input: pd.DataFrame) -> np.ndarray:
        texts = model_input["text"].tolist()

        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="np",
        )

        inputs = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }

        outputs = self.session.run(None, inputs)
        logits = outputs[0]
        logits_torch = torch.from_numpy(logits)
        probs_torch = torch.softmax(logits_torch, dim=1)
        return probs_torch[:, 1].numpy()


def main():
    model_path = Path(__file__).resolve().parent.parent / "rubert_tiny2_toxic/best_model"
    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(
            name="toxic_russian_comments_onnx",
            python_model=ToxicONNXModel(),
            artifacts={"onnx_model": str(model_path)},
        )

        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/toxic_russian_comments_onnx"
        mlflow.register_model(model_uri, "toxic_russian_comments_onnx")


if __name__ == "__main__":
    main()
