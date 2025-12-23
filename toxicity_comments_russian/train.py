import subprocess
from pathlib import Path

import hydra
import mlflow
import numpy as np
from datasets import load_from_disk
from dvc.repo import Repo
from omegaconf import DictConfig
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    probs = pred.predictions[:, 1]
    return {
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "roc_auc": roc_auc_score(labels, probs),
    }


# def pull_data_dvc():
#     repo = Repo(str(Path(__file__).parent))
#     repo.pull(force=True)
#     repo.close()


def get_git_commit_id():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig):
    mlflow.set_tracking_uri(cfg.logging.mlflow_uri)
    mlflow.set_experiment(cfg.logging.experiment_name)
    commit_id = get_git_commit_id()

    # pull_data_dvc()

    with mlflow.start_run():
        mlflow.log_params(
            {
                "pretrained_model": cfg.model.pretrained_model_name,
                "num_labels": cfg.model.num_labels,
                "freeze_backbone": cfg.model.freeze_backbone,
                "train_batch_size": cfg.training.train_batch_size,
                "eval_batch_size": cfg.training.eval_batch_size,
                "learning_rate": cfg.training.learning_rate,
                "num_epochs": cfg.training.num_epochs,
                "weight_decay": cfg.training.weight_decay,
            }
        )
        mlflow.log_param("git_commit_id", commit_id)

        data_dir = Path(cfg.data.data_dir)
        train_dataset = load_from_disk(data_dir / "train_toxic_dataset_clean")
        val_dataset = load_from_disk(data_dir / "val_toxic_dataset_clean")
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.pretrained_model_name)

        def preprocess(batch):
            return tokenizer(
                batch["text"], padding="max_length", truncation=True, max_length=cfg.data.max_length
            )

        train_dataset = train_dataset.map(preprocess, batched=True)
        val_dataset = val_dataset.map(preprocess, batched=True)
        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model.pretrained_model_name, num_labels=cfg.model.num_labels
        )
        training_args = TrainingArguments(
            output_dir=Path(cfg.training.output_dir),
            per_device_train_batch_size=cfg.training.train_batch_size,
            per_device_eval_batch_size=cfg.training.eval_batch_size,
            num_train_epochs=cfg.training.num_epochs,
            learning_rate=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
            eval_strategy=cfg.training.eval_strategy,
            eval_steps=cfg.training.eval_steps,
            logging_steps=cfg.training.logging_steps,
            save_steps=cfg.training.save_steps,
            save_total_limit=cfg.training.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="roc_auc",
            greater_is_better=True,
            dataloader_num_workers=cfg.training.num_workers,
            eval_on_start=True,
            report_to=["mlflow"],
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        model_save_path = Path(cfg.training.output_dir) / "best_model"
        trainer.save_model(model_save_path)
        trainer.tokenizer.save_pretrained(model_save_path)


if __name__ == "__main__":
    train()
