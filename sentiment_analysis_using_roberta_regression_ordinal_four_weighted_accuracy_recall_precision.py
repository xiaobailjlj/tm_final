# ordinal regression model, weighted accuracy, and custom metrics
# weighted_accuracy_distance = (accuracy_distance_0 + accuracy_distance_1) / 2
# Test set results:
# eval_loss: 0.5043
# eval_accuracy: 0.3816
# eval_weighted_accuracy: 0.6425
# eval_ordinal_penalty: 0.8890
# eval_runtime: 11.4080
# eval_samples_per_second: 54.4360
# eval_steps_per_second: 3.4190
# epoch: 5.0000

# weighted_accuracy_distance = (accuracy_distance_0 * 2 + accuracy_distance_1) / 3
# Test set results:
# eval_loss: 0.5020
# eval_accuracy: 0.4010
# eval_weighted_accuracy: 0.5674
# eval_ordinal_penalty: 0.8580
# eval_runtime: 11.5529
# eval_samples_per_second: 53.7530
# eval_steps_per_second: 3.3760
# epoch: 5.0000

# weighted_accuracy_distance = (accuracy_distance_0*3 + accuracy_distance_1) / 4
# Test set results:
# eval_loss: 0.4981
# eval_accuracy: 0.3945
# eval_weighted_accuracy: 0.5145
# eval_ordinal_penalty: 0.9857
# eval_runtime: 11.8154
# eval_samples_per_second: 52.5580
# eval_steps_per_second: 3.3010
# epoch: 5.0000



import pandas as pd
from datasets import Dataset
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Optional, Tuple, Union
import json
import os
import numpy as np
from sklearn.metrics import precision_score, recall_score
import numpy as np
import torch

BASE_MODEL = "roberta-base"
LEARNING_RATE = 2e-5
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 5
NUM_CLASSES = 4  # 0, 1, 2, 3


class OrdinalRegressionModel(nn.Module):
    def __init__(self, base_model_name, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.roberta = AutoModel.from_pretrained(base_model_name)
        self.dropout = nn.Dropout(0.1)
        self.binary_classifiers = nn.ModuleList([
            nn.Linear(self.roberta.config.hidden_size, 1)
            for _ in range(num_classes - 1)
        ])

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)

        logits = []
        for classifier in self.binary_classifiers:
            logits.append(classifier(pooled_output))
        logits = torch.cat(logits, dim=1)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            batch_size = labels.size(0)
            ordinal_labels = torch.zeros((batch_size, self.num_classes - 1), device=labels.device)
            for i in range(batch_size):
                for threshold in range(self.num_classes - 1):
                    ordinal_labels[i, threshold] = 1 if labels[i] > threshold else 0

            loss = loss_fct(logits, ordinal_labels.float())

        return (loss, logits) if loss is not None else logits


def preprocess_function(examples):
    text = examples["sentence_text"]
    label = examples["bias_score"]

    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    for key in encoding:
        encoding[key] = encoding[key].squeeze(0)

    encoding["labels"] = label
    return encoding


class OrdinalRegressionTrainer(Trainer):
    # def _save_checkpoint(self, model, trial, metrics=None):
    #     # Ensure the output directory exists
    #     if not os.path.exists(self.args.output_dir):
    #         os.makedirs(self.args.output_dir)
    #
    #     # Save the model
    #     self.save_model(self.args.output_dir)
    #
    #     # Save the trainer state
    #     state_dict = self.state.__dict__.copy()
    #     state_dict = {str(k): v for k, v in state_dict.items()}  # Convert keys to strings
    #     with open(os.path.join(self.args.output_dir, TRAINER_STATE_NAME), "w") as f:
    #         json.dump(state_dict, f, indent=2, sort_keys=True)
    #
    #     # Save the optimizer and scheduler
    #     if self.optimizer is not None:
    #         torch.save(self.optimizer.state_dict(), os.path.join(self.args.output_dir, OPTIMIZER_NAME))
    #     if self.lr_scheduler is not None:
    #         torch.save(self.lr_scheduler.state_dict(), os.path.join(self.args.output_dir, SCHEDULER_NAME))
    #
    #     # Save the metrics
    #     if metrics is not None:
    #         with open(os.path.join(self.args.output_dir, "metrics.json"), "w") as f:
    #             json.dump(metrics, f, indent=2, sort_keys=True)
    #
    #     # Save the trial state if using hyperparameter search
    #     if trial is not None:
    #         with open(os.path.join(self.args.output_dir, "trial.json"), "w") as f:
    #             json.dump(trial, f, indent=2, sort_keys=True)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(input_ids=inputs["input_ids"],
                       attention_mask=inputs["attention_mask"],
                       labels=labels)
        loss = outputs[0]
        return (loss, outputs) if return_outputs else loss


# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = (torch.sigmoid(torch.tensor(logits)) > 0.5).long().sum(dim=1).numpy()
#
#     metrics = {
#         "loss": float(mean_squared_error(labels, predictions)),
#         "accuracy": float(np.mean(predictions == labels)),
#         "ordinal_accuracy": float(np.mean(np.abs(predictions - labels) <= 1)),
#         "mse": float(mean_squared_error(labels, predictions)),
#         "mae": float(mean_absolute_error(labels, predictions)),
#         "r2": float(r2_score(labels, predictions))
#     }
#     print(f"Computed metrics: {metrics}")  # Debugging output
#
#     return metrics

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = (torch.sigmoid(torch.tensor(logits)) > 0.5).long().sum(dim=1).numpy()

    # Print the labels and predictions for debugging
    print(f'labels: {labels}')
    print(f'logits: {logits}')
    print(f'predictions: {predictions}')

    # Basic accuracy metrics
    exact_match = float(np.mean(predictions == labels))

    # Class distribution analysis
    unique_labels, label_counts = np.unique(labels, return_counts=True)

    class_weights = {label: 1.0 / count for label, count in zip(unique_labels, label_counts)}
    weighted_accuracies = []
    for true_label in unique_labels:
        mask = labels == true_label
        if mask.any():
            class_correct = predictions[mask] == labels[mask]
            weighted_accuracies.extend([class_weights[true_label]] * len(class_correct))

    weighted_accuracy = float(np.average(predictions == labels, weights=weighted_accuracies))

    # Strict ordinal accuracy: weight larger mistakes more heavily
    ordinal_penalties = abs(
        np.subtract(predictions, labels)) ** 2  # Square the difference to penalize larger mistakes more
    weighted_ordinal_error = float(np.average(ordinal_penalties, weights=weighted_accuracies))

    # Calculate precision and recall for each class
    precision = precision_score(labels, predictions, average=None, zero_division=0)
    recall = recall_score(labels, predictions, average=None, zero_division=0)

    # Calculate weighted precision and recall (if required)
    weighted_precision = np.average(precision, weights=label_counts)
    weighted_recall = np.average(recall, weights=label_counts)

    print(f'predictions: {predictions}')
    print(f'labels: {labels}')
    print(f'precision: {precision}')
    print(f'recall: {recall}')

    accuracy_distance_0 = float(np.mean(predictions == labels))
    accuracy_distance_1 = float(np.mean(np.abs(predictions - labels) <= 1))
    weighted_accuracy_distance = (accuracy_distance_0 * 2 + accuracy_distance_1) / 3

    metrics = {
        "accuracy": exact_match,
        "weighted_accuracy": weighted_accuracy_distance,  # weighted_accuracy
        "ordinal_penalty": weighted_ordinal_error,
        "precision": precision.tolist(),  # Return precision for each class
        "recall": recall.tolist(),  # Return recall for each class
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall
    }

    return metrics


def load_data():
    train = pd.read_csv('./news_bias_dataset/preprocessed_dataset.csv', delimiter=',')
    new_df = train[['sentence_text', 'bias_score']]

    train_size = 0.7
    valid_size = 0.5
    train_data = new_df.sample(frac=train_size, random_state=200)
    test_valid_data = new_df.drop(train_data.index).reset_index(drop=True)
    valid_data = test_valid_data.sample(frac=valid_size, random_state=200)
    test_data = test_valid_data.drop(valid_data.index).reset_index(drop=True)

    return (Dataset.from_pandas(df) for df in [train_data, valid_data, test_data])


if __name__ == '__main__':
    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = OrdinalRegressionModel(BASE_MODEL, NUM_CLASSES)

    # Load and preprocess data
    raw_train_ds, raw_valid_ds, raw_test_ds = load_data()
    ds = {"train": raw_train_ds, "valid": raw_valid_ds, "test": raw_test_ds}

    for split in ds:
        ds[split] = ds[split].map(
            preprocess_function,
            remove_columns=["sentence_text", "bias_score"]
        )

    # Training arguments with corrected metric name
    training_args = TrainingArguments(
        output_dir="./models/roberta-ordinal-regression",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="weighted_accuracy",  # Using weighted accuracy as the primary metric
        greater_is_better=True,
        load_best_model_at_end=True,
        weight_decay=0.01,
        logging_steps=100,
        save_total_limit=1,
    )

    trainer = OrdinalRegressionTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["valid"],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    print("Training model...")
    train_result = trainer.train()

    print("\nFinal training metrics:")
    print(train_result.metrics)

    print("\nEvaluating on test set...")
    eval_results = trainer.evaluate(ds["test"])
    print("\nTest set results:")
    for metric, value in eval_results.items():
        print(f"{metric}: {value:.4f}")

    # Save the best model
    trainer.save_model("./models/best_ordinal_regression_model")