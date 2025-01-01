# four classification
# {'eval_loss': 1.1621623039245605, 'eval_accuracy': 0.46859903381642515, 'eval_runtime': 11.1691, 'eval_samples_per_second': 55.6, 'eval_steps_per_second': 3.492, 'epoch': 5.0}
import os

import pandas as pd
from datasets import Dataset
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from torch.utils.data import DataLoader
import numpy as np
from evaluate import load
from transformers import TrainingArguments
from transformers import Trainer
from evaluate import load
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

BASE_MODEL = "roberta-base"
LEARNING_RATE = 2e-5
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 5

# 4 labels, 0 1 2 3
id2label = {k:k for k in range(4)}
label2id = {k:k for k in range(4)}

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, id2label=id2label, label2id=label2id)

# tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
# model = AutoModelForSequenceClassification.from_pretrained(
#     BASE_MODEL,
#     num_labels=1,
#     problem_type="regression"  # Explicitly set as regression
# )
# # Modify the model's final layer to remove sigmoid activation
# # This allows the model to predict unbounded values
# model.classifier.activation = None



device = 'cpu'
model.to(device)

def load_data():
    train = pd.read_csv('./news_bias_dataset/preprocessed_dataset.csv', delimiter=',')
    print(train['bias_score'].unique())
    print(train.describe())
    new_df = train[['sentence_text', 'bias_score']]
    train_size = 0.7
    valid_size = 0.5
    train_data = new_df.sample(frac=train_size, random_state=200)
    train_data = train_data.reset_index(drop=True)

    test_valid_data = new_df.drop(train_data.index).reset_index(drop=True)
    valid_data = test_valid_data.sample(frac=valid_size, random_state=200)
    valid_data = valid_data.reset_index(drop=True)

    test_data = test_valid_data.drop(valid_data.index).reset_index(drop=True)

    print("FULL Dataset: {}".format(new_df.shape))
    print("TRAIN Dataset: {}".format(train_data.shape))
    print("TEST Dataset: {}".format(test_data.shape))
    print("VALID Dataset: {}".format(valid_data.shape))

    raw_train_ds = Dataset.from_pandas(train_data)
    raw_valid_ds = Dataset.from_pandas(valid_data)
    raw_test_ds = Dataset.from_pandas(test_data)

    return raw_train_ds, raw_valid_ds, raw_test_ds


def preprocess_function(examples):
    if "bias_score" in examples and examples["bias_score"] is not None:
        label = min(int(examples["bias_score"]), 3)
        label = max(label, 0)
    else:
        label = 0
    results = tokenizer(examples["sentence_text"], truncation=True, padding="max_length", max_length=256)
    results["label"] = int(label)
    return results

# def preprocess_function(examples):
#     # Ensure labels are properly scaled between 0 and 1
#     label = float(examples["bias_score"]) / 3.0  # This scales them to 0-1
#
#     result = tokenizer(
#         examples["sentence_text"],
#         truncation=True,
#         padding="max_length",
#         max_length=MAX_LENGTH
#     )
#     result["label"] = label
#     return result


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def compute_metrics_for_regression(eval_pred):
    logits, labels = eval_pred
    logits = logits.squeeze()
    labels = labels.squeeze()

    print(f'logis: {logits}')
    print(f'labels: {labels}')

    # If you normalized the labels earlier, you might want to denormalize predictions here
    # logits = logits * 4 + 1  # Example: converting back to 1-5 scale

    mse = mean_squared_error(labels, logits)
    mae = mean_absolute_error(labels, logits)
    r2 = r2_score(labels, logits)

    # Calculate accuracy with a tolerance
    # Use 0.125 tolerance (slightly less than half the distance between classes)
    tolerance = 0.125
    accuracy = np.mean(np.abs(logits - labels) < tolerance)

    pred_classes = np.round(logits * 3).clip(0, 3)
    true_classes = np.round(labels * 3).clip(0, 3)
    ordinal_accuracy = np.mean(pred_classes == true_classes)

    return {
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "regression_accuracy": accuracy,
        "ordinal_accuracy": ordinal_accuracy
    }


def train_on_fake_labels(trainer, ds, raw_train_ds, raw_test_ds, map_function=preprocess_function):
    new_data = pd.read_csv('./news_bias_dataset/augmented_dataset.csv', delimiter=',')
    new_ds = Dataset.from_pandas(new_data)
    new_ds = new_ds.map(map_function, remove_columns=["sentence_text"])
    predictions = trainer.predict(new_ds)

    # Extract true labels and predicted labels
    logits = predictions.predictions
    logits = logits - np.min(logits, axis=1, keepdims=True)  # Subtract the minimum value
    logits = logits / np.sum(logits, axis=1, keepdims=True)  # Normalize by the sum
    print(f"logits: {logits}")
    predicted_labels = np.argmax(logits, axis=1)
    confidences = np.max(logits, axis=1)

    high_confidence_indices = np.where(confidences > 0.45)[0]
    high_confidence_data = new_data.iloc[high_confidence_indices]
    high_confidence_data['bias_score'] = predicted_labels[high_confidence_indices]

    # Split 30% of high_confidence_data into test set
    test_size = 0.3
    high_confidence_test_data = high_confidence_data.sample(frac=test_size, random_state=200)
    high_confidence_train_data = high_confidence_data.drop(high_confidence_test_data.index)

    # Combine with original training data
    combined_data = pd.concat([raw_train_ds.to_pandas(), high_confidence_train_data])
    combined_train_ds = Dataset.from_pandas(combined_data)
    combined_train_ds = combined_train_ds.map(map_function, remove_columns=["sentence_text", "bias_score"])

    # Retrain the model with the new combined dataset
    trainer.train_dataset = combined_train_ds
    print("Retraining model with high confidence labels")
    trainer.train()

    ds["test"] = Dataset.from_pandas(pd.concat([raw_test_ds.to_pandas(), high_confidence_test_data]))
    ds_test = ds["test"].map(map_function, remove_columns=["sentence_text", "bias_score"])
    trainer.eval_dataset = ds_test

    print("Final evaluation on test set")
    print(trainer.evaluate())
    return trainer, ds


if __name__ == '__main__':
    raw_train_ds, raw_valid_ds, raw_test_ds = load_data()
    print(raw_train_ds)
    print(raw_valid_ds)
    print(raw_test_ds)
    print(raw_train_ds[0])

    ds = {"train": raw_train_ds, "valid": raw_valid_ds, "test": raw_test_ds}
    for split in ds:
        ds[split] = ds[split].map(preprocess_function, remove_columns=["sentence_text", "bias_score"])
        # ds[split] = ds[split].map(preprocess_function)

    metric = load("accuracy")


    training_args = TrainingArguments(
        output_dir="./models/roberta-fine-tuned-regression",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="accuracy",
        load_best_model_at_end=True,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["valid"],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,  # Ensure the tokenizer is passed
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),  # Optional: Handle padding dynamically
    )

    output_model_file = './models/sentiment_analysis_using_roberta_classification.bin'
    output_vocab_file = './models/'

    if os.path.exists(output_model_file):
        print("Loading existing model")
        model = torch.load(output_model_file)
        # tokenizer = AutoTokenizer.from_pretrained(output_vocab_file)
    else:
        print("Training model")
        trainer.train()

        model_to_save = model
        torch.save(model_to_save, output_model_file)
        tokenizer.save_vocabulary(output_vocab_file)

    print("Evaluating on test set")
    trainer.eval_dataset = ds["test"]
    print(trainer.evaluate())


    print("Evaluating on test set")
    # Perform evaluation
    predictions = trainer.predict(trainer.eval_dataset)

    for i in range(3):
        trainer, ds = train_on_fake_labels(trainer, ds, raw_train_ds, raw_test_ds)

