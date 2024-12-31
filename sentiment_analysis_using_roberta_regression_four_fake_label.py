# four regression
# {'eval_loss': 0.746597945690155, 'eval_model_preparation_time': 0.0013, 'eval_mse': 0.746597945690155, 'eval_mae': 0.7047160863876343, 'eval_r2': 0.11859804391860962, 'eval_regression_accuracy': 0.38164251207729466, 'eval_ordinal_accuracy': 0.38164251207729466, 'eval_runtime': 2.8967, 'eval_samples_per_second': 214.38, 'eval_steps_per_second': 13.463, 'epoch': 5.0}
import os

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from evaluate import load
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from transformers import Trainer
from transformers import TrainingArguments

from sentiment_analysis_using_roberta_classification_four_fake_label import train_on_fake_labels

BASE_MODEL = "roberta-base"
LEARNING_RATE = 2e-5
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 5

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=1,
    problem_type="regression"  # Explicitly set as regression
)
# Modify the model's final layer to remove sigmoid activation
# This allows the model to predict unbounded values
model.classifier.activation = None



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

def preprocess_function(examples):
    if "bias_score" in examples and examples["bias_score"] is not None:
        label = min(float(examples["bias_score"]), 3.0)
        label = max(label, 0.0)
    else:
        label = 0.0
    results = tokenizer(examples["sentence_text"], truncation=True, padding="max_length", max_length=256)
    results["label"] = float(label)
    return results

def load_data():
    train_raw_dataset = pd.read_csv('./news_bias_dataset/preprocessed_dataset.csv', delimiter=',')
    print(train_raw_dataset['bias_score'].unique())
    print(train_raw_dataset.describe())
    new_df = train_raw_dataset[['sentence_text', 'bias_score']]

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
    # Use 0.5 tolerance (slightly less than half the distance between classes)
    tolerance = 0.5
    accuracy = np.mean(np.abs(logits - labels) < tolerance)

    pred_classes = np.round(logits).clip(0, 3)
    true_classes = np.round(labels).clip(0, 3)
    ordinal_accuracy = np.mean(pred_classes == true_classes)

    return {
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "regression_accuracy": accuracy,
        "ordinal_accuracy": ordinal_accuracy
    }

class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs[0][:, 0]
        loss = torch.nn.functional.mse_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss


def train_on_fake_labels_on_regression(trainer, ds, raw_train_ds, raw_test_ds, map_function=preprocess_function):
    new_data = pd.read_csv('./news_bias_dataset/augmented_dataset.csv', delimiter=',')
    new_ds = Dataset.from_pandas(new_data)
    new_ds = new_ds.map(map_function, remove_columns=["sentence_text"])
    predictions = trainer.predict(new_ds)

    # Extract true labels and predicted labels
    logits = np.round(predictions.predictions).clip(0, 3)

    high_confidence_data = new_data
    high_confidence_data['bias_score'] = logits

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
    print("len(ds_test): ", len(ds_test))
    print("len(eval_dataset): ", len(trainer.eval_dataset))

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

    metric = load("accuracy")


    training_args = TrainingArguments(
        output_dir="./models/roberta-fine-tuned-regression",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="mse",
        load_best_model_at_end=True,
        weight_decay=0.01,
    )

    trainer = RegressionTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["valid"],
        compute_metrics=compute_metrics_for_regression,
        tokenizer=tokenizer,  # Ensure the tokenizer is passed
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),  # Optional: Handle padding dynamically
    )


    output_model_file = './models/sentiment_analysis_using_roberta_regression.bin'
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

    # Extract true labels and predicted labels
    true_labels = predictions.label_ids  # Ground truth labels
    predicted_labels = predictions.predictions.argmax(axis=1)  # Predicted labels (for classification tasks)

    for i in range(3):
        trainer, ds = train_on_fake_labels_on_regression(trainer, ds, raw_train_ds, raw_test_ds, map_function=preprocess_function)

