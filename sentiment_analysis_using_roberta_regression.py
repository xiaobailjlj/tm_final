# https://lajavaness.medium.com/regression-with-text-input-using-bert-and-transformers-71c155034b13

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
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score



BASE_MODEL = "roberta-base"
LEARNING_RATE = 2e-5
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 10

# 5 labels
id2label = {k:k for k in range(5)}
label2id = {k:k for k in range(5)}

# tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
# model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, id2label=id2label, label2id=label2id)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=1)

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
    label = examples["bias_score"]
    # examples = tokenizer(examples["sentence_text"], truncation=True, padding="max_length", max_length=256)
    # examples["label"] = label
    result = tokenizer(examples["sentence_text"], truncation=True, padding="max_length", max_length=256)
    # result["label"] = label
    result["label"] = float(label)
    return result


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def compute_metrics_for_regression(eval_pred):
    logits, labels = eval_pred
    labels = labels.reshape(-1, 1)

    mse = mean_squared_error(labels, logits)
    mae = mean_absolute_error(labels, logits)
    r2 = r2_score(labels, logits)
    single_squared_errors = ((logits - labels).flatten() ** 2).tolist()

    # Compute accuracy
    # Based on the fact that the rounded score = true score only if |single_squared_errors| < 0.5
    accuracy = sum([1 for e in single_squared_errors if e < 0.25]) / len(single_squared_errors)

    return {"mse": mse, "mae": mae, "r2": r2, "accuracy": accuracy}


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
    print(f'ds["train"][0]: {ds["train"][0]}')

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

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=ds["train"],
    #     eval_dataset=ds["valid"],
    #     compute_metrics=compute_metrics
    # )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["valid"],
        compute_metrics=compute_metrics_for_regression,
        tokenizer=tokenizer,  # Ensure the tokenizer is passed
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),  # Optional: Handle padding dynamically
    )

    print("Training model")
    trainer.train()

    output_model_file = './models/sentiment_analysis_using_roberta_regression.bin'
    output_vocab_file = './models/'

    model_to_save = model
    torch.save(model_to_save, output_model_file)
    tokenizer.save_vocabulary(output_vocab_file)

    print("Evaluating on test set")
    trainer.eval_dataset = ds["test"]
    trainer.evaluate()



