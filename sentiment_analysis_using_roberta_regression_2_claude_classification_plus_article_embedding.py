# binary classification: neutral / bias


import pandas as pd
import transformers
from datasets import Dataset
import torch
from datasets import Dataset
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, \
    RobertaPreTrainedModel, RobertaModel
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
EPOCHS = 5

# 2 labels
id2label = {k:k for k in range(4)}
label2id = {k:k for k in range(4)}

# tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
# model = AutoModelForSequenceClassification.from_pretrained(
#     BASE_MODEL,
#     num_labels=1,
#     problem_type="regression"  # Explicitly set as regression
# )
# # Modify the model's final layer to remove sigmoid activation
# # This allows the model to predict unbounded values
# model.classifier.activation = None

# 首先在模型中添加调试信息
class RobertaForBiasClassification(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)

        # 为article_bias和source_bias创建embedding层
        self.article_bias_embeddings = nn.Embedding(4, 16)
        self.source_bias_embeddings = nn.Embedding(4, 16)

        # 合并所有特征的线性层
        combined_dim = config.hidden_size + 16 + 16
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, config.num_labels)
        )

    def forward(self,
                input_ids=None,
                attention_mask=None,
                article_bias=None,
                source_bias=None,
                labels=None):

        # 添加输入验证
        if article_bias is None or source_bias is None:
            raise ValueError("article_bias and source_bias must be provided")

        # 确保数据类型正确
        article_bias = article_bias.to(torch.long)
        source_bias = source_bias.to(torch.long)

        # 确保值在有效范围内
        if torch.max(article_bias) >= 4 or torch.min(article_bias) < 0:
            raise ValueError(f"article_bias values must be between 0 and 3, got: {article_bias}")
        if torch.max(source_bias) >= 4 or torch.min(source_bias) < 0:
            raise ValueError(f"source_bias values must be between 0 and 3, got: {source_bias}")

        # 获取RoBERTa的输出
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        sequence_output = outputs[0]
        pooled_output = sequence_output[:, 0, :]

        # 获取bias embeddings
        article_bias_embed = self.article_bias_embeddings(article_bias)
        source_bias_embed = self.source_bias_embeddings(source_bias)

        # 连接所有特征
        combined_features = torch.cat([
            pooled_output,
            article_bias_embed,
            source_bias_embed
        ], dim=1)

        # 通过分类器
        logits = self.classifier(combined_features)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        return transformers.modeling_outputs.SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = RobertaForBiasClassification.from_pretrained(
    BASE_MODEL,
    id2label=id2label,
    label2id=label2id
)



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

def load_data():
    train = pd.read_csv('./news_bias_dataset/preprocessed_dataset.csv', delimiter=',')
    print(train['bias_score'].unique())
    print(train.describe())
    new_df = train[['source_bias', 'sentence_text', 'article_bias', 'bias_score']]
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

    # 确保article_bias和source_bias是整数
    article_bias = int(examples["article_bias"])
    source_bias = int(examples["source_bias"])

    # 确保值在0-4范围内
    article_bias = max(0, min(4, article_bias))
    source_bias = max(0, min(4, source_bias))

    results = tokenizer(
        examples["sentence_text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

    results["label"] = label
    results["article_bias"] = article_bias
    results["source_bias"] = source_bias

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
    print(f'ds["train"][1]: {ds["train"][1]}')
    print(f'ds["train"][2]: {ds["train"][2]}')
    print(f'ds["train"][99]: {ds["train"][99]}')

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
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,  # Ensure the tokenizer is passed
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),  # Optional: Handle padding dynamically
    )


    print("Training model")
    trainer.train()

    output_model_file = './models/sentiment_analysis_using_roberta_classification.bin'
    output_vocab_file = './models/'

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

    # Print the results
    for idx, (true, pred) in enumerate(zip(true_labels, predicted_labels)):
        print(f"Example {idx + 1}:")
        print(f"  True Label: {true}")
        print(f"  Predicted Label: {pred}")

