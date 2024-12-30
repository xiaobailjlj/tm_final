import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

train_raw_data = pd.read_csv('./news_bias_dataset/preprocessed_dataset.csv', delimiter=',')

train_raw_data['bias_score'].unique()
# print('train bias_score')
# print(train['bias_score'].unique())

MAX_LEN = 256
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载预训练模型和分词器
tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')
base_model = AutoModelForSequenceClassification.from_pretrained(
    'cardiffnlp/twitter-roberta-base-sentiment-latest'
).base_model

train_raw_data.describe()

new_df = train_raw_data[['sentence_text', 'bias_score']]

class SentimentData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.sentence_text
        self.targets = self.data.bias_score
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

train_size = 0.8
train_data=new_df.sample(frac=train_size,random_state=200)
test_data=new_df.drop(train_data.index).reset_index(drop=True)
train_data = train_data.reset_index(drop=True)
print("FULL Dataset: {}".format(new_df.shape))
print("TRAIN Dataset: {}".format(train_data.shape))
print("TEST Dataset: {}".format(test_data.shape))

training_set = SentimentData(train_data, tokenizer, MAX_LEN)
testing_set = SentimentData(test_data, tokenizer, MAX_LEN)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)


class OrdinalRegressionHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.classifier = nn.Linear(input_dim, 1)
        self.thresholds = nn.Parameter(torch.zeros(num_classes - 1))

    def forward(self, x):
        logits = self.classifier(x)
        repeated_logits = logits.repeat(1, self.num_classes - 1)
        return repeated_logits - self.thresholds


class BiasClassifier(nn.Module):
    def __init__(self, pretrained_model, num_classes=4):
        super().__init__()
        self.num_classes = num_classes
        self.roberta = pretrained_model
        self.ordinal_head = OrdinalRegressionHead(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Use CLS token
        logits = self.ordinal_head(pooled_output)
        return logits


def train_model(model, train_loader, val_loader, device, num_epochs=5):
    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = nn.BCEWithLogitsLoss()

    best_val_f1 = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            input_ids = batch['ids'].to(device)
            attention_mask = batch['mask'].to(device)
            labels = batch['targets'].to(device)

            # Convert labels to binary representation
            binary_labels = torch.zeros(labels.size(0), model.num_classes - 1).to(device)
            for i, label in enumerate(labels):
                binary_labels[i, :int(label.item())] = 1

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, binary_labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Validation
        model.eval()
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['ids'].to(device)
                attention_mask = batch['mask'].to(device)
                labels = batch['targets'].to(device)

                outputs = model(input_ids, attention_mask)
                predictions = (outputs >= 0).sum(dim=1)

                val_preds.extend(predictions.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        val_acc = accuracy_score(val_labels, val_preds)

        print(f'Epoch {epoch + 1}:')
        print(f'Average Loss: {total_loss / len(train_loader):.4f}')
        print(f'Validation F1: {val_f1:.4f}')
        print(f'Validation Accuracy: {val_acc:.4f}')

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'best_model.pt')


def main():
    # 初始化模型
    model = BiasClassifier(base_model).to(device)

    # 训练模型
    train_model(model, training_loader, testing_loader, device)


if __name__ == '__main__':
    main()