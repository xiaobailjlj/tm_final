# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import seaborn as sns
import transformers
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer
import logging
import transformers
from torch import cuda
# import tensorflow_ranking as tfr

logging.basicConfig(level=logging.ERROR)

# Setting up the device for GPU usage
device = 'cpu'

train = pd.read_csv('./news_bias_dataset/preprocessed_dataset.csv', delimiter=',')

train['bias_score'].unique()
# print('train bias_score')
# print(train['bias_score'].unique())

train.describe()

new_df = train[['sentence_text', 'bias_score']]

# Defining some key variables that will be used later on in the training
MAX_LEN = 256
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
# EPOCHS = 1
LEARNING_RATE = 1e-05
tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)

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


class RobertaClass(torch.nn.Module):
    def __init__(self):
        super(RobertaClass, self).__init__()
        self.l1 = RobertaModel.from_pretrained("roberta-base")
        self.pre_classifier = torch.nn.Linear(768, 768)     # 768 is the dimension of roberta-base
        self.dropout = torch.nn.Dropout(0.3)        # 0.3 is the dropout rate
        self.classifier = torch.nn.Linear(768, 4)     # 4 classes

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

model = RobertaClass()
model.to(device)

# class RobertaClass(torch.nn.Module):
#     def __init__(self, num_classes):
#         super(RobertaClass, self).__init__()
#         self.l1 = RobertaModel.from_pretrained("roberta-base")
#         self.pre_classifier = torch.nn.Linear(768, 768)
#         self.dropout = torch.nn.Dropout(0.3)
#         self.classifier = torch.nn.Linear(768, num_classes - 1)  # num_classes - 1 thresholds
#
#     def forward(self, input_ids, attention_mask, token_type_ids):
#         output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
#         hidden_state = output_1[0]
#         pooler = hidden_state[:, 0]
#         pooler = self.pre_classifier(pooler)
#         pooler = torch.nn.ReLU()(pooler)
#         pooler = self.dropout(pooler)
#         output = self.classifier(pooler)
#         return output

# num_classes = 4  # Example with 5 classes
# model = RobertaClass(num_classes=num_classes)
# model.to(device)






# Creating the loss function and optimizer, use Ordinal Cross Entropy Loss

class OrdinalCrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, y_pred, y_true):
        y_pred = self.sigmoid(y_pred)  # Convert logits to probabilities
        bins = torch.nn.functional.one_hot(y_true, num_classes=y_pred.shape[1])
        bins = torch.cumsum(bins, dim=1)
        loss = -torch.mean(bins * torch.log(y_pred + 1e-7) +
                           (1 - bins) * torch.log(1 - y_pred + 1e-7))
        return loss

# class OrdinalCrossEntropyLoss(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.sigmoid = torch.nn.Sigmoid()
#
#     def forward(self, y_pred, y_true):
#         # y_true = y_true - 1
#         y_pred = self.sigmoid(y_pred)
#         bins = torch.nn.functional.one_hot(y_true, num_classes=y_pred.shape[1]).float()
#         loss = torch.nn.functional.binary_cross_entropy(y_pred, bins, reduction='mean')
#         return loss
#
# loss_function = OrdinalCrossEntropyLoss()
# optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)


# loss_function = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

# loss_function = torch.nn.MSELoss()
# optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

# class OrdinalRegressionLoss(torch.nn.Module):
#     def __init__(self):
#         super(OrdinalRegressionLoss, self).__init__()
#         self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
#
#     def forward(self, logits, targets):
#         """
#         Args:
#             logits: Tensor of shape [batch_size, num_classes]
#             targets: Tensor of shape [batch_size] containing true class labels
#         Returns:
#             Loss value penalizing larger misclassifications more heavily.
#         """
#         # Compute standard cross-entropy loss for each example
#         ce_loss = self.cross_entropy(logits, targets)
#
#         # Compute the ordinal penalty
#         predicted_probs = torch.nn.functional.softmax(logits, dim=1)
#         predicted_labels = torch.argmax(predicted_probs, dim=1)
#
#         # Penalize by the absolute distance between predicted and true labels
#         penalties = torch.abs(predicted_labels - targets).float()
#
#         # Weight the loss by the ordinal penalties
#         ordinal_loss = ce_loss * (1 + penalties)
#
#         # Return the mean loss
#         return ordinal_loss.mean()
#
# loss_function = OrdinalRegressionLoss()
# optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

# class SigmoidOrdinalRegressionLoss(torch.nn.Module):
#     def __init__(self):
#         super(SigmoidOrdinalRegressionLoss, self).__init__()
#
#     def forward(self, logits, targets):
#         """
#         Args:
#             logits: Tensor of shape [batch_size, num_classes - 1]
#             targets: Tensor of shape [batch_size] containing true class labels
#         Returns:
#             Loss value penalizing larger misclassifications more heavily.
#         """
#         # Apply sigmoid to logits to get cumulative probabilities
#         cum_probs = torch.sigmoid(logits)
#
#         # Convert targets into binary cumulative format
#         batch_size, num_classes_minus_1 = logits.shape
#         cumulative_labels = torch.zeros((batch_size, num_classes_minus_1), device=logits.device)
#         for i in range(num_classes_minus_1):
#             cumulative_labels[:, i] = (targets > i).float()
#
#         # Compute binary cross-entropy loss for cumulative probabilities
#         bce_loss = torch.nn.functional.binary_cross_entropy(cum_probs, cumulative_labels, reduction="none")
#
#         # Penalize errors by weighting based on ordinal distance
#         ordinal_distances = torch.abs(cumulative_labels - cum_probs)
#         weighted_loss = bce_loss * (1 + ordinal_distances)
#
#         # Return the mean loss
#         return weighted_loss.mean()

# class SigmoidOrdinalRegressionLoss(torch.nn.Module):
#     def __init__(self):
#         super(SigmoidOrdinalRegressionLoss, self).__init__()
#
#     def forward(self, logits, targets):
#         """
#         Args:
#             logits: Tensor of shape [batch_size, num_classes - 1]
#             targets: Tensor of shape [batch_size] containing true class labels
#         Returns:
#             Loss value penalizing squared differences for ordinal regression.
#         """
#         # Apply sigmoid to logits to get cumulative probabilities
#         cum_probs = torch.sigmoid(logits)
#
#         # Convert targets into binary cumulative format
#         batch_size, num_classes_minus_1 = logits.shape
#         cumulative_labels = torch.zeros((batch_size, num_classes_minus_1), device=logits.device)
#         for i in range(num_classes_minus_1):
#             cumulative_labels[:, i] = (targets > i).float()
#
#         print(f"cum_probs: {cum_probs}")
#         print(f"cumulative_labels: {cumulative_labels}")
#         # Calculate Mean Squared Error (MSE) between cumulative probabilities and cumulative labels
#         mse_loss = torch.mean((cum_probs - cumulative_labels) ** 2)
#
#         return mse_loss
#
# loss_function = SigmoidOrdinalRegressionLoss()
# optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)


# loss_function = tfr.keras.losses.OrdinalLoss(ordinal_size=2)
# optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

def calcuate_accuracy(preds, targets):
    n_correct = (preds==targets).sum().item()
    return n_correct

def calculate_metrics(preds, targets):
    n_correct = (preds == targets).sum().item()
    n_total = targets.size(0)
    n_positive = (preds == 1).sum().item()
    n_true_positive = ((preds == 1) & (targets == 1)).sum().item()
    n_actual_positive = (targets == 1).sum().item()

    accuracy = n_correct / n_total
    precision = n_true_positive / n_positive if n_positive > 0 else 0
    recall = n_true_positive / n_actual_positive if n_actual_positive > 0 else 0

    return accuracy, precision, recall

# Defining the training function on the 80% of the dataset for tuning the distilbert model

def train(epoch):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    for _,data in tqdm(enumerate(training_loader, 0)):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.long)
        # targets = torch.tensor(self.targets[index], dtype=torch.long)

        outputs = model(ids, mask, token_type_ids)
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        print(f"Outputs: {outputs}")
        print(f"big_idx: {big_idx}")
        print(f"Targets: {targets}")
        # accuracy, precision, recall = calculate_metrics(big_idx, targets)
        n_correct += calcuate_accuracy(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples+=targets.size(0)

        if _%5==0:
            loss_step = tr_loss/nb_tr_steps
            accu_step = (n_correct*100)/nb_tr_examples
            print(f"Training Loss per 5 steps: {loss_step}")
            print(f"Training Accuracy per 5 steps: {accu_step}")

        optimizer.zero_grad()
        loss.backward()
        # # When using GPU
        optimizer.step()

    print(f'The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}')
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_accu}")

    return

EPOCHS = 1
for epoch in range(EPOCHS):
    train(epoch)


def valid(model, testing_loader):
    model.eval()
    n_correct = 0; n_wrong = 0; total = 0; tr_loss=0; nb_tr_steps=0; nb_tr_examples=0
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0)):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype = torch.long)
            outputs = model(ids, mask, token_type_ids).squeeze()
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calcuate_accuracy(big_idx, targets)

            nb_tr_steps += 1
            nb_tr_examples+=targets.size(0)

            if _%5==0:
                loss_step = tr_loss/nb_tr_steps
                accu_step = (n_correct*100)/nb_tr_examples
                print(f"Validation Loss per 5 steps: {loss_step}")
                print(f"Validation Accuracy per 5 steps: {accu_step}")
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Validation Loss Epoch: {epoch_loss}")
    print(f"Validation Accuracy Epoch: {epoch_accu}")

    return epoch_accu

acc = valid(model, testing_loader)
print("Accuracy on test data = %0.2f%%" % acc)

output_model_file = 'pytorch_roberta_sentiment.bin'
output_vocab_file = './'

model_to_save = model
torch.save(model_to_save, output_model_file)
tokenizer.save_vocabulary(output_vocab_file)

print('All files saved')
print('This tutorial is completed')

