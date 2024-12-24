# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch import cuda
import seaborn as sns
import transformers
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer
import logging
import transformers
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 3)  # 只需要3个二元分类器
        # 添加bias初始化
        self.classifier.bias = torch.nn.Parameter(torch.tensor([0.0, 0.0, -1.0]))  # 使得类别3更难预测

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        logits = self.classifier(pooler)
        return logits


# Creating the loss function and optimizer, use Ordinal Cross Entropy Loss

# class OrdinalRegressionLoss(nn.Module):
#     def __init__(self, num_classes=4, weight_power=2.0):
#         super(OrdinalRegressionLoss, self).__init__()
#         self.sigmoid = nn.Sigmoid()
#         self.num_classes = num_classes
#         self.weight_power = weight_power
#
#     def get_weight_matrix(self, targets):
#         """
#         创建非线性权重矩阵
#         weight_power: 控制权重增长的速度，更大的值会让远距离的惩罚更重
#         """
#         device = targets.device
#         weights = torch.zeros((self.num_classes, self.num_classes), device=device)
#
#         for i in range(self.num_classes):
#             for j in range(self.num_classes):
#                 # 使用幂函数来增加距离较远的类别之间的权重
#                 weights[i, j] = torch.pow(torch.tensor(abs(i - j)), self.weight_power)
#
#         return weights
#
#     def forward(self, predictions, targets):
#         batch_size = predictions.size(0)
#
#         # 转换为序数编码
#         ordinal_targets = torch.zeros((batch_size, self.num_classes - 1), device=predictions.device)
#         for i in range(self.num_classes - 1):
#             ordinal_targets[:, i] = (targets > i).float()
#
#         # 获取概率
#         probs = self.sigmoid(predictions[:, :-1])
#         print(f"probs:{probs}")
#
#         # 基础损失：二元交叉熵
#         base_loss = nn.BCELoss(reduction='none')(probs, ordinal_targets)
#
#         # 获取权重矩阵
#         weight_matrix = self.get_weight_matrix(targets)
#
#         # 应用权重到损失
#         weighted_loss = base_loss * (1.0 + weight_matrix[targets.long()][:, :-1])
#
#         return weighted_loss.mean()


class ImprovedOrdinalRegressionLoss(nn.Module):
    def __init__(self, num_classes=4, class_counts=None):
        super(ImprovedOrdinalRegressionLoss, self).__init__()
        self.num_classes = num_classes

        # 计算类别权重
        if class_counts is not None:
            total_samples = sum(class_counts)
            weights = [total_samples / (len(class_counts) * count) for count in class_counts]
            # 归一化权重
            weights = [w / sum(weights) * len(weights) for w in weights]
            self.class_weights = torch.tensor(weights, dtype=torch.float32)
        else:
            self.class_weights = torch.ones(num_classes)

    def forward(self, logits, targets):
        device = logits.device
        self.class_weights = self.class_weights.to(device)
        batch_size = logits.size(0)

        # 创建序数标签
        ordinal_targets = torch.zeros((batch_size, self.num_classes - 1), device=device)
        for i in range(self.num_classes - 1):
            ordinal_targets[:, i] = (targets > i).float()

        # 计算每个决策点的损失
        losses = []
        for i in range(self.num_classes - 1):
            positive_weight = self.class_weights[i + 1:].sum() / self.class_weights[:i + 1].sum()
            bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([positive_weight]).to(device))
            loss_i = bce(logits[:, i], ordinal_targets[:, i])
            losses.append(loss_i)

        # 添加一致性约束
        consistency_loss = torch.tensor(0., device=device)
        probs = torch.sigmoid(logits)
        for i in range(self.num_classes - 2):
            consistency_loss += torch.mean(torch.relu(probs[:, i + 1] - probs[:, i]))

        total_loss = sum(losses) + 0.1 * consistency_loss
        return total_loss

# 修改学习率和优化器设置
INITIAL_LEARNING_RATE = 2e-5  # 稍微提高初始学习率
WEIGHT_DECAY = 0.01  # 添加权重衰减以防止过拟合


def get_optimizer_and_scheduler(model):
    # 使用AdamW优化器，它比普通的Adam更好地处理权重衰减
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=INITIAL_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )

    # 添加学习率调度器
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',  # 监控损失值
        factor=0.5,  # 当触发时将学习率减半
        patience=2,  # 等待2个epoch后再调整学习率
        verbose=True,  # 打印学习率变化
        min_lr=1e-6  # 最小学习率
    )

    return optimizer, scheduler

def calcuate_accuracy(preds, targets):
    n_correct = (preds==targets).sum().item()
    return n_correct


# Defining the training function on the 80% of the dataset for tuning the distilbert model

def train(epoch, model, optimizer, scheduler, loss_function):
    model.train()
    tr_loss = 0
    predictions_list = []
    targets_list = []
    n_correct = 0
    tr_loss = 0
    nb_tr_steps = 0
    nb_tr_examples = 0

    for _, data in tqdm(enumerate(training_loader, 0)):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.long)

        outputs = model(ids, mask, token_type_ids)
        print(f"outputs:{outputs}")
        print(f"targets:{targets}")
        loss = loss_function(outputs, targets)

        # 计算预测类别
        probs = torch.sigmoid(outputs)
        print(f"probs:{probs}")
        threshold = 0.5
        predicted_classes = torch.zeros_like(targets)
        for i in range(len(probs)):
            if probs[i, 0] <= threshold:
                predicted_classes[i] = 0
            elif probs[i, 1] <= threshold:
                predicted_classes[i] = 1
            elif probs[i, 2] <= threshold:
                predicted_classes[i] = 2
            else:
                predicted_classes[i] = 3
        n_correct += calcuate_accuracy(predicted_classes, targets)
        nb_tr_steps += 1
        nb_tr_examples+=targets.size(0)

        predictions_list.extend(predicted_classes.cpu().numpy())
        targets_list.extend(targets.cpu().numpy())


        if _ % 5 == 0:
            print(f"\nStep: {_}")
            print(f"Raw logits: {outputs[0]}")  # 打印原始logits
            print(f"Probabilities: {probs[0]}")  # 打印概率
            print(f"Predictions distribution: {np.bincount(predictions_list, minlength=4)}")
            print(f"Targets distribution: {np.bincount(targets_list, minlength=4)}")
            print(f"Loss: {loss.item()}")
            accu_step = (n_correct*100)/nb_tr_examples
            print(f"Training Accuracy per 5 steps: {accu_step}")


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = tr_loss / len(training_loader)
    scheduler.step(epoch_loss)

    return epoch_loss

model = RobertaClass()
model.to(device)
class_counts = [1113, 1481, 1212, 334]
loss_function = ImprovedOrdinalRegressionLoss(num_classes=4, class_counts=class_counts)
optimizer, scheduler = get_optimizer_and_scheduler(model)

EPOCHS = 1
for epoch in range(EPOCHS):
    epoch_loss, epoch_accu = train(epoch, model, optimizer, scheduler, loss_function)

output_model_file = 'pytorch_roberta_sentiment.bin'
output_vocab_file = './'

model_to_save = model
torch.save(model_to_save, output_model_file)
tokenizer.save_vocabulary(output_vocab_file)

print('All files saved')
print('This tutorial is completed')