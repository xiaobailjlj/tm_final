import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("kevinscaria/atsc_tk-instruct-base-def-pos-neg-neut-combined")


# Custom model for ordinal regression with LSTM
class OrdinalRegressionModel(nn.Module):
    def __init__(self, pretrained_model, num_classes, lstm_hidden_size=256, num_lstm_layers=2, lstm_dropout=0.2):
        super(OrdinalRegressionModel, self).__init__()
        self.seq2seq = pretrained_model

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.seq2seq.config.d_model,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if num_lstm_layers > 1 else 0
        )

        # Dropout layer
        self.dropout = nn.Dropout(0.3)

        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, lstm_hidden_size),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(lstm_hidden_size, num_classes - 1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # Create decoder_input_ids
        decoder_input_ids = torch.full(
            (input_ids.shape[0], 1),
            self.seq2seq.config.decoder_start_token_id,
            device=input_ids.device
        )

        # Get outputs from the base model
        outputs = self.seq2seq(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=True,
            return_dict=True
        )

        # Get decoder hidden states
        decoder_hidden_states = outputs.decoder_hidden_states[-1]  # Shape: [batch_size, seq_len, hidden_size]

        # Pass through LSTM
        lstm_output, (hidden, cell) = self.lstm(decoder_hidden_states)

        # Concatenate the final forward and backward hidden states
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        hidden = self.dropout(hidden)

        # Pass through classifier
        logits = self.classifier(hidden)
        probas = self.sigmoid(logits)

        return probas


# Custom loss function without additional regularization
class OrdinalLoss(nn.Module):
    def __init__(self):
        super(OrdinalLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, predictions, targets):
        norm_targets = targets.float() / (predictions.shape[1])
        norm_targets = norm_targets.view(-1, 1)  # Reshape to match predictions
        return self.mse(predictions, norm_targets)


# Dataset class remains the same
class OrdinalSentimentData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.sentence_text
        bias_score = dataframe.bias_score
        # article_bias = dataframe.article_bias
        # source_bias = dataframe.source_bias
        # combined_bias = bias_score * 0.6 + article_bias * 0.25 + source_bias * 0.15

        self.targets = bias_score.astype(int).tolist()
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

        return {
            'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(inputs["token_type_ids"], dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.long)
        }


# Modified training function with gradient clipping
def train(model, optimizer, loss_function, training_loader, device, clip_grad_norm=1.0):
    model.train()
    tr_loss = 0
    nb_tr_steps = 0
    nb_tr_examples = 0

    for _, data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.long)

        outputs = model(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids)
        loss = loss_function(outputs, targets)  # Pass model for L2 regularization

        tr_loss += loss.item()
        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)

        if _ % 100 == 0:
            print(f"Training Loss after {nb_tr_steps} steps: {tr_loss / nb_tr_steps}")

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

        optimizer.step()

    return tr_loss / nb_tr_steps


# Testing function remains mostly the same
def test(model, loss_function, testing_loader, device):
    model.eval()
    test_loss = 0
    predictions = []
    actual_labels = []

    with torch.no_grad():
        for data in testing_loader:
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            targets = data['targets'].to(device)

            outputs = model(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids)
            loss = loss_function(outputs, targets)
            test_loss += loss.item()

            # 修改预测方法
            pred_probs = outputs.cpu().numpy()
            pred_labels = np.argmax(pred_probs, axis=1)

            predictions.extend(pred_labels)
            actual_labels.extend(targets.cpu().numpy())

    # 计算1分误差内的准确率
    accuracy = sum(abs(x - y) <= 0.5 for x, y in zip(predictions, actual_labels)) / len(predictions)
    mae = np.mean(np.abs(np.array(predictions) - np.array(actual_labels)))

    print(f"Test Loss: {test_loss}, Accuracy: {accuracy}, MAE: {mae}")

    return accuracy, mae


def main():
    # Configuration
    MAX_LEN = 256
    TRAIN_BATCH_SIZE = 8
    VALID_BATCH_SIZE = 4
    LEARNING_RATE = 2e-05
    EPOCHS = 3

    # Load and prepare data
    train_raw_data = pd.read_csv('./news_bias_dataset/preprocessed_dataset.csv', delimiter=',')
    new_df = train_raw_data[['sentence_text', 'bias_score', 'article_bias', 'source_bias']]

    # Split data
    train_size = 0.8
    train_data = new_df.sample(frac=train_size, random_state=200)
    test_data = new_df.drop(train_data.index).reset_index(drop=True)
    train_data = train_data.reset_index(drop=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        "kevinscaria/atsc_tk-instruct-base-def-pos-neg-neut-combined"
    )
    num_classes = len(train_raw_data['bias_score'].unique())

    # Initialize model with LSTM
    model = OrdinalRegressionModel(
        base_model,
        num_classes,
        lstm_hidden_size=256,
        num_lstm_layers=2,
        lstm_dropout=0.2
    ).to(device)

    # Initialize datasets and dataloaders
    training_set = OrdinalSentimentData(train_data, tokenizer, MAX_LEN)
    testing_set = OrdinalSentimentData(test_data, tokenizer, MAX_LEN)

    train_params = {'batch_size': TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}
    test_params = {'batch_size': VALID_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

    # Initialize optimizer with weight decay
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=1,
        verbose=True
    )

    loss_function = OrdinalLoss()

    # Training loop with early stopping
    best_accuracy = 0
    patience = 3
    patience_counter = 0

    for epoch in range(EPOCHS):
        print(f"Training Epoch: {epoch + 1}")
        train_loss = train(model, optimizer, loss_function, training_loader, device)
        print(f"Epoch {epoch + 1} - Average Training Loss: {train_loss}")

        print("Evaluating on test set...")
        accuracy, mae = test(model, loss_function, testing_loader, device)

        # Learning rate scheduling
        scheduler.step(train_loss)

        # Early stopping
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), "best_ordinal_regression_model.bin")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    print(f"Best accuracy achieved: {best_accuracy}")


if __name__ == "__main__":
    main()