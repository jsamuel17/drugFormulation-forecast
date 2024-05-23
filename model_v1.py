# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from imblearn.over_sampling import SMOTE, SVMSMOTE


# %%
class DataPreparation:
    def __init__(
        self,
        mol2vec_path,
        descriptors_2d_path,
        include_2d_descriptors=True,
        selected_2d_features=None,
        batch_size=32,
        train_val_split=0.15,
        train_split=0.85,
    ):
        self.mol2vec_path = mol2vec_path
        self.descriptors_2d_path = descriptors_2d_path
        self.include_2d_descriptors = include_2d_descriptors
        self.selected_2d_features = selected_2d_features
        self.batch_size = batch_size
        self.train_val_split = train_val_split
        self.train_split = train_split
        self.train_dataloader, self.val_dataloader, self.test_dataloader = (
            self.prepare_data()
        )

    def prepare_data(self, seed=42):

        def _collate_fn(batch):
            api_features = torch.stack([item["api_features"] for item in batch])
            outcomes = torch.stack([item["outcome"] for item in batch])
            # Handling variable number of excipients by padding
            excipient_features = [item["excipient_features"] for item in batch]
            excipient_features_padded = pad_sequence(
                excipient_features, batch_first=True
            )
            return api_features, excipient_features_padded, outcomes

        # Load and preprocess data
        self.combined_features = self.load_and_combine_features()
        # Split data
        train_val_data, test_data = train_test_split(
            self.combined_features, test_size=self.train_val_split, random_state=seed
        )
        train_data, val_data = train_test_split(
            train_val_data, test_size=1 - self.train_split, random_state=seed
        )

        # Initialize datasets
        train_dataset = APIExcpDataset(train_data)
        val_dataset = APIExcpDataset(val_data)
        test_dataset = APIExcpDataset(test_data)

        # Initialize dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=_collate_fn,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=_collate_fn,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=_collate_fn,
        )
        return train_dataloader, val_dataloader, test_dataloader

    def load_and_combine_features(self):
        mol2vec_df = pd.read_csv(self.mol2vec_path)
        _2d_df = pd.read_csv(self.descriptors_2d_path)

        # TODO -- Simple NaN removal from 2D dataset
        remove_nan = set(
            [i.split(".1")[0] for i in _2d_df.loc[:, _2d_df.isnull().any()].columns]
        )
        _2d_df = _2d_df.drop(columns=list(remove_nan) + [i + ".1" for i in remove_nan])

        # Preparing Mol2Vec data
        api_features_mol2vec = mol2vec_df.iloc[:, :100].values
        excipient_features_mol2vec = mol2vec_df.iloc[:, 100:200].values
        outcomes_mol2vec = mol2vec_df["Outcome1"].values

        mol2vec_data = [
            (api_features, [excipient_features], outcome)
            for api_features, excipient_features, outcome in zip(
                api_features_mol2vec, excipient_features_mol2vec, outcomes_mol2vec
            )
        ]

        combined_features = mol2vec_data
        # Preparing 2D-Descriptors data
        if self.include_2d_descriptors:
            if self.selected_2d_features is not None:
                if isinstance(
                    self.selected_2d_features[0], int
                ):  # If indices are provided
                    _2d_df = _2d_df.iloc[:, self.selected_2d_features]
                else:  # If column names are provided
                    _2d_df = _2d_df[self.selected_2d_features]

            ## TODO HARD CODED, FIX LATER
            num_feature_columns = (_2d_df.shape[1] - 5) // 2
            api_features_2d = _2d_df.iloc[:, :num_feature_columns].values
            excipient_features_2d = _2d_df.iloc[
                :, num_feature_columns : 2 * num_feature_columns
            ].values
            outcomes_2d = _2d_df["Outcome1"].values

            _2d_data = [
                (api_features, [excipient_features], outcome)
                for api_features, excipient_features, outcome in zip(
                    api_features_2d, excipient_features_2d, outcomes_2d
                )
            ]

            combined_features = [
                (
                    np.concatenate((api_m, api_2d)),
                    np.concatenate((exc_m, exc_2d), axis=1),
                    outcome,
                )
                for (api_m, exc_m, outcome), (api_2d, exc_2d, _) in zip(
                    mol2vec_data, _2d_data
                )
            ]
        # TODO - fix output if not includign 2d
        return combined_features

    def get_label_count(self):
        label0 = len([data[2] for data in self.combined_features if data[2] == 0])
        label1 = len([data[2] for data in self.combined_features if data[2] == 1])
        return label0, label1


class APIExcpDataset(Dataset):
    def __init__(self, data):
        """
        Initializes the dataset.
        - data: List of tuples (API features, List of excipient features, Outcome).
            Example: [(api_features), [(excipient_features),...], (outcome)]

        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        api_features, excipient_features, outcome = self.data[idx]
        return {
            "api_features": torch.tensor(api_features, dtype=torch.float),
            "excipient_features": torch.tensor(excipient_features, dtype=torch.float),
            "outcome": torch.tensor(outcome, dtype=torch.float),
        }


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.api_embedding = nn.Linear(input_size, hidden_size)
        self.excipient_rnn = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(
            hidden_size * 2, 1
        )  # Combining API and excipients' hidden states
        self.sigmoid = nn.Sigmoid()

    def forward(self, api_features, excipient_features):
        api_embedded = self.api_embedding(api_features)
        _, (hidden, _) = self.excipient_rnn(excipient_features)
        excipient_embedded = hidden[-1, :, :]
        combined = torch.cat((api_embedded, excipient_embedded), dim=1)
        output = self.fc(self.dropout(combined))
        compatibility = self.sigmoid(output)  # UNDO IF USING nn.BCELoss
        return compatibility  # UNDO IF USING nn.BCELoss
        # return output # UNDO IF USING nn.BCEWithLogitsLoss


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for data in dataloader:
        api_features, excipient_features, labels = (
            data[0].to(device),
            data[1].to(device),
            data[2].to(device),
        )

        optimizer.zero_grad()
        outputs = model(api_features, excipient_features)  # UNDO IF USING nn.BCELoss
        # logits = model(api_features, excipient_features) # UNDO IF USING nn.BCEWithLogitsLoss
        # outputs = torch.sigmoid(logits) # UNDO IF USING nn.BCEWithLogitsLoss
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * api_features.size(0)
    return total_loss / len(dataloader.dataset)


def evaluate_loss(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data in dataloader:
            api_features, excipient_features, labels = (
                data[0].to(device),
                data[1].to(device),
                data[2].to(device),
            )

            outputs = model(
                api_features, excipient_features
            )  # UNDO IF USING nn.BCELoss
            # logits = model(api_features, excipient_features) # UNDO IF USING nn.BCEWithLogitsLoss
            # outputs = torch.sigmoid(logits) # UNDO IF USING nn.BCEWithLogitsLoss
            loss = criterion(outputs.squeeze(), labels)
            total_loss += loss.item() * api_features.size(0)
    return total_loss / len(dataloader.dataset)


def evaluate_performance(model, dataloader, device):
    def calculate_metrics(outputs, labels):
        # Convert sigmoid outputs to binary predictions
        predicted_labels = (
            outputs > 0.5
        ).int()  # Threshold is 0.5 probablity, 1 if > 0.5, 0 otherwise
        classes = [0, 1]

        acc = accuracy_score(labels.cpu(), predicted_labels.cpu())
        f1 = f1_score(labels.cpu(), predicted_labels.cpu(), zero_division=0)
        precision = precision_score(
            labels.cpu(), predicted_labels.cpu(), zero_division=0
        )
        recall = recall_score(labels.cpu(), predicted_labels.cpu(), zero_division=0)
        cm = confusion_matrix(labels.cpu(), predicted_labels.cpu(), labels=classes)
        return acc, f1, precision, recall, cm

    model.eval()
    total_metrics = {"accuracy": 0, "f1": 0, "precision": 0, "recall": 0}
    total_samples = 0
    full_confusion_matrix = None
    with torch.no_grad():
        for data in dataloader:
            api_features, excipient_features, labels = (
                data[0].to(device),
                data[1].to(device),
                data[2].to(device),
            )
            outputs = model(api_features, excipient_features).squeeze()
            acc, f1, prec, rec, cm = calculate_metrics(outputs, labels)
            total_samples += labels.size(0)
            total_metrics["accuracy"] += acc * labels.size(0)
            total_metrics["f1"] += f1 * labels.size(0)
            total_metrics["precision"] += prec * labels.size(0)
            total_metrics["recall"] += rec * labels.size(0)

            if full_confusion_matrix is None:
                full_confusion_matrix = cm
            else:
                full_confusion_matrix += cm

    for metric in total_metrics:
        total_metrics[metric] /= total_samples
    return total_metrics, full_confusion_matrix


# %%
# Load data
mol2vec_path = "mol2vec_data.csv"
descriptors_2d_path = "2D_data.csv"

data_prep = DataPreparation(
    mol2vec_path=mol2vec_path,
    descriptors_2d_path=descriptors_2d_path,
    include_2d_descriptors=True,
    selected_2d_features=None,
    batch_size=32,
)

train_dataloader, val_dataloader, test_dataloader = (
    data_prep.train_dataloader,
    data_prep.val_dataloader,
    data_prep.test_dataloader,
)

# Model, Loss, Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Handle inbalance
# class0_count, class1_count = data_prep.get_label_count()
# pos_weight = torch.tensor([class0_count / class1_count], dtype=torch.float).to(
#     device
# )  ## MORE 0 than 1
input_size = len(train_dataloader.dataset.data[0][0])
hidden_size = 128
num_layers = 2
dropout = 0.5

model = LSTMModel(input_size, hidden_size, num_layers, dropout).to(device)
criterion = nn.BCELoss()  # # REQUIRES SIGMOID
# criterion = nn.BCEWithLogitsLoss(
#     pos_weight=pos_weight
# )  # Handle inbalance, DOESNT REQUIRE SIGMOID
# This means during the computation of the loss, the errors associated
# with the minority class are considered more significant,
# effectively addressing the class imbalance.
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 10
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    train_loss = train(model, train_dataloader, criterion, optimizer, device)
    val_loss = evaluate_loss(model, val_dataloader, criterion, device)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(
        f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
    )

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.title("Training and Validation Losses")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# Evaluating Loss & Performance on Test Set
test_loss = evaluate_loss(model, test_dataloader, criterion, device)
print(f"Test Loss: {test_loss:.4f}")
metrics, c_matrix = evaluate_performance(model, test_dataloader, device)
print("Test Metrics:", json.dumps(metrics, indent=4, sort_keys=True))
disp = ConfusionMatrixDisplay(confusion_matrix=c_matrix)
disp.plot()
plt.show()


## TODO
# 1. Handle Inbalance of label 0 > 1
# 2. Normalize / Standardize Features
# 3. Feature reduction (i.e., remove 2D Descriptors)
# 4. HyperParameter Tuning
