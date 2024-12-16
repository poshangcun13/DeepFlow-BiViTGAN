from vit_pytorch import ViT
import torch
import torch.nn as nn
import torch.optim as optim
import csv
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import socket

# Define the Generator model
class ViTClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ViTClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, 128 * 128 * 3) # Output a large vector for fake data
        # Vision Transformer layer
        self.vit = ViT(
            image_size=128, # Image size
            patch_size=32,  # Patch size
            num_classes=1,  # Output for benign/attack classification
            dim=16,
            depth=4,
            heads=2,
            mlp_dim=64,
            dropout=0.2,
            emb_dropout=0.2
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 3, 128, 128)  # Reshape to image format
        x = self.vit(x)
        return x

# Define hyperparameters
input_dim   = 87   # Input dimension (19 features)
num_classes = 1    # 1 class for fake/real
batch_size  = 32
lr = 0.000001      # Learning rate
num_epochs  = 30   # Number of training epochs
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def data_processing(csv_read):
    # global data_p
    data_p = []
    i = []
    for row in csv_read:
        row[1] = 0
        row[2] = int.from_bytes(socket.inet_aton(row[2]), byteorder='big')
        row[4] = int.from_bytes(socket.inet_aton(row[4]), byteorder='big')
        row[7] = 0
        for j, s in enumerate(row[0:87]):
            try:
                row[j] = float(s)
            except ValueError:
                row[j] = 0
        a = list(map(float, row[0:87]))
        data_p.append(a)
        i.append(row[-1])
    # Normalize benign data
    mu = np.mean(data_p, axis=0)
    epsilon = 1e-8  # small constant to avoid division by zero
    sigma   = np.std(data_p, axis=0) + epsilon
    c = (data_p - mu) / sigma
    c[np.isnan(c)] = 0  # Replace NaN values with zero
    c = list(c)
    return c

def load_data():
    global feature1 # benign dataset
    global feature2 # attack dataset
    global feature3 # test   dataset
    feature1 = []   # benign data features
    feature2 = []   # attack data features
    feature3 = []   # test   dataset
    i=[]
    filepath = 'CIC19data.csv'
    # Read dataset from CSV file
    with open(filepath, 'r') as data:
        csv_read2 = csv.reader(data)
        for row in csv_read2:
            i.append(row[-1])

    with open(filepath, 'r') as data:
        csv_read2 = csv.reader(data)
        csv_data=data_processing(csv_read2)
        print(len(csv_data),len(i))
        for j,row in enumerate(i):
            #a = list(map(float, row[8:87]))
            if row == 'BENIGN':
                    feature1.append(list(csv_data[j]))
            else:
                    feature2.append(list(csv_data[j]))

# Main execution
if __name__ == '__main__':
    load_data()

    train = []
    trainlable = []
    test = []
    testlable  = []

    # Shuffle and split benign data
    index = [i for i in range(len(feature1))]
    random.shuffle(index)
    feature44 = []
    for i in index:
        feature44.append(feature1[i])
    for i, row in enumerate(feature44):  # Split benign data into training and testing
        if i < 4000:
            train.append(row)
            trainlable.append(1)
        elif i > 3999 and i < 58500:
            test.append(row)
            testlable.append(1)

    # Shuffle attack samples data
    index = [i for i in range(len(feature2))]
    random.shuffle(index)
    feature33 = []
    for i in index:
        feature33.append(feature2[i])
    # From shuffled attack data, take 4000 samples
    for i, row in enumerate(feature33):  # Split attack data into training and testing
        if i < 4000:
            train.append(row)
            trainlable.append(0)
        elif i > 3999 and i < 58500:
            test.append(row)
            testlable.append(0)

# Initialize the model
model     = ViTClassifier(input_dim, num_classes).to(device)
criterion = nn.BCEWithLogitsLoss()      # Binary classification loss
optimizer = optim.Adam(model.parameters(), lr=lr)

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels   = labels
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label   = torch.tensor(self.labels[idx], dtype=torch.float32)
        return feature, label

# Create DataLoader
train_dataset = CustomDataset(train, trainlable)
test_dataset  = CustomDataset(test, testlable)
train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader   = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 训练函数
def train_model(model, criterion, optimizer, train_loader, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")
        for features, labels in progress_bar:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features).squeeze(1)  # 去掉多余维度
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}")

# Testing function
def evaluate_model(model, test_loader, device):
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features).squeeze(1)
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
    # Convert to NumPy arrays
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    # Compute metrics
    accuracy  = np.mean(all_predictions == all_labels)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    # Print results
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

# Train and evaluate
if __name__ == '__main__':
    train_model(model, criterion, optimizer, train_loader, num_epochs, device)
    evaluate_model(model, test_loader, device)

model_save_path = "vit_classifier-CIC19.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Prediction results
# Test Accuracy: 0.9969
# Precision: 0.9841
# Recall: 0.9963
# F1 Score: 0.9901