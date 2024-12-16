from vit_pytorch import ViT
import torch
import torch.nn as nn
import torch.optim as optim
import csv
import random
import numpy as np
import socket
from sklearn.metrics import precision_score, recall_score, f1_score

# Define Generator model
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Fully connected layers to transform input features
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),    # Input size is 87 features
            nn.ReLU(),
            nn.Linear(128, 128 * 128 * 3), # Output a large vector for fake data
            nn.ReLU()
        )
        # Vision Transformer (ViT) layer
        self.vit = ViT(
            image_size=128, # Image size
            patch_size=32,  # Patch size
            num_classes=output_dim, # Output fake features
            dim=16,         # Dimension of each patch embedding
            depth=4,        # Number of transformer layers
            heads=4,        # Number of attention heads
            mlp_dim=64,     # MLP hidden dimension
            dropout=0.2,    # Dropout probability
            emb_dropout=0.2 # Dropout for embeddings
        )
        self.sg = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)   # Pass through fully connected layers to generate data
        x = x.reshape(1, 3, 128, 128)   # Reshape to image format for ViT input
        x = self.vit(x)  # Pass through ViT to generate fake data
        x = self.sg(x)   # Apply ReLU activation function
        return x

# Define Discriminator model
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        # Fully connected layers for feature transformation
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128 * 128 * 3),  # Output benign/attack data
            nn.ReLU()
        )
        # Vision Transformer (ViT) layer
        self.vit = ViT(
            image_size=128, # Image size
            patch_size=32,  # Patch size
            num_classes=1,  # Output for benign/attack classification
            dim=16,
            depth=4,
            heads=4,
            mlp_dim=64,
            dropout=0.2,
            emb_dropout=0.2
        )
        self.sg = nn.Sigmoid() # Sigmoid activation for probability output

    def forward(self, x):
        x = self.fc(x)   # Transform features through fully connected layers
        x = x.reshape(1, 3, 128, 128)  # Reshape to image format
        x = self.vit(x)  # Use ViT for determining benign/attack
        x = self.sg(x)   # Apply Sigmoid activation for output probability
        return x

# Define hyperparameters
input_dim = 87  # Input dimension
output_dim = 1  # Output dimension (binary: benign/attack)
batch_size = 32
lr = 0.000001   # Learning rate
num_epochs = 10 # Number of epochs
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   # Use GPU if available

# Initialize Generator and Discriminator models
generator_attack = Generator(input_dim, 87).to(device)
generator_benign = Generator(input_dim, 87).to(device)
discriminator_attack = Discriminator(input_dim).to(device)
discriminator_benign = Discriminator(input_dim).to(device)

# Define loss function and optimizers
criterion = nn.BCELoss() # Binary Cross-Entropy loss
optimizer_G_attack = optim.Adam(generator_attack.parameters(), lr=lr)
optimizer_D_attack = optim.Adam(discriminator_attack.parameters(), lr=lr)
optimizer_G_benign = optim.Adam(generator_benign.parameters(), lr=lr)
optimizer_D_benign = optim.Adam(discriminator_benign.parameters(), lr=lr)

# Data processing function to convert CSV data
def data_processing(csv_read):
    data_p = []
    i = []
    for row in csv_read:
        row[1] = 0
        row[2] = int.from_bytes(socket.inet_aton(row[2]), byteorder='big')  # Convert IP to integer
        row[4] = int.from_bytes(socket.inet_aton(row[4]), byteorder='big')  # Convert IP to integer
        row[7] = 0
        # Convert features to float
        for j, s in enumerate(row[0:87]):
            try:
                row[j] = float(s)
            except ValueError:
                row[j] = 0
        a = list(map(float, row[0:87]))  # Extract first 87 features as float
        data_p.append(a)
        i.append(row[-1])  # Append label
    # Normalize data
    mu = np.mean(data_p, axis=0)
    epsilon = 1e-8  # Small constant to avoid division by zero
    sigma = np.std(data_p, axis=0) + epsilon
    c = (data_p - mu) / sigma  # Standardize features
    c[np.isnan(c)] = 0  # Handle NaN values
    c = list(c)
    return c

# Load data from CSV
def load_data():
    global feature1  # benign data
    global feature2  # attack data
    global feature3
    feature1 = []  # benign data features
    feature2 = []  # Attack data features
    i = []  # Labels
    filepath = 'CIC19data.csv'
    with open(filepath, 'r') as data:
        csv_read2 = csv.reader(data)
        for row in csv_read2:
            i.append(row[-1])
    with open(filepath, 'r') as data:
        csv_read2 = csv.reader(data)
        csv_data = data_processing(csv_read2)  # Process CSV data
        # Separate real and fake data based on labels
        for j, row in enumerate(i):
            if row == 'BENIGN':
                feature1.append(list(csv_data[j]))  # benign data
            else:
                feature2.append(list(csv_data[j]))  # attack data

if __name__ == '__main__':
    load_data()
    feature11 =  []
    feature22 =  []
    test1 = []
    testlable1 = []
    test2 = []
    testlable2 = []
    test =  []
    testlable =  []
    # Shuffle benign data samples
    index = [i for i in range(len(feature1))]
    random.shuffle(index)
    feature44 = []
    for i in index:
        feature44.append(feature1[i])
    for i, row in enumerate(feature44):   # Split benign data into training and testing
        if i < 4000:
            feature11.append(row)
        elif i > 3999 and i < 58500:
            test2.append(row)
            testlable2.append(1)
            test.append(row)
            testlable.append(1)

    # Shuffle attack data samples
    index = [i for i in range(len(feature2))]
    random.shuffle(index)
    feature33 = []
    for i in index:
        feature33.append(feature2[i])
    for i, row in enumerate(feature33):   # Split attack data into training and testing
        if i < 4000:
            feature22.append(row)
        elif i > 3999 and i < 58500:
            test1.append(row)
            testlable1.append(0)
            test.append(row)
            testlable.append(0)

    # Load pre-trained discriminator weights
    discriminator_attack.load_state_dict(torch.load('discriminator_CIC19_attack.pth'))
    discriminator_benign.load_state_dict(torch.load('discriminator_CIC19_benign.pth'))
    discriminator_attack.eval()  # Set model to evaluation mode
    discriminator_benign.eval()  # Set model to evaluation mode

    # Iterate over test data and calculate metrics
    n = 0
    y_true = []
    y_pred = []
    for i, test_data in enumerate(test):
        test_data = torch.Tensor(np.array(test_data)).to(device)  # Convert to tensor and send to device
        test_l = torch.Tensor(np.array(testlable[i]))
        discriminator_benign = discriminator_benign.to(device)
        discriminator_attack = discriminator_attack.to(device)
        c = discriminator_benign(test_data)  # Get output from discriminator_benign
        d = discriminator_attack(test_data)  # Get output from discriminator_attack
        # observation and comparison process
        print(c)
        print(d)
        print(len(test))
        print(testlable[i])

        if d > c and testlable[i] == 0:   # attack data predicted as attack
            n +=1
            y_true.append(0)
            y_pred.append(0)
        elif d < c and testlable[i] == 1: # benign data predicted as benign
            n += 1
            y_true.append(1)
            y_pred.append(1)
        else: # Otherwise, assign label based on discriminator decision
            y_true.append(testlable[i])
            y_pred.append(1 if d < c else 0)
    # Calculate accuracy, precision, recall, and F1 score
    accuracy = n / len(test)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

# Prediction results
# Accuracy: 0.9983
# Precision: 0.9845
# Recall: 0.9995
# F1 Score: 0.9920