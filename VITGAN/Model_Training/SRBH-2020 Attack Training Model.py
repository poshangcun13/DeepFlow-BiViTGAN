from vit_pytorch import ViT
import torch
import torch.nn as nn
import torch.optim as optim
import csv
import random
import numpy as np
from tqdm import tqdm

# Define the Generator model
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Fully connected layers to transform input features
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),    # Input size is 19 features
            nn.ReLU(),
            nn.Linear(128, 128 * 128 * 3), # Output a large vector for fake data
            nn.ReLU()
        )
        # Vision Transformer (ViT) layer
        # Vision Transformer layer
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

# Define the Discriminator model
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        # Fully connected layers to transform features
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128 * 128 * 3),  # Output as benign/attack data
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
        self.sg = nn.Sigmoid() # Sigmoid activation for output probability

    def forward(self, x):
        x = self.fc(x)   # Transform features through fully connected layers
        x = x.reshape(1, 3, 128, 128)  # Reshape to image format
        x = self.vit(x)  # Use ViT to predict benign/attack
        x = self.sg(x)   # Apply Sigmoid activation for output probability
        return x

# Define hyperparameters
input_dim = 19  # Input dimension (19 features)
output_dim = 1  # Output dimension (1 class for fake/real)
batch_size = 32
lr = 0.000001   # Learning rate
num_epochs = 10 # Number of training epochs
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   # Use GPU if available

# Initialize generator and discriminator models
generator_attack = Generator(input_dim, 19).to(device)
generator_benign = Generator(input_dim, 19).to(device)
discriminator_attack = Discriminator(input_dim).to(device)
discriminator_benign = Discriminator(input_dim).to(device)

# Define loss function and optimizers
criterion = nn.BCELoss() # Binary Cross-Entropy loss
optimizer_G_attack = optim.Adam(generator_attack.parameters(), lr=lr)
optimizer_D_attack = optim.Adam(discriminator_attack.parameters(), lr=lr)
optimizer_G_benign = optim.Adam(generator_benign.parameters(), lr=lr)
optimizer_D_benign = optim.Adam(discriminator_benign.parameters(), lr=lr)

# Function to load data
def load_data():
    global feature1 # benign dataset
    global feature2 # Attack dataset
    global label0
    global label1
    feature1 = []   # benign data features
    feature2 = []   # Attack data features
    data1 = []      # benign data
    data2 = []      # Attack data
    label0 = []     # Attack labels
    label1 = []     # benign labels
    filename = 'SRBHdata.csv'
    # Read dataset from CSV file
    with open(filename, 'r') as data:
        csv_read = csv.reader(data)
        for row in csv_read:
            if int(row[19]) == 1:    # Label 1 for benign data
                label1.append(1)
                data1.append(list(map(float, row[:19])))
            elif int(row[19]) == 0:  # Label 0 for attack data
                data2.append(list(map(float, row[:19])))

        # Normalize benign data (feature1)
        mu = np.mean(data1, axis=0)
        epsilon = 1e-8  # small constant to avoid division by zero
        sigma = np.std(data1, axis=0) + epsilon
        c = (data1 - mu) / sigma
        c[np.isnan(c)] = 0  # Replace NaN values with zero
        feature1 = list(c)
        # Normalize attack data (feature2)
        mu = np.mean(data2, axis=0)
        sigma = np.std(data2, axis=0) + epsilon
        b = (data2 - mu) / sigma
        b[np.isnan(b)] = 0  # Replace NaN values with zero
        feature2 = list(b)

# Main execution
if __name__ == '__main__':
    load_data()
    feature11 = []
    feature22 = []
    test1 = []
    testlable1 = []
    test2 = []
    testlable2 = []
    test = []
    testlable = []

    # Shuffle and split benign data
    index = [i for i in range(len(feature1))]
    random.shuffle(index)
    feature44 = []
    for i in index:
        feature44.append(feature1[i])
    for i, row in enumerate(feature44):  # Split benign data into training and testing
        if i < 4000:
            feature11.append(row)
        elif i > 3999 and i < 4500:
            test2.append(row)
            testlable2.append(1)
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
            feature22.append(row)
        elif i > 3999 and i < 4500:
            test1.append(row)
            testlable1.append(0)
            test.append(row)
            testlable.append(0)

    # Train attack data model
    for epoch in range(num_epochs):
    # Display training progress bar
        for i, attack_data in tqdm(enumerate(feature22), total=len(feature22),
                                   desc=f"Training Attack Epoch {epoch + 1}/{num_epochs}"):
            attack_data = torch.Tensor(np.array(attack_data)).to(device)
            benign_data = torch.Tensor(np.array(feature11[i % len(feature11)])).to(device)  # Use benign sample data
            # Train generator attack
            optimizer_G_attack.zero_grad()
            generated_true = generator_attack(benign_data)
            pred_true = discriminator_attack(generated_true)
            loss_G_attack = criterion(pred_true, torch.ones_like(pred_true))
            loss_G_attack.backward()
            optimizer_G_attack.step()
            # Train discriminator attack
            optimizer_D_attack.zero_grad()
            pred_fake1_detach = discriminator_attack(generated_true.detach())  # Discriminator prediction of generated data (fake data)
            pred_real1 = discriminator_attack(attack_data)  # Discriminator prediction of attack sample
            pred_true1 = discriminator_attack(benign_data)  # Discriminator prediction of benign sample
            # Discriminator loss: Distinguish between attack data and benign data
            loss_D_attack = criterion(pred_fake1_detach, torch.zeros_like(pred_fake1_detach)) + \
                            criterion(pred_real1, torch.ones_like(pred_real1)) + \
                            criterion(pred_true1, torch.zeros_like(
                                pred_true1))  # Attack samples are predicted as 1 (real), benign samples as 0 (fake)
            loss_D_attack.backward()
            optimizer_D_attack.step()

            if i % 100 == 0:  # Output loss every 100 steps
                print(
                    f"Step {i}, Loss G attack: {loss_G_attack.item():.4f}, Loss D attack: {loss_D_attack.item():.4f}")

        # Evaluate attack accuracy
        n = 0
        for i, test_data in enumerate(test1):
            test_data1 = torch.Tensor(np.array(test_data)).to(device)
            test_l1 = torch.Tensor(np.array(testlable1[i]))
            discriminator_attack = discriminator_attack.to(device)
            c = discriminator_attack(test_data1)
            if c > 0.5:
                n = n + 1
        attack_accuracy = n / 500
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss G attack: {loss_G_attack.item():.4f}, Loss D attack: {loss_D_attack.item():.4f}, attack Accuracy: {attack_accuracy:.4f}")

    # Train benign data model
    for epoch in range(num_epochs):
    # Display training progress bar
        for i, benign_data in tqdm(enumerate(feature11), total=len(feature11),
                                   desc=f"Training benign Epoch {epoch + 1}/{num_epochs}"):
            benign_data = torch.Tensor(np.array(benign_data)).to(device)
            attack_data2 = torch.Tensor(np.array(feature22[i % len(feature22)])).to(device)  # Use fake data

            # Train generator Btrue
            optimizer_G_benign.zero_grad()
            generated_fake = generator_benign(attack_data2)
            pred_fake = discriminator_benign(generated_fake)
            loss_G_benign = criterion(pred_fake, torch.ones_like(pred_fake))
            loss_G_benign.backward()
            optimizer_G_benign.step()

            # Train discriminator Btrue
            optimizer_D_benign.zero_grad()
            pred_fake_detach = discriminator_benign(
                generated_fake.detach())  # Discriminator prediction of generated data (fake data)
            pred_real = discriminator_benign(benign_data)  # Discriminator prediction of benign sample
            pred_fake = discriminator_benign(attack_data2) # Discriminator prediction of attack sample
            # Discriminator loss: Distinguish between attack data and benign data
            loss_D_benign = criterion(pred_fake_detach, torch.zeros_like(pred_fake_detach)) + \
                            criterion(pred_real, torch.ones_like(pred_real)) + \
                            criterion(pred_fake, torch.zeros_like(pred_fake))
            loss_D_benign.backward()
            optimizer_D_benign.step()

            if i % 100 == 0:  # Output loss every 100 steps
                print(
                    f"Step {i}, Loss G benign: {loss_G_benign.item():.4f}, Loss D benign: {loss_D_benign.item():.4f}")

        # Evaluate true data accuracy
        n = 0
        for i, test_data in enumerate(test2):
            test_data2 = torch.Tensor(np.array(test_data)).to(device)
            test_l2 = torch.Tensor(np.array(testlable2[i]))
            discriminator_benign = discriminator_benign.to(device)
            c = discriminator_benign(test_data2)
            if c > 0.5:
                n = n + 1
        true_accuracy = n / 500
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss G benign: {loss_G_benign.item():.4f}, Loss D benign: {loss_D_benign.item():.4f}, True Accuracy: {true_accuracy:.4f}")

    # Final evaluation on test data
    n = 0
    for i, test_data in enumerate(test):
        test_data = torch.Tensor(np.array(test_data)).to(device)
        test_l = torch.Tensor(np.array(testlable[i]))
        discriminator_benign = discriminator_benign.to(device)
        discriminator_attack = discriminator_attack.to(device)
        c = discriminator_benign(test_data)
        d = discriminator_attack(test_data)
        print(c)
        print(d)
        print(len(test))
        print(testlable[i])
        # Condition check and correct prediction count
        if d > c and testlable[i] == 0:
            n += 1
        elif d < c and testlable[i] == 1:
            n += 1
    # Calculate accuracy
    accuracy = n / len(test)
    print(f"Accuracy: {accuracy:.4f}")

    # Save the models
    torch.save(generator_attack.state_dict(), "generator_SR20_attack.pth")
    torch.save(discriminator_attack.state_dict(), "discriminator_SR20_attack.pth")
    torch.save(generator_benign.state_dict(), "generator_SR20_benign.pth")
    torch.save(discriminator_benign.state_dict(), "discriminator_SR20_benign.pth")