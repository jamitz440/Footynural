import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from datetime import datetime


# Step 1: Prepare the data

class ScoreDataset(Dataset):
    def __init__(self, data, team_to_idx):
        self.data = data
        self.team_to_idx = team_to_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        date = datetime.strptime(self.data[idx][4], "%Y-%m-%d")
        date_values = (date.year, date.month, date.day)
        team_indices = (self.team_to_idx[self.data[idx][0]], self.team_to_idx[self.data[idx][3]])
        input_data = torch.tensor(team_indices + date_values, dtype=torch.float32)
        output_data = torch.tensor(self.data[idx][1:3], dtype=torch.float32)
        return input_data, output_data





with open('cleaned_data.pkl', 'rb') as file:
        data = pickle.load(file)


def convert_to_int(value):
    try:
        return int(value)
    except ValueError:
        return 0  # or any other default value

data = [(match[0], convert_to_int(match[1]), convert_to_int(match[2]), match[3], match[4]) for match in data]

# Create a set of unique team names
unique_teams = set()
for match in data:
    unique_teams.add(match[0])
    unique_teams.add(match[3])



# Create a dictionary to map team names to indices
team_to_idx = {team: idx for idx, team in enumerate(unique_teams)}

# Data preparation
import random
random.shuffle(data)

split_ratio = 0.8
split_index = int(len(data) * split_ratio)

train_data = data[:split_index]
val_data = data[split_index:]

# Create ScoreDataset instances for training and validation
batch_size = 32
train_dataset = ScoreDataset(train_data, team_to_idx)
val_dataset = ScoreDataset(val_data, team_to_idx)

# Create DataLoaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Step 2: Define the neural network architecture
class ScorePredictor(nn.Module):
    def __init__(self):
        super(ScorePredictor, self).__init__()
        self.fc1 = nn.Linear(5, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 2)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x)) 
        x = self.leaky_relu(self.fc2(x)) 
        x = self.leaky_relu(self.fc3(x)) 
        x = self.fc4(x)
        return x

model = ScorePredictor()

# Step 3: Define the loss function and optimizer
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 4: Train the neural network
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            val_batch_loss = loss_function(outputs, targets)
            val_loss += val_batch_loss.item()

            val_loss /= len(val_loader)
            print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.4f}")

# Step 5: save the trained model
torch.save(model.state_dict(), 'model.pth')

# Step 6: Using the model for prediction
# ...

# Step 6: Using the model for prediction
def predict_score(model, team_a, team_b, date_values):
    model.eval()
    team_a_idx = torch.tensor([team_to_idx[team_a]], dtype=torch.float32)
    team_b_idx = torch.tensor([team_to_idx[team_b]], dtype=torch.float32)
    date = torch.tensor(date_values, dtype=torch.float32)
    input_tensor = torch.cat((team_a_idx, team_b_idx, date), dim=0).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
    return output.squeeze().tolist()

# Evaluate the model using the excluded data# Evaluate the model using the excluded data
correct_predictions = 0
total_predictions = len(val_data)

for match in val_data:
    team_a = match[0]
    team_b = match[3]
    actual_scores = (match[1], match[2])
    date_str = match[4]
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    date_values = (date_obj.year, date_obj.month, date_obj.day)
    
    predicted_scores = predict_score(model, team_a, team_b, date_values)
    
    actual_result = 1 if actual_scores[0] > actual_scores[1] else -1 if actual_scores[0] < actual_scores[1] else 0
    predicted_result = 1 if predicted_scores[0] > predicted_scores[1] else -1 if predicted_scores[0] < predicted_scores[1] else 0

    if actual_result == predicted_result:
        correct_predictions += 1

    print(f"Actual scores for {team_a} vs {team_b}: {actual_scores}")
    print(f"Predicted scores for {team_a} vs {team_b}: {predicted_scores}")
    print("\n")

print(f"Total correct predictions: {correct_predictions}")
print(f"Total predictions: {total_predictions}")
print(f"Accuracy: {correct_predictions / total_predictions:.2%}")



