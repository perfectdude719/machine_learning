import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Hyperparameters for tuning
default_params = {
    "hidden_layers": 1,  # Number of hidden layers
    "hidden_size": 10,   # Number of neurons per hidden layer
    "batch_size": 15,    # Batch size for SGD
    "learning_rate": 0.01  # Learning rate
}

# Define the ranges for each hyperparameter
param_ranges = {
    "hidden_layers": [1, 3, 5, 7],
    "hidden_size": [10, 20, 30, 40],
    "batch_size": [15, 30, 60, 120],
    "learning_rate": [0.1, 0.01, 0.001, 0.0001]
}

# Load MNIST dataset with transformations
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

# Split the training dataset into train and validation sets
train_data, val_data = train_test_split(train_dataset, test_size=0.2, stratify=train_dataset.targets)

# Function to build a model with variable hidden layers
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_layers, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        layers = []
        for i in range(hidden_layers):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_size))
            else:
                layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, num_classes))  # Final output layer
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten input
        return self.network(x)

# Training and evaluation function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    y_true, y_pred = [], []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # Move to GPU/CPU
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = correct_train / total_train * 100
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        # Validation phase
        model.eval()
        running_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        avg_val_loss = running_loss / len(val_loader)
        val_accuracy = correct_val / total_val * 100
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
              f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
    
    return train_losses, val_losses, train_accuracies, val_accuracies, y_true, y_pred

# Hyperparameter tuning function
def tune_hyperparameters(default_params, param_ranges, num_epochs=10):
    results = []
    
    # Iterate over each parameter
    for param, values in param_ranges.items():
        for value in values:
            # Update the parameter value
            params = default_params.copy()
            params[param] = value
            
            # Update DataLoader if batch_size changes
            batch_size = params["batch_size"]
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

            # Build the model
            model = NeuralNetwork(
                input_size=28 * 28,
                hidden_layers=params["hidden_layers"],
                hidden_size=params["hidden_size"],
                num_classes=10
            ).to(device)
            
            # Define optimizer with new learning rate
            optimizer = optim.SGD(model.parameters(), lr=params["learning_rate"])
            criterion = nn.CrossEntropyLoss()
            
            # Train the model
            train_losses, val_losses, train_accuracies, val_accuracies, y_true, y_pred = train_model(
                model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs
            )
            
            # Record results
            results.append({
                "param": param,
                "value": value,
                "train_accuracy": train_accuracies[-1],
                "val_accuracy": val_accuracies[-1],
                "train_loss": train_losses[-1],
                "val_loss": val_losses[-1],
                "y_true": y_true,
                "y_pred": y_pred
            })
            
            print(f"Completed training for {param} = {value}")
    
    return results

# Perform hyperparameter tuning
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results = tune_hyperparameters(default_params, param_ranges)

# Convert results to DataFrame for easy analysis
df = pd.DataFrame(results)

# 1. Plot training and validation loss versus the parameter values
def plot_loss(results, param_name):
    plt.figure(figsize=(12, 6))
    sns.lineplot(x="value", y="train_loss", data=results, label="Train Loss")
    sns.lineplot(x="value", y="val_loss", data=results, label="Validation Loss")
    plt.title(f'Training and Validation Loss vs. {param_name}')
    plt.xlabel(param_name)
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# 2. Plot training and validation accuracy versus the parameter values
def plot_accuracy(results, param_name):
    plt.figure(figsize=(12, 6))
    sns.lineplot(x="value", y="train_accuracy", data=results, label="Train Accuracy")
    sns.lineplot(x="value", y="val_accuracy", data=results, label="Validation Accuracy")
    plt.title(f'Training and Validation Accuracy vs. {param_name}')
    plt.xlabel(param_name)
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.show()

# 3. Plot confusion matrix for each trained model
def plot_confusion_matrix(y_true, y_pred, param_value):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
    plt.title(f'Confusion Matrix for Parameter Value {param_value}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Plot loss and accuracy for each parameter
for param in param_ranges:
    plot_loss(df, param)
    plot_accuracy(df, param)

    # Plot confusion matrix for each parameter value
    for result in results:
        if result["param"] == param:
            plot_confusion_matrix(result["y_true"], result["y_pred"], result["value"])

