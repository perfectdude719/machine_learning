import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,Subset,ConcatDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


# Apply transformations (convert to tensor and normalize)
transform = transforms.Compose([
    transforms.ToTensor(), #convert vector values in range [0,1]
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1] range (z socre)
])

# Load the MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform) #this structure contains data and labels
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

#combine the datasets
combined_dataset=ConcatDataset([train_dataset,test_dataset])

#extract labels
train_labels=train_dataset.targets
test_labels=test_dataset.targets

# Combine labels from both train and test datasets
all_labels = torch.cat((train_labels, test_labels), dim=0)

#split into 60-40 
train_idx, temp_idx = train_test_split(
    range(len(combined_dataset)), test_size=0.4, stratify=all_labels, random_state=42) # stratify 3lshan not to get skewed data


#split again 20-20
# Split temp (40%) into validation (20%) and test (20%)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.5, stratify=all_labels[temp_idx], random_state=42) # returns indices 


# Create Subset objects for train, validation, and test sets
train_data = Subset(combined_dataset, train_idx)
val_data = Subset(combined_dataset, val_idx)
test_data = Subset(combined_dataset, test_idx) 


##recreate dataset w inheirt combatible structure with data loader
# Create DataLoader objects
batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Print split sizes
print(f"Training set size: {len(train_data)}")
print(f"Validation set size: {len(val_data)}")
print(f"Test set size: {len(test_data)}")
print("Data preparation complete!")

# 2 create neural network structure
class Neural_Network_1_hidden_layer(nn.Module):
        def __init__(self, input_size=28*28, hidden_size=10, num_classes=10):
            super(Neural_Network_1_hidden_layer, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size) 
            self.fc2 = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            x = x.view(-1, 28*28)  # Flatten the input
            x = torch.relu(self.fc1(x))  # Apply ReLU activation after first hidden layer
            x = self.fc2(x)  # Output layer
            return x
        
        

# 3. Set device to GPU if available, else fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the model and move it to the selected device
model = Neural_Network_1_hidden_layer().to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)




# 4. Training Process
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0 # this is the cumulative loss during epoch
        correct_train = 0  
        total_train = 0
        
        # Train the model on the training set
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # Move to GPU
            
            optimizer.zero_grad()
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass (compute gradients)
            optimizer.step()  # Update weights

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)  # Get the predicted class
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
                images, labels = images.to(device), labels.to(device)  # Move to GPU
                outputs = model(images)  # Forward pass
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

        avg_val_loss = running_loss / len(val_loader)
        val_accuracy = correct_val / total_val * 100
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
              f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
    
    return train_losses, val_losses, train_accuracies, val_accuracies



# Train the model
train_losses, val_losses, train_accuracies, val_accuracies = train_model(
    model, train_loader, val_loader, criterion, optimizer, num_epochs=10
)



# 5. Plot Training and Validation Loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 6. Plot Training and Validation Accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.show()


# Hyperparameters for tuning
default_params = {
    "hidden_layers": 1,  # Number of hidden layers
    "hidden_size": 10,   # Number of neurons per hidden layer
    "batch_size": 32,    # Batch size for SGD
    "learning_rate": 0.01  # Learning rate
}

# Define the ranges for each hyperparameter
param_ranges = {
    "hidden_layers": [1, 3, 5, 7,9,11],
    "hidden_size": [10, 20, 30, 40, 50 , 60],
    "batch_size": [32, 64, 128, 256,500, 1000 ],
    "learning_rate": [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
}


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
            optimizer.zero_grad() # remove gradients to prevent gradient accumulatioin
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() # add current batch loss
            _, predicted = torch.max(outputs, 1) # index of max probability
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)# keep track of all samples trained

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
                y_true.extend(labels.cpu().numpy()) ##move to cpu for transforming into numpy array 
                y_pred.extend(predicted.cpu().numpy())

        avg_val_loss = running_loss / len(val_loader)
        val_accuracy = correct_val / total_val * 100
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
              f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
    
    # Return all six variables
    return train_losses, val_losses, train_accuracies, val_accuracies, y_true, y_pred



# Function to build a model with variable hidden layers
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_layers, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        layers = []
        for i in range(hidden_layers):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_size)) # for inout layer
            else:
                layers.append(nn.Linear(hidden_size, hidden_size)) ##every layer in between
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, num_classes))  # Final output layer
        self.network = nn.Sequential(*layers) ##link layers

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten input
        return self.network(x)
    
    
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


import pandas as pd

# Convert the results list to a DataFrame
df = pd.DataFrame(results)

# Inspect the columns to ensure they match expected names
print("Columns in DataFrame:", df.columns)

# Verify sample rows to confirm data integrity
print(df.head())




# Import necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Plot Validation Accuracy and Validation Loss for each parameter separately
unique_params = df["param"].unique()

for param in unique_params:
    # Filter the DataFrame for the current parameter
    subset = df[df["param"] == param]

    print(f"---------------------------------------------validation and accuracy for {param}-------------------------------------------- ")

    # Plot Validation Accuracy
    plt.figure(figsize=(10, 5))
    plt.scatter(subset["value"], subset["val_accuracy"], marker='o', s=80, label="Validation Accuracy")
    plt.plot(subset["value"], subset["val_accuracy"], linestyle='-', label="Validation Accuracy")
    plt.title(f"Validation Accuracy vs {param}")
    plt.xlabel(f"{param} Value")
    plt.ylabel("Validation Accuracy (%)")
    plt.grid()
    plt.legend()
    plt.show()

    # Plot Validation Loss
    plt.figure(figsize=(10, 5))
    plt.scatter(subset["value"], subset["val_loss"], marker='o', s=80, label="Validation Loss")
    plt.plot(subset["value"], subset["val_loss"], linestyle='-', label="Validation Loss")
    plt.title(f"Validation Loss vs {param}")
    plt.xlabel(f"{param} Value")
    plt.ylabel("Validation Loss")
    plt.grid()
    plt.legend()
    plt.show()

print(f"---------------------------------------------confusion matrix for {param}-------------------------------------------- ")

# Plot confusion matrix for each trained model
def plot_confusion_matrix(y_true, y_pred, param, value):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
    plt.title(f'Confusion Matrix for {param} = {value}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Generate confusion matrices for each parameter and value
for idx, row in df.iterrows():
    plot_confusion_matrix(row["y_true"], row["y_pred"], row["param"], row["value"])
