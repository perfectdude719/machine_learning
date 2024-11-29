import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


hidden_layers=[1,3,5,7]
neurons=[10,20,30,40]
batch_size=[15,30,60,120]
learning_rates=[0.0000001,0.000001,0.00001,0.0001]



for size in batch_size:
    for rate in learning_rates:
         
         class Neural_Network_1_hidden_layer(nn.Module):
            def __init__(self, input_size=28*28, hidden_size=neuron, num_classes=10):
                super(Neural_Network_1_hidden_layer, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size) 
                self.fc2 = nn.Linear(hidden_size, num_classes)

            def forward(self, x):
                x = x.view(-1, 28*28)  # Flatten the input
                x = torch.relu(self.fc1(x))  # Apply ReLU activation after first hidden layer
                x = self.fc2(x)  # Output layer
                return x
            
    class Neural_Network_3_hidden_layer(nn.Module):
        def __init__(self, input_size=28*28, hidden_size=neuron, num_classes=10):
            super(Neural_Network_3_hidden_layer, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size) 
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, num_classes)


        def forward(self, x):      #forward probagation
            x = x.view(-1, 28*28)  # Flatten the input
            x = torch.relu(self.fc1(x))  # Apply ReLU activation after first hidden layer
            x = self.fc2(x)
            x=  sel.fc3(x)
            return x





   