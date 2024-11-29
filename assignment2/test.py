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