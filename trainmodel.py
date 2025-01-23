import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoConfig
import torch.nn as nn
from sklearn.model_selection import train_test_split
import time, os

# Custom Dataset class for EEG data
class EEGDataset(Dataset):
    """
    PyTorch Dataset class for handling EEG data.

    Parameters:
        data: numpy array
            Array of EEG segments with shape (num_samples, sequence_length, num_features).
        labels: numpy array
            Array of labels corresponding to the EEG segments.
    """
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)  # Convert to PyTorch tensors
        self.labels = torch.tensor(labels, dtype=torch.long)  # Convert labels to long tensors for classification tasks

    def __len__(self):
        return len(self.data)  # Number of samples in the dataset

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]  # Return a single sample and its label

# Transformer-based model for EEG classification
class EEGTransformer(nn.Module):
    """
    A transformer-based neural network for EEG classification.
    
    Parameters:
        input_dim: int
            Number of input features (e.g., EEG channels).
        num_labels: int
            Number of classes for classification.
    """
    def __init__(self, input_dim, num_labels):
        super(EEGTransformer, self).__init__()
        self.transformer = AutoModel.from_pretrained("distilbert-base-uncased")  # Pretrained transformer model
        self.fc = nn.Linear(768, num_labels)  # Fully connected layer for classification
        self.input_projection = nn.Linear(input_dim, 768)  # Project input to match transformer dimension

    def forward(self, x):
        x = self.input_projection(x)  # Project input dimensions
        x = self.transformer(inputs_embeds=x)[0]  # Pass through transformer (return hidden states)
        x = x.mean(dim=1)  # Average across sequence dimension
        x = self.fc(x)  # Final classification layer
        return x

# Load preprocessed data
segments = np.load(os.path.join("training_data\preprocessed\eeg_segments.npy"))  # CHANGE THIS PATH
labels = np.load(os.path.join("training_data\preprocessed\eeg_labels.npy"))  # CHANGE THIS PATH

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(segments, labels, test_size=0.2, random_state=42)

# Print data shapes for verification
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Create PyTorch datasets and dataloaders
train_dataset = EEGDataset(X_train, y_train)
test_dataset = EEGDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # Adjust batch size as needed
test_loader = DataLoader(test_dataset, batch_size=16)

# Initialize the EEG Transformer model
input_dim = X_train.shape[2]  # Number of features (e.g., EEG channels)
num_labels = len(set(labels))  # Number of unique classes
model = EEGTransformer(input_dim=input_dim, num_labels=num_labels)

# Set up optimizer, loss function, and device
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)  # Learning rate can be adjusted
loss_fn = nn.CrossEntropyLoss()  # Loss function for classification
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
model.to(device)  # Send model to the device (CPU/GPU)

# Training loop
if __name__ == "__main__":
    """
    Main script for training the EEG Transformer model.

    - Trains the model for 5 epochs (can be adjusted).
    - Saves the trained model to the specified path.
    """
    for epoch in range(10):  # Adjust the number of epochs as needed
        model.train()
        total_loss = 0
        print(f"Starting Epoch {epoch + 1}...")
        
        # Loop through training batches
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)  # Send data to device
            
            optimizer.zero_grad()  # Reset gradients
            outputs = model(data)  # Forward pass
            loss = loss_fn(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            total_loss += loss.item()  # Accumulate loss
            print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item()}")

        # Log average loss for the epoch
        print(f"Epoch {epoch + 1} completed. Average Loss: {total_loss / len(train_loader)}")

    # Save the trained model
    model_save_path = os.path.join("eeg_transformer_model")  # CHANGE THIS PATH
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
