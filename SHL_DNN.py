import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class FCNeuralNetwork(nn.Module):
    """
    A modular Fully Connected Neural Network with one input layer, 
    one hidden layer, and one output layer.
    
    Features:
    - ReLU activation
    - Dropout for regularization
    - Dynamic layer size configuration
    """
    
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        output_size: int, 
        dropout_rate: float = 0.2
    ):
        """
        Initialize the Fully Connected Neural Network.
        
        Args:
            input_size (int): Number of input features
            hidden_size (int): Number of neurons in the hidden layer
            output_size (int): Number of output neurons
            dropout_rate (float): Dropout probability (default: 0.2)
        """
        super(FCNeuralNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        
        # Define layers
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, output_size)
        
        # Dropout layer (applied after ReLU activation)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/Glorot initialization."""
        for layer in [self.input_layer, self.hidden_layer]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size)
        """
        # Input to hidden layer with ReLU activation
        hidden = F.relu(self.input_layer(x))
        
        # Apply dropout (only during training)
        hidden = self.dropout(hidden)
        
        # Hidden to output layer
        output = self.hidden_layer(hidden)
        
        return output
    
    def get_model_info(self) -> dict:
        """
        Get information about the model architecture.
        
        Returns:
            dict: Model information including layer sizes and parameters
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'dropout_rate': self.dropout_rate,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }
    
    def save_model(self, save_dir: str):
        """
        Save the model state dict to a folder with standard filename.
        
        Args:
            save_dir (str): Directory path where to save the model
        """
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Use standard model filename
        model_path = os.path.join(save_dir, "model.pth")
        
        # Save model state dict (standard PyTorch way)
        torch.save(self.state_dict(), model_path)
        print(f"Model saved to: {model_path}")


class FCNNTrainer:
    """
    A trainer class for the Fully Connected Neural Network.
    Provides training, validation, and evaluation functionality.
    """
    
    def __init__(
        self, 
        model: FCNeuralNetwork, 
        learning_rate: float = 0.001,
        device: Optional[str] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model (FCNeuralNetwork): The neural network model
            learning_rate (float): Learning rate for optimization
            device (str, optional): Device to run training on ('cuda' or 'cpu')
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()  # Default to MSE, can be changed
        
        self.train_losses = []
        self.val_losses = []
    
    def set_criterion(self, criterion):
        """Set a custom loss criterion."""
        self.criterion = criterion
    
    def train_epoch(
        self, 
        train_loader: torch.utils.data.DataLoader
    ) -> float:
        """
        Train the model for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(
        self, 
        val_loader: torch.utils.data.DataLoader
    ) -> float:
        """
        Validate the model.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            float: Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def train(
        self, 
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        epochs: int = 100,
        verbose: bool = True
    ) -> Tuple[list, list]:
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
            epochs: Number of training epochs
            verbose: Whether to print training progress
            
        Returns:
            Tuple[list, list]: Training losses and validation losses
        """
        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            val_loss = None
            if val_loader:
                val_loss = self.validate(val_loader)
            
            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                if val_loss:
                    print(f'Epoch [{epoch+1}/{epochs}], '
                          f'Train Loss: {train_loss:.4f}, '
                          f'Val Loss: {val_loss:.4f}')
                else:
                    print(f'Epoch [{epoch+1}/{epochs}], '
                          f'Train Loss: {train_loss:.4f}')
        
        return self.train_losses, self.val_losses
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions with the trained model.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Model predictions
        """
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            return self.model(x)


def create_fc_network(
    input_size: int,
    hidden_size: int,
    output_size: int,
    dropout_rate: float = 0.2
) -> FCNeuralNetwork:
    """
    Factory function to create a Fully Connected Neural Network.
    
    Args:
        input_size (int): Number of input features
        hidden_size (int): Number of neurons in the hidden layer
        output_size (int): Number of output neurons
        dropout_rate (float): Dropout probability
        
    Returns:
        FCNeuralNetwork: Configured neural network
    """
    return FCNeuralNetwork(input_size, hidden_size, output_size, dropout_rate)


def create_sample_data(
    num_samples: int = 1000,
    input_size: int = 10,
    output_size: int = 1,
    noise_level: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create sample data for testing the neural network.
    
    Args:
        num_samples: Number of samples to generate
        input_size: Number of input features
        output_size: Number of output features
        noise_level: Level of noise to add to the data
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Input and target tensors
    """
    # Generate random input data
    X = torch.randn(num_samples, input_size)
    
    # Create a simple linear relationship with some non-linearity
    weights = torch.randn(input_size, output_size)
    y = torch.mm(X, weights) + torch.sin(X.sum(dim=1, keepdim=True))
    
    # Add noise
    y += noise_level * torch.randn_like(y)
    
    return X, y