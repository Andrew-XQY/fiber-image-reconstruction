import torch
import torch.nn as nn
import torch.optim as optim
import os
from typing import Optional, Union


class SHLNeuralNetwork(nn.Module):
    """
    Simple 3-layer Fully Connected Neural Network:
    - Input layer -> Hidden layer (ReLU + Dropout)
    - Hidden layer -> Output layer (Sigmoid)
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout_rate: float = 0.2):
        super(SHLNeuralNetwork, self).__init__()
        
        # Store dimensions for saving/loading
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        
        # Three layers: input -> hidden -> output
        self.shl1 = nn.Linear(input_size, hidden_size)  # Input layer
        self.shl2 = nn.Linear(hidden_size, output_size)  # Output layer
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # Input -> Hidden (ReLU + Dropout)
        x = torch.relu(self.shl1(x))
        x = self.dropout(x)
        
        # Hidden -> Output (Sigmoid)
        x = torch.sigmoid(self.shl2(x))
        
        return x
    
    def save_model(self, save_dir: str, model_name: str = "model"):
        """Save the model state dict and metadata."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save state dict
        model_path = os.path.join(save_dir, f"{model_name}.pth")
        
        # Save both model state and metadata
        save_dict = {
            'state_dict': self.state_dict(),
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'dropout_rate': self.dropout_rate,
        }
        
        torch.save(save_dict, model_path)
        print(f"Model saved to {model_path}")
    
    @classmethod
    def load_model(cls, filepath: str, device: Optional[Union[str, torch.device]] = None):
        """Load a saved model."""
        # Load the saved data
        saved_data = torch.load(filepath, map_location=device)
        
        # Create model with saved parameters
        model = cls(
            input_size=saved_data['input_size'],
            hidden_size=saved_data['hidden_size'],
            output_size=saved_data['output_size'],
            dropout_rate=saved_data['dropout_rate'],
        )
        
        # Load state dict
        model.load_state_dict(saved_data['state_dict'])
        
        # Move to device if specified
        if device is not None:
            model = model.to(device)
        
        return model