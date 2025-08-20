"""
Clean Transmission Matrix Module - Minimal Implementation

A simple linear transmission matrix for image-to-image mapping.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union
import os


class TransmissionMatrix(nn.Module):
    """
    Simple transmission matrix for image-to-image mapping.
    Just a linear layer that maps input images to output images.
    """
    
    def __init__(
        self,
        input_height: int,
        input_width: int,
        output_height: int,
        output_width: int,
        initialization: str = "random",
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        add_bias: bool = False,
    ):
        super().__init__()
        
        # Store dimensions
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        
        # Calculate flattened dimensions
        self.input_dim = input_height * input_width
        self.output_dim = output_height * output_width
        
        # Create the transmission matrix as a linear layer
        self.transmission_matrix = nn.Linear(
            self.input_dim, 
            self.output_dim, 
            bias=add_bias,
            dtype=dtype,
            device=device
        )
        
        # Initialize the matrix
        self._initialize_matrix(initialization)
    
    def _initialize_matrix(self, method: str):
        """Initialize the transmission matrix weights with reproducible methods."""
        if method == "random":
            # Use Xavier uniform for consistent, seed-controlled initialization
            nn.init.xavier_uniform_(self.transmission_matrix.weight)
        elif method == "xavier":
            nn.init.xavier_uniform_(self.transmission_matrix.weight)
        elif method == "xavier_normal":
            nn.init.xavier_normal_(self.transmission_matrix.weight)
        elif method == "kaiming":
            nn.init.kaiming_uniform_(self.transmission_matrix.weight)
        elif method == "identity":
            # Only works for square matrices
            if self.input_dim == self.output_dim:
                nn.init.eye_(self.transmission_matrix.weight)
            else:
                nn.init.xavier_uniform_(self.transmission_matrix.weight)
        else:
            raise ValueError(f"Unknown initialization method: {method}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: apply transmission matrix to input.
        
        Args:
            x: Input tensor of shape (batch_size, height, width) or (batch_size, features)
        
        Returns:
            Output tensor of shape (batch_size, output_height, output_width)
        """
        batch_size = x.size(0)
        
        # Flatten if necessary
        if x.dim() == 3:  # (batch_size, height, width)
            x_flat = x.view(batch_size, -1)
        elif x.dim() == 2:  # (batch_size, features)
            x_flat = x
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D")
        
        # Apply transmission matrix
        output_flat = self.transmission_matrix(x_flat)
        
        # Reshape to image format
        return output_flat.view(batch_size, self.output_height, self.output_width)
    
    def save_model(self, save_dir: str, model_name: str = "model"):
        """Save the model state dict and metadata."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save state dict
        model_path = os.path.join(save_dir, f"{model_name}.pth")
        
        # Save both model state and metadata
        save_dict = {
            'state_dict': self.state_dict(),
            'input_height': self.input_height,
            'input_width': self.input_width,
            'output_height': self.output_height,
            'output_width': self.output_width,
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
            input_height=saved_data['input_height'],
            input_width=saved_data['input_width'],
            output_height=saved_data['output_height'],
            output_width=saved_data['output_width'],
        )
        
        # Load state dict
        model.load_state_dict(saved_data['state_dict'])
        
        # Move to device if specified
        if device is not None:
            model = model.to(device)
        
        return model
