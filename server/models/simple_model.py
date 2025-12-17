"""
Simple PyTorch model for inference demonstration.

This is a minimal MLP (Multi-Layer Perceptron) that can run on both CPU and GPU.
The model is intentionally small for PoC purposes.
"""

import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """
    A simple 3-layer MLP for demonstration purposes.
    
    Input: 128 features
    Hidden: 256 features
    Output: 64 features
    """
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, output_dim: int = 64):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def create_model(device: str = "cpu") -> nn.Module:
    """
    Create and initialize the model on the specified device.
    
    Args:
        device: Target device ("cpu" or "cuda:0", "cuda:1", etc.)
    
    Returns:
        Model instance on the specified device
    """
    model = SimpleMLP()
    model = model.to(device)
    model.eval()  # Set to evaluation mode for inference
    return model

