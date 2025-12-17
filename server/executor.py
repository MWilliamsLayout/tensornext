"""
Executor for running inference on CPU or GPU.

The executor loads the model and executes inference requests on the
scheduled device. It handles device-specific operations and ensures
proper tensor placement.
"""

from typing import Dict, Any
import threading
import torch
import torch.nn as nn


class Executor:
    """
    Executes inference requests on the specified device.
    
    The executor manages model loading and inference execution,
    handling device placement and tensor operations.
    """
    
    def __init__(self, model: nn.Module, device: str = "cpu"):
        """
        Initialize the executor with a model.
        
        Args:
            model: PyTorch model instance
            device: Target device for execution ("cpu" or "cuda:0", etc.)
        """
        self.model = model
        self.device = device
        # Ensure model is on the correct device
        self.model = self.model.to(device)
        self.model.eval()
        # Lock to ensure atomic device changes and execution
        self._lock = threading.Lock()
        
    def execute(self, input_data: torch.Tensor, device: str = None) -> torch.Tensor:
        """
        Execute inference on the input data.
        
        Args:
            input_data: Input tensor (will be moved to the specified device)
            device: Optional device to execute on. If None, uses executor's current device.
                    When provided, executes on this device without modifying executor state.
        
        Returns:
            Output tensor from the model
        """
        # Use provided device or fall back to executor's device
        execution_device = device if device is not None else self.device
        
        # Move input to the execution device (this is safe, doesn't modify executor state)
        input_tensor = input_data.to(execution_device)
        
        # Use lock to ensure atomic device changes and execution
        with self._lock:
            # Handle device placement: preserve executor state if using different device
            model_was_moved = False
            original_device = None
            
            if device is not None and device != self.device:
                # Store original device and move model temporarily
                original_device = self.device
                self.model = self.model.to(execution_device)
                model_was_moved = True
            
            try:
                # Run inference (no gradient computation needed)
                with torch.no_grad():
                    output = self.model(input_tensor)
            finally:
                # Restore model to original device if it was moved
                if model_was_moved and original_device is not None:
                    self.model = self.model.to(original_device)
        
        return output
    
    def set_device(self, device: str):
        """
        Change the execution device.
        
        Args:
            device: New target device ("cpu" or "cuda:0", etc.)
        """
        with self._lock:
            self.device = device
            self.model = self.model.to(device)
            self.model.eval()
    
    def get_device(self) -> str:
        """
        Get the current execution device.
        
        Returns:
            Current device string
        """
        return self.device

