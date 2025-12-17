"""
Tests for SimpleMLP model and model creation utilities.
"""

import pytest
import torch
import sys
import os

# Add parent directory to path to import server modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from server.models.simple_model import SimpleMLP, create_model


class TestSimpleMLP:
    """Test cases for SimpleMLP class."""
    
    def test_model_initialization_default(self):
        """Test model initialization with default parameters."""
        model = SimpleMLP()
        assert model is not None
        assert model.fc1.in_features == 128
        assert model.fc1.out_features == 256
        assert model.fc2.in_features == 256
        assert model.fc2.out_features == 256
        assert model.fc3.in_features == 256
        assert model.fc3.out_features == 64
    
    def test_model_initialization_custom(self):
        """Test model initialization with custom parameters."""
        model = SimpleMLP(input_dim=64, hidden_dim=128, output_dim=32)
        assert model.fc1.in_features == 64
        assert model.fc1.out_features == 128
        assert model.fc2.in_features == 128
        assert model.fc2.out_features == 128
        assert model.fc3.in_features == 128
        assert model.fc3.out_features == 32
    
    def test_forward_pass_default_shape(self):
        """Test forward pass with default model dimensions."""
        model = SimpleMLP()
        model.eval()
        
        batch_size = 4
        input_tensor = torch.randn(batch_size, 128)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        assert output.shape == (batch_size, 64)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_forward_pass_custom_shape(self):
        """Test forward pass with custom model dimensions."""
        model = SimpleMLP(input_dim=32, hidden_dim=64, output_dim=16)
        model.eval()
        
        batch_size = 2
        input_tensor = torch.randn(batch_size, 32)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        assert output.shape == (batch_size, 16)
    
    def test_forward_pass_single_sample(self):
        """Test forward pass with single sample (batch_size=1)."""
        model = SimpleMLP()
        model.eval()
        
        input_tensor = torch.randn(1, 128)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        assert output.shape == (1, 64)
    
    def test_model_on_cpu(self):
        """Test model can be moved to CPU."""
        model = SimpleMLP()
        model = model.to("cpu")
        model.eval()
        
        input_tensor = torch.randn(2, 128)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        assert output.device.type == "cpu"
        assert output.shape == (2, 64)


class TestCreateModel:
    """Test cases for create_model function."""
    
    def test_create_model_cpu(self):
        """Test model creation on CPU."""
        model = create_model(device="cpu")
        
        assert model is not None
        assert not model.training  # Should be in eval mode
        
        # Test inference
        input_tensor = torch.randn(1, 128)
        with torch.no_grad():
            output = model(input_tensor)
        
        assert output.shape == (1, 64)
        assert output.device.type == "cpu"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_create_model_gpu(self):
        """Test model creation on GPU if available."""
        model = create_model(device="cuda:0")
        
        assert model is not None
        assert not model.training  # Should be in eval mode
        
        # Test inference
        input_tensor = torch.randn(1, 128).to("cuda:0")
        with torch.no_grad():
            output = model(input_tensor)
        
        assert output.shape == (1, 64)
        assert output.device.type == "cuda"
    
    def test_create_model_default(self):
        """Test model creation with default device (CPU)."""
        model = create_model()
        
        assert model is not None
        assert not model.training
        
        input_tensor = torch.randn(1, 128)
        with torch.no_grad():
            output = model(input_tensor)
        
        assert output.device.type == "cpu"
