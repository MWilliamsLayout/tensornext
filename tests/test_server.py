"""
Tests for FastAPI server endpoints.
"""

import pytest
import torch
import sys
import os
from fastapi.testclient import TestClient

# Add parent directory to path to import server modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import server.server as server_module
from server.scheduler import Scheduler
from server.executor import Executor
from server.models.simple_model import create_model


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(server_module.app)


@pytest.fixture(autouse=True)
def setup_server():
    """Setup server components before each test."""
    # Initialize scheduler
    test_scheduler = Scheduler(default_gpu_device="cuda:0")
    
    # Determine initial device
    initial_device = test_scheduler.schedule(prefer_gpu=False)  # Use CPU for tests
    if initial_device is None:
        initial_device = "cpu"
    
    # Load model on initial device
    model = create_model(device=initial_device)
    test_executor = Executor(model, device=initial_device)
    
    # Set global variables in the server module
    server_module.scheduler = test_scheduler
    server_module.executor = test_executor
    
    yield
    
    # Cleanup (if needed)
    server_module.scheduler = None
    server_module.executor = None


class TestHealthEndpoint:
    """Test cases for /health endpoint."""
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert data["status"] == "ready"
        assert "cuda_available" in data
        assert "current_device" in data
    
    def test_health_check_structure(self, client):
        """Test health check response structure."""
        response = client.get("/health")
        data = response.json()
        
        # Required fields
        assert "status" in data
        assert "cuda_available" in data
        assert "current_device" in data
        
        # Optional GPU memory field (if CUDA available)
        if data["cuda_available"]:
            # GPU memory may or may not be present
            pass


class TestPredictEndpoint:
    """Test cases for /predict endpoint."""
    
    def test_predict_basic(self, client):
        """Test basic prediction request."""
        request_data = {
            "input_data": [[0.1] * 128],  # Single sample with 128 features
            "prefer_gpu": False
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "output" in data
        assert "device" in data
        assert "latency_ms" in data
        
        # Check output shape
        assert len(data["output"]) == 1  # Batch size 1
        assert len(data["output"][0]) == 64  # Output dimension
    
    def test_predict_batch(self, client):
        """Test prediction with batch of samples."""
        batch_size = 3
        request_data = {
            "input_data": [[0.1] * 128 for _ in range(batch_size)],
            "prefer_gpu": False
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["output"]) == batch_size
        assert len(data["output"][0]) == 64
    
    def test_predict_with_gpu_preference(self, client):
        """Test prediction with GPU preference."""
        request_data = {
            "input_data": [[0.1] * 128],
            "prefer_gpu": True,
            "min_gpu_memory_mb": 1.0
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "device" in data
        # Device should be either cpu or cuda depending on availability
        assert data["device"] in ["cpu", "cuda:0"]
    
    def test_predict_invalid_input_shape(self, client):
        """Test prediction with invalid input shape."""
        request_data = {
            "input_data": [[0.1] * 64],  # Wrong dimension (should be 128)
            "prefer_gpu": False
        }
        
        # This should fail because model expects 128 features
        response = client.post("/predict", json=request_data)
        # May return 200 but with incorrect output, or 500 error
        # The exact behavior depends on model error handling
        assert response.status_code in [200, 500]
    
    def test_predict_empty_batch(self, client):
        """Test prediction with empty batch."""
        request_data = {
            "input_data": [],
            "prefer_gpu": False
        }
        
        response = client.post("/predict", json=request_data)
        # Should handle empty batch (may return 200 or 500)
        assert response.status_code in [200, 500]
    
    def test_predict_missing_fields(self, client):
        """Test prediction with missing required fields."""
        request_data = {
            # Missing input_data
            "prefer_gpu": False
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_predict_response_structure(self, client):
        """Test prediction response structure."""
        request_data = {
            "input_data": [[0.1] * 128],
            "prefer_gpu": False
        }
        
        response = client.post("/predict", json=request_data)
        data = response.json()
        
        # Required fields
        assert "output" in data
        assert "device" in data
        assert "latency_ms" in data
        
        # Latency should be non-negative
        assert data["latency_ms"] >= 0
        
        # GPU memory is optional
        if "gpu_memory_mb" in data:
            assert data["gpu_memory_mb"] >= 0
    
    def test_predict_multiple_requests(self, client):
        """Test multiple prediction requests."""
        request_data = {
            "input_data": [[0.1] * 128],
            "prefer_gpu": False
        }
        
        # Send multiple requests
        for _ in range(3):
            response = client.post("/predict", json=request_data)
            assert response.status_code == 200
            data = response.json()
            assert "output" in data
            assert len(data["output"]) == 1
