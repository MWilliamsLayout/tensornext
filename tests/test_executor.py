"""
Tests for Executor class.
"""

import pytest
import torch
import torch.nn as nn
import sys
import os
import threading

# Add parent directory to path to import server modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from server.executor import Executor
from server.models.simple_model import SimpleMLP


class TestExecutor:
    """Test cases for Executor class."""
    
    def test_executor_initialization(self):
        """Test executor initialization."""
        model = SimpleMLP()
        executor = Executor(model, device="cpu")
        
        assert executor.model is not None
        assert executor.get_device() == "cpu"
    
    def test_execute_on_cpu(self):
        """Test execution on CPU."""
        model = SimpleMLP()
        executor = Executor(model, device="cpu")
        
        input_tensor = torch.randn(2, 128)
        output = executor.execute(input_tensor)
        
        assert output.shape == (2, 64)
        assert output.device.type == "cpu"
    
    def test_execute_with_explicit_device(self):
        """Test execution with explicit device parameter."""
        model = SimpleMLP()
        executor = Executor(model, device="cpu")
        
        input_tensor = torch.randn(2, 128)
        # Execute on CPU even though executor is initialized with CPU
        output = executor.execute(input_tensor, device="cpu")
        
        assert output.shape == (2, 64)
        assert output.device.type == "cpu"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_execute_on_gpu(self):
        """Test execution on GPU if available."""
        model = SimpleMLP()
        executor = Executor(model, device="cuda:0")
        
        input_tensor = torch.randn(2, 128).to("cuda:0")
        output = executor.execute(input_tensor)
        
        assert output.shape == (2, 64)
        assert output.device.type == "cuda"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_execute_temporary_device_change(self):
        """Test that executor can execute on different device temporarily."""
        model = SimpleMLP()
        executor = Executor(model, device="cpu")
        
        # Execute on GPU temporarily
        input_tensor = torch.randn(2, 128).to("cuda:0")
        output = executor.execute(input_tensor, device="cuda:0")
        
        assert output.shape == (2, 64)
        assert output.device.type == "cuda"
        # Executor should still be on CPU after temporary execution
        assert executor.get_device() == "cpu"
    
    def test_set_device(self):
        """Test changing executor device."""
        model = SimpleMLP()
        executor = Executor(model, device="cpu")
        
        assert executor.get_device() == "cpu"
        
        # Set device should work (even if same device)
        executor.set_device("cpu")
        assert executor.get_device() == "cpu"
    
    def test_execute_batch_variations(self):
        """Test execution with different batch sizes."""
        model = SimpleMLP()
        executor = Executor(model, device="cpu")
        
        # Single sample
        output1 = executor.execute(torch.randn(1, 128))
        assert output1.shape == (1, 64)
        
        # Multiple samples
        output2 = executor.execute(torch.randn(5, 128))
        assert output2.shape == (5, 64)
        
        # Large batch
        output3 = executor.execute(torch.randn(32, 128))
        assert output3.shape == (32, 64)
    
    def test_thread_safety(self):
        """Test that executor is thread-safe."""
        model = SimpleMLP()
        executor = Executor(model, device="cpu")
        
        results = []
        errors = []
        
        def run_inference(thread_id):
            try:
                input_tensor = torch.randn(2, 128)
                output = executor.execute(input_tensor)
                results.append((thread_id, output.shape))
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Create multiple threads
        threads = []
        num_threads = 5
        for i in range(num_threads):
            thread = threading.Thread(target=run_inference, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check that all threads completed successfully
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == num_threads
        for thread_id, shape in results:
            assert shape == (2, 64)
