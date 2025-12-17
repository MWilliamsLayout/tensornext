"""
Tests for Scheduler class.
"""

import pytest
import torch
import sys
import os

# Add parent directory to path to import server modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from server.scheduler import Scheduler


class TestScheduler:
    """Test cases for Scheduler class."""
    
    def test_scheduler_initialization_default(self):
        """Test scheduler initialization with default GPU device."""
        scheduler = Scheduler()
        assert scheduler.default_gpu_device == "cuda:0"
        assert isinstance(scheduler.is_gpu_available(), bool)
    
    def test_scheduler_initialization_custom(self):
        """Test scheduler initialization with custom GPU device."""
        scheduler = Scheduler(default_gpu_device="cuda:1")
        assert scheduler.default_gpu_device == "cuda:1"
    
    def test_is_gpu_available(self):
        """Test GPU availability check."""
        scheduler = Scheduler()
        # Should return boolean matching torch.cuda.is_available()
        assert scheduler.is_gpu_available() == torch.cuda.is_available()
    
    def test_schedule_prefer_cpu(self):
        """Test scheduling when CPU is preferred."""
        scheduler = Scheduler()
        device = scheduler.schedule(prefer_gpu=False)
        assert device == "cpu"
    
    def test_schedule_no_gpu_available(self):
        """Test scheduling falls back to CPU when GPU not available."""
        scheduler = Scheduler()
        # If CUDA is not available, should always return CPU
        if not scheduler.is_gpu_available():
            device = scheduler.schedule(prefer_gpu=True)
            assert device == "cpu"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_schedule_gpu_available(self):
        """Test scheduling to GPU when available."""
        scheduler = Scheduler()
        if scheduler.is_gpu_available():
            device = scheduler.schedule(prefer_gpu=True, min_gpu_memory_mb=1.0)
            assert device.startswith("cuda")
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_schedule_insufficient_gpu_memory(self):
        """Test scheduling falls back to CPU when GPU memory is insufficient."""
        scheduler = Scheduler()
        if scheduler.is_gpu_available():
            # Request extremely high memory requirement
            device = scheduler.schedule(prefer_gpu=True, min_gpu_memory_mb=1e9)
            assert device == "cpu"
    
    def test_get_gpu_memory_free_mb_no_cuda(self):
        """Test GPU memory check when CUDA not available."""
        scheduler = Scheduler()
        if not scheduler.is_gpu_available():
            memory = scheduler.get_gpu_memory_free_mb("cuda:0")
            assert memory is None
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_gpu_memory_free_mb_with_cuda(self):
        """Test GPU memory check when CUDA is available."""
        scheduler = Scheduler()
        if scheduler.is_gpu_available():
            memory = scheduler.get_gpu_memory_free_mb("cuda:0")
            assert memory is not None
            assert memory >= 0
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_gpu_memory_free_mb_invalid_device(self):
        """Test GPU memory check with invalid device string."""
        scheduler = Scheduler()
        if scheduler.is_gpu_available():
            # Invalid device format
            memory = scheduler.get_gpu_memory_free_mb("invalid")
            assert memory is None
    
    def test_schedule_with_low_memory_threshold(self):
        """Test scheduling with very low memory threshold."""
        scheduler = Scheduler()
        device = scheduler.schedule(prefer_gpu=True, min_gpu_memory_mb=0.1)
        # Should either be CPU or GPU depending on availability
        assert device in ["cpu", "cuda:0"]
    
    def test_schedule_multiple_calls(self):
        """Test that scheduler can handle multiple scheduling calls."""
        scheduler = Scheduler()
        
        # Make multiple scheduling decisions
        devices = []
        for _ in range(5):
            device = scheduler.schedule(prefer_gpu=False)
            devices.append(device)
        
        # All should be CPU when prefer_gpu=False
        assert all(d == "cpu" for d in devices)
