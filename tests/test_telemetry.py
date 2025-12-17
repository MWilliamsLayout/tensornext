"""
Tests for TelemetryCollector class.
"""

import pytest
import time
import torch
import sys
import os

# Add parent directory to path to import server modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from server.telemetry import TelemetryCollector


class TestTelemetryCollector:
    """Test cases for TelemetryCollector class."""
    
    def test_initialization(self):
        """Test telemetry collector initialization."""
        telemetry = TelemetryCollector()
        assert telemetry.start_time is None
        assert telemetry.end_time is None
        assert telemetry.device is None
    
    def test_start_stop(self):
        """Test start and stop timing."""
        telemetry = TelemetryCollector()
        
        telemetry.start()
        assert telemetry.start_time is not None
        
        time.sleep(0.01)  # Sleep for 10ms
        
        telemetry.stop()
        assert telemetry.end_time is not None
        assert telemetry.end_time > telemetry.start_time
    
    def test_get_latency_ms(self):
        """Test latency calculation."""
        telemetry = TelemetryCollector()
        
        # Before start/stop, should return 0
        assert telemetry.get_latency_ms() == 0.0
        
        telemetry.start()
        time.sleep(0.01)  # Sleep for 10ms
        telemetry.stop()
        
        latency = telemetry.get_latency_ms()
        assert latency > 0
        assert latency >= 10  # Should be at least 10ms
    
    def test_set_device(self):
        """Test setting device."""
        telemetry = TelemetryCollector()
        
        telemetry.set_device("cpu")
        assert telemetry.device == "cpu"
        
        telemetry.set_device("cuda:0")
        assert telemetry.device == "cuda:0"
    
    def test_get_gpu_memory_mb_cpu(self):
        """Test GPU memory check for CPU device."""
        telemetry = TelemetryCollector()
        memory = telemetry.get_gpu_memory_mb("cpu")
        assert memory is None
    
    def test_get_gpu_memory_mb_no_cuda(self):
        """Test GPU memory check when CUDA not available."""
        telemetry = TelemetryCollector()
        if not torch.cuda.is_available():
            memory = telemetry.get_gpu_memory_mb("cuda:0")
            assert memory is None
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_gpu_memory_mb_with_cuda(self):
        """Test GPU memory check when CUDA is available."""
        telemetry = TelemetryCollector()
        if torch.cuda.is_available():
            memory = telemetry.get_gpu_memory_mb("cuda:0")
            # Should return a value (could be 0 or more)
            assert memory is not None
            assert memory >= 0
    
    def test_get_free_gpu_memory_mb_cpu(self):
        """Test free GPU memory check for CPU device."""
        telemetry = TelemetryCollector()
        memory = telemetry.get_free_gpu_memory_mb("cpu")
        assert memory is None
    
    def test_get_free_gpu_memory_mb_no_cuda(self):
        """Test free GPU memory check when CUDA not available."""
        telemetry = TelemetryCollector()
        if not torch.cuda.is_available():
            memory = telemetry.get_free_gpu_memory_mb("cuda:0")
            assert memory is None
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_free_gpu_memory_mb_with_cuda(self):
        """Test free GPU memory check when CUDA is available."""
        telemetry = TelemetryCollector()
        if torch.cuda.is_available():
            memory = telemetry.get_free_gpu_memory_mb("cuda:0")
            assert memory is not None
            assert memory >= 0
    
    def test_to_dict_no_timing(self):
        """Test conversion to dict without timing."""
        telemetry = TelemetryCollector()
        telemetry.set_device("cpu")
        
        result = telemetry.to_dict()
        assert result["device"] == "cpu"
        assert result["latency_ms"] == 0.0
        assert "gpu_memory_mb" not in result
    
    def test_to_dict_with_timing(self):
        """Test conversion to dict with timing."""
        telemetry = TelemetryCollector()
        telemetry.set_device("cpu")
        telemetry.start()
        time.sleep(0.01)
        telemetry.stop()
        
        result = telemetry.to_dict()
        assert result["device"] == "cpu"
        assert result["latency_ms"] > 0
        assert "gpu_memory_mb" not in result
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_to_dict_with_gpu(self):
        """Test conversion to dict with GPU device."""
        telemetry = TelemetryCollector()
        telemetry.set_device("cuda:0")
        telemetry.start()
        time.sleep(0.001)  # Short sleep
        telemetry.stop()
        
        result = telemetry.to_dict()
        assert result["device"] == "cuda:0"
        assert result["latency_ms"] >= 0
        # GPU memory may or may not be present depending on allocation
        if "gpu_memory_mb" in result:
            assert result["gpu_memory_mb"] >= 0
    
    def test_multiple_start_stop(self):
        """Test multiple start/stop cycles."""
        telemetry = TelemetryCollector()
        
        # First cycle
        telemetry.start()
        time.sleep(0.005)
        telemetry.stop()
        latency1 = telemetry.get_latency_ms()
        
        # Second cycle
        telemetry.start()
        time.sleep(0.005)
        telemetry.stop()
        latency2 = telemetry.get_latency_ms()
        
        # Both should have positive latency
        assert latency1 > 0
        assert latency2 > 0
