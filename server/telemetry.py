"""
Telemetry collection for inference execution.

This module provides utilities to collect execution metadata including:
- End-to-end latency
- Device used (CPU/GPU)
- GPU memory usage (if applicable)
"""

import time
from typing import Dict, Optional
import torch


class TelemetryCollector:
    """
    Collects execution telemetry for inference requests.
    
    This class tracks timing, device usage, and GPU memory statistics.
    """
    
    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.device: Optional[str] = None
        
    def start(self):
        """Mark the start of execution."""
        self.start_time = time.perf_counter()
        
    def stop(self):
        """Mark the end of execution."""
        self.end_time = time.perf_counter()
        
    def set_device(self, device: str):
        """Set the device used for execution."""
        self.device = device
        
    def get_latency_ms(self) -> float:
        """
        Get end-to-end latency in milliseconds.
        
        Returns:
            Latency in milliseconds, or 0.0 if timing not completed
        """
        if self.start_time is None or self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000.0
    
    def get_gpu_memory_mb(self, device: str) -> Optional[float]:
        """
        Get GPU memory usage in MB for the specified device.
        
        Args:
            device: CUDA device string (e.g., "cuda:0")
        
        Returns:
            Memory usage in MB, or None if not a CUDA device or CUDA unavailable
        """
        if not device.startswith("cuda"):
            return None
        
        if not torch.cuda.is_available():
            return None
        
        try:
            device_idx = int(device.split(":")[1])
            allocated = torch.cuda.memory_allocated(device_idx) / (1024 ** 2)  # Convert to MB
            return allocated
        except (IndexError, ValueError, RuntimeError):
            return None
    
    def get_free_gpu_memory_mb(self, device: str) -> Optional[float]:
        """
        Get free GPU memory in MB for the specified device.
        
        Args:
            device: CUDA device string (e.g., "cuda:0")
        
        Returns:
            Free memory in MB, or None if not a CUDA device or CUDA unavailable
        """
        if not device.startswith("cuda"):
            return None
        
        if not torch.cuda.is_available():
            return None
        
        try:
            device_idx = int(device.split(":")[1])
            total = torch.cuda.get_device_properties(device_idx).total_memory / (1024 ** 2)
            allocated = torch.cuda.memory_allocated(device_idx) / (1024 ** 2)
            free = total - allocated
            return free
        except (IndexError, ValueError, RuntimeError):
            return None
    
    def to_dict(self) -> Dict:
        """
        Convert telemetry data to a dictionary for API response.
        
        Returns:
            Dictionary containing execution metadata
        """
        result = {
            "device": self.device or "unknown",
            "latency_ms": round(self.get_latency_ms(), 2)
        }
        
        if self.device and self.device.startswith("cuda"):
            gpu_memory = self.get_gpu_memory_mb(self.device)
            if gpu_memory is not None:
                result["gpu_memory_mb"] = round(gpu_memory, 2)
        
        return result

