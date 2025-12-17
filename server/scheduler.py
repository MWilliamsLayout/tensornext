"""
Scheduler for deciding execution target (CPU vs GPU).

The scheduler inspects request constraints and GPU availability to make
routing decisions. This is a simple implementation suitable for PoC.
"""

from typing import Optional
import torch


class Scheduler:
    """
    Makes scheduling decisions for inference requests.
    
    The scheduler checks:
    - Request preferences (prefer_gpu)
    - GPU availability
    - GPU memory availability
    """
    
    def __init__(self, default_gpu_device: str = "cuda:0"):
        """
        Initialize the scheduler.
        
        Args:
            default_gpu_device: Default CUDA device to use (e.g., "cuda:0")
        """
        self.default_gpu_device = default_gpu_device
        self.cuda_available = torch.cuda.is_available()
        
    def is_gpu_available(self) -> bool:
        """
        Check if CUDA is available.
        
        Returns:
            True if CUDA is available, False otherwise
        """
        return self.cuda_available
    
    def get_gpu_memory_free_mb(self, device: str) -> Optional[float]:
        """
        Get free GPU memory in MB for the specified device.
        
        Args:
            device: CUDA device string (e.g., "cuda:0")
        
        Returns:
            Free memory in MB, or None if not available
        """
        if not self.cuda_available:
            return None
        
        try:
            device_idx = int(device.split(":")[1])
            total = torch.cuda.get_device_properties(device_idx).total_memory / (1024 ** 2)
            allocated = torch.cuda.memory_allocated(device_idx) / (1024 ** 2)
            free = total - allocated
            return free
        except (IndexError, ValueError, RuntimeError):
            return None
    
    def schedule(self, prefer_gpu: bool = True, min_gpu_memory_mb: float = 100.0) -> str:
        """
        Schedule an inference request to either CPU or GPU.
        
        Args:
            prefer_gpu: Whether the client prefers GPU execution
            min_gpu_memory_mb: Minimum free GPU memory required (in MB) to use GPU
        
        Returns:
            Device string: "cpu" or "cuda:0" (or other CUDA device)
        """
        # If GPU is not preferred, route to CPU
        if not prefer_gpu:
            return "cpu"
        
        # If CUDA is not available, fall back to CPU
        if not self.cuda_available:
            return "cpu"
        
        # Check GPU memory availability
        free_memory = self.get_gpu_memory_free_mb(self.default_gpu_device)
        if free_memory is None or free_memory < min_gpu_memory_mb:
            # Not enough GPU memory, fall back to CPU
            return "cpu"
        
        # GPU is available and has sufficient memory
        return self.default_gpu_device

