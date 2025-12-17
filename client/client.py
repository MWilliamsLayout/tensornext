"""
Client CLI application for sending inference requests to the server.

This is a simple command-line client that demonstrates how to interact
with the distributed AI inference server.
"""

import argparse
import requests
import json
import time
import sys
from typing import List


def create_sample_input(batch_size: int = 1, input_dim: int = 128) -> List[List[float]]:
    """
    Create sample input data for inference.
    
    Args:
        batch_size: Number of samples in the batch
        input_dim: Dimension of each input vector
    
    Returns:
        2D list representing batch of input vectors
    """
    import random
    return [[random.random() for _ in range(input_dim)] for _ in range(batch_size)]


def send_request(
    server_url: str,
    input_data: List[List[float]],
    prefer_gpu: bool = True,
    min_gpu_memory_mb: float = 100.0
) -> dict:
    """
    Send an inference request to the server.
    
    Args:
        server_url: Base URL of the server (e.g., "http://localhost:8000")
        input_data: Input data as 2D list
        prefer_gpu: Whether to prefer GPU execution
        min_gpu_memory_mb: Minimum GPU memory required
    
    Returns:
        Response dictionary from the server
    """
    url = f"{server_url}/predict"
    
    payload = {
        "input_data": input_data,
        "prefer_gpu": prefer_gpu,
        "min_gpu_memory_mb": min_gpu_memory_mb
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30.0)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error sending request: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")
        raise


def print_results(response: dict):
    """
    Pretty-print the inference results and metadata.
    
    Args:
        response: Response dictionary from the server
    """
    print("\n" + "="*60)
    print("INFERENCE RESULTS")
    print("="*60)
    
    output = response.get("output", [])
    print(f"\nOutput shape: {len(output)} x {len(output[0]) if output else 0}")
    print(f"Output (first sample, first 5 values): {output[0][:5] if output else 'N/A'}")
    
    print("\n" + "-"*60)
    print("EXECUTION METADATA")
    print("-"*60)
    print(f"Device used:        {response.get('device', 'unknown')}")
    print(f"Latency:            {response.get('latency_ms', 0):.2f} ms")
    
    gpu_memory = response.get('gpu_memory_mb')
    if gpu_memory is not None:
        print(f"GPU memory used:     {gpu_memory:.2f} MB")
    else:
        print(f"GPU memory used:     N/A (CPU execution)")
    
    print("="*60 + "\n")


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Client for distributed AI inference server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Send request preferring GPU
  python client.py --server http://192.168.1.100:8000 --prefer-gpu

  # Send request forcing CPU
  python client.py --server http://192.168.1.100:8000 --no-prefer-gpu

  # Send multiple requests
  python client.py --server http://192.168.1.100:8000 --count 5
        """
    )
    
    parser.add_argument(
        "--server",
        type=str,
        default="http://localhost:8000",
        help="Server URL (default: http://localhost:8000)"
    )
    
    parser.add_argument(
        "--prefer-gpu",
        action="store_true",
        help="Prefer GPU execution (default: True)"
    )
    
    parser.add_argument(
        "--no-prefer-gpu",
        dest="prefer_gpu",
        action="store_false",
        help="Force CPU execution"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (default: 1)"
    )
    
    parser.add_argument(
        "--input-dim",
        type=int,
        default=128,
        help="Input dimension (default: 128)"
    )
    
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of requests to send (default: 1)"
    )
    
    parser.add_argument(
        "--min-gpu-memory",
        type=float,
        default=100.0,
        help="Minimum GPU memory required in MB (default: 100.0)"
    )
    
    args = parser.parse_args()
    
    # Set default prefer_gpu to True if --no-prefer-gpu was not explicitly specified
    # (action="store_true" defaults to False, so we need to set True as the default)
    # Only override if --no-prefer-gpu was not specified (to respect explicit False)
    if '--no-prefer-gpu' not in sys.argv:
        args.prefer_gpu = True
    
    # Check server health
    try:
        health_url = f"{args.server}/health"
        health_response = requests.get(health_url, timeout=5.0)
        health_response.raise_for_status()
        health_data = health_response.json()
        print(f"Server status: {health_data.get('status')}")
        print(f"CUDA available: {health_data.get('cuda_available')}")
        print(f"Current device: {health_data.get('current_device')}")
        if 'gpu_memory_free_mb' in health_data and health_data['gpu_memory_free_mb'] is not None:
            print(f"GPU memory free: {health_data['gpu_memory_free_mb']:.2f} MB")
    except requests.exceptions.RequestException as e:
        print(f"Warning: Could not check server health: {e}")
        print("Continuing anyway...")
    
    # Send inference requests
    print(f"\nSending {args.count} request(s) to {args.server}")
    print(f"Prefer GPU: {args.prefer_gpu}")
    print(f"Batch size: {args.batch_size}, Input dim: {args.input_dim}\n")
    
    for i in range(args.count):
        if args.count > 1:
            print(f"\n--- Request {i+1}/{args.count} ---")
        
        input_data = create_sample_input(
            batch_size=args.batch_size,
            input_dim=args.input_dim
        )
        
        start_time = time.time()
        response = send_request(
            server_url=args.server,
            input_data=input_data,
            prefer_gpu=args.prefer_gpu,
            min_gpu_memory_mb=args.min_gpu_memory
        )
        client_latency = (time.time() - start_time) * 1000
        
        print_results(response)
        print(f"Client-side latency (including network): {client_latency:.2f} ms")


if __name__ == "__main__":
    main()

