import time
import torch

def benchmark(device, size=(1024, 1024), runs=10):
    """Benchmark matrix multiplication on a given device."""
    print(f"Benchmarking on {device}...")
    times = []
    for _ in range(runs):
        # Generate random tensors
        a = torch.randn(size, device=device)
        b = torch.randn(size, device=device)
        
        # Synchronize and measure time
        torch.cuda.synchronize(device) if device != "cpu" else None
        start = time.time()
        c = torch.matmul(a, b)
        torch.cuda.synchronize(device) if device != "cpu" else None
        times.append(time.time() - start)
    
    avg_time = sum(times) / len(times)
    print(f"Avg time for {runs} runs: {avg_time:.5f} seconds")
    return avg_time

if __name__ == "__main__":
    if torch.backends.mps.is_available():
        benchmark("mps")
    benchmark("cpu")
