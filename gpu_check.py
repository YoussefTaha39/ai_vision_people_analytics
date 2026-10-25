import subprocess
try:
    import torch
    print("PyTorch version:", torch.__version__)
    print("PyTorch CUDA runtime:", torch.version.cuda)
    print("torch.cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("PyTorch import failed:", exc)
try:
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
    print("\n--- nvidia-smi ---\n" + (result.stdout if result.returncode == 0 else result.stderr))
except Exception as exc:
    print("nvidia-smi not available:", exc)
