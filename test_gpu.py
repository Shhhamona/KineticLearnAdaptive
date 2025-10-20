"""
Test script to check GPU/CUDA availability with PyTorch
"""

import torch
import sys

print("="*70)
print("GPU/CUDA Availability Test")
print("="*70)

# Check PyTorch version
print(f"\nPyTorch version: {torch.__version__}")

# Check CUDA availability
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuda_available:
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    
    # Test a simple tensor operation on GPU
    print("\n" + "="*70)
    print("Testing tensor operations on GPU...")
    print("="*70)
    
    try:
        device = torch.device('cuda')
        x = torch.randn(1000, 1000).to(device)
        y = torch.randn(1000, 1000).to(device)
        z = torch.matmul(x, y)
        print("✅ Successfully performed matrix multiplication on GPU!")
        print(f"   Result shape: {z.shape}")
        print(f"   Result device: {z.device}")
    except Exception as e:
        print(f"❌ Error during GPU operation: {e}")
else:
    print("\n" + "="*70)
    print("CUDA is not available. Possible reasons:")
    print("="*70)
    print("1. No NVIDIA GPU detected")
    print("2. CUDA drivers not installed")
    print("3. PyTorch installed without CUDA support (CPU-only version)")
    print("\nTo install PyTorch with CUDA support, visit:")
    print("https://pytorch.org/get-started/locally/")
    print("\nFor CUDA 11.8, use:")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("\nFor CUDA 12.1, use:")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

# Check if running on CPU
print("\n" + "="*70)
print("Testing on CPU (fallback)...")
print("="*70)

try:
    device = torch.device('cpu')
    x = torch.randn(100, 100).to(device)
    y = torch.randn(100, 100).to(device)
    z = torch.matmul(x, y)
    print("✅ CPU operations working correctly!")
    print(f"   Result shape: {z.shape}")
    print(f"   Result device: {z.device}")
except Exception as e:
    print(f"❌ Error during CPU operation: {e}")

print("\n" + "="*70)
print("Recommendation:")
print("="*70)

if cuda_available:
    print("✅ Your system is ready for GPU-accelerated training!")
else:
    print("⚠️  GPU not available. Training will use CPU.")
    print("   This is slower but will still work.")
    print("   For faster training, install CUDA-enabled PyTorch.")

print("="*70)
