"""Plan milestone 4: PyTorch sees a CUDA device and multiplies on it.

`torch.cuda.is_available()` must be True, and `randn(4096, 4096) @ ...` on the
device must match a CPU reference. Seeded, so a run through Tuna can be
compared with a native run on the same GPU: the checksum line should be
identical (same GPU, same libraries, same inputs).

Usage: python torch_matmul.py
"""

import sys

import torch


def main():
    available = torch.cuda.is_available()
    print(f"available: {available}", flush=True)
    if not available:
        print("FAIL: no CUDA device", flush=True)
        return 1
    print(f"device: {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}", flush=True)

    torch.manual_seed(0)
    a = torch.randn(4096, 4096, device="cuda")
    b = torch.randn(4096, 4096, device="cuda")
    c = a @ b
    reference = a.cpu().double() @ b.cpu().double()
    error = (c.cpu().double() - reference).abs().max().item()
    print(f"max abs error vs CPU (fp64): {error:.3e}", flush=True)
    print(f"checksum: {c.double().sum().item()!r}", flush=True)
    # fp32 GEMM over k = 4096 accumulates ~1e-3 absolute error on N(0, 64) values.
    ok = error < 1e-2
    print("PASS" if ok else "FAIL: result does not match the CPU", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
