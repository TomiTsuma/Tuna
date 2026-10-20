"""Plan milestone 5 (part 1): a small PyTorch training loop on the GPU.

A 3-layer MLP classifier trained with Adam on a fixed synthetic dataset shaped
like MNIST (784 features, 10 classes; no download needed). Everything after
the data generation runs on the CUDA device: forward (Linear = cuBLAS GEMMs
with a bias epilogue), cross-entropy, autograd, and the optimizer's fused
kernels.

Prints one machine-readable line per check; the loss trajectory is printed
with full precision so a run through Tuna can be compared against a native
run on the same GPU (test_torch.py does that). Exits non-zero if training did
not reduce the loss.

Usage: python torch_train_mlp.py [--steps N]
"""

import argparse
import sys
import time

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    torch.manual_seed(0)
    # A learnable synthetic task: labels come from a fixed random linear teacher.
    features = torch.randn(4096, 784)
    teacher = torch.randn(784, 10)
    labels = (features @ teacher).argmax(dim=1)

    device = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name(0)}", flush=True)
    x, y = features.to(device), labels.to(device)
    model = torch.nn.Sequential(
        torch.nn.Linear(784, 256), torch.nn.ReLU(),
        torch.nn.Linear(256, 256), torch.nn.ReLU(),
        torch.nn.Linear(256, 10),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    losses = []
    torch.cuda.synchronize()
    start = time.perf_counter()
    for step in range(args.steps):
        batch = slice((step * 256) % 4096, (step * 256) % 4096 + 256)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x[batch]), y[batch])
        loss.backward()
        optimizer.step()
        if step % 20 == 0 or step == args.steps - 1:
            losses.append((step, loss.item()))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    with torch.no_grad():
        accuracy = (model(x).argmax(dim=1) == y).float().mean().item()
    for step, value in losses:
        print(f"loss step={step} value={value!r}", flush=True)
    print(f"train accuracy: {accuracy:.4f}", flush=True)
    print(f"time: {elapsed:.3f} s for {args.steps} steps ({1000 * elapsed / args.steps:.2f} ms/step)", flush=True)
    ok = losses[-1][1] < 0.5 * losses[0][1]
    print("PASS" if ok else "FAIL: the loss did not drop", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
