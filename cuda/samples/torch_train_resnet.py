"""Plan milestone 5 (part 2): a ResNet-18 training loop on the GPU.

ResNet-18 (He et al., 2016, "Deep Residual Learning for Image Recognition"),
written out here so the sample needs only torch, trained with SGD + momentum
on a fixed synthetic CIFAR-shaped dataset (3x32x32, 10 classes). Convolutions,
batch norm, ReLU, pooling, a linear head, cross-entropy and autograd all run
on the CUDA device.

--no-cudnn runs PyTorch's own convolution kernels instead of cuDNN's.

cuDNN runs in deterministic mode: its default convolution backward passes
accumulate with atomics, so two runs on the same GPU would differ, and
training amplifies the difference. Deterministic, every run (native or
through Tuna) prints the same "loss step=..." lines, which test_torch.py
compares bit for bit.

Usage: python torch_train_resnet.py [--steps N] [--batch B] [--no-cudnn]
"""

import argparse
import sys
import time

import torch
from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + self.shortcut(x))


class ResNet18(nn.Module):
    def __init__(self, classes=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        layers, cin = [], 64
        for cout, stride in ((64, 1), (128, 2), (256, 2), (512, 2)):
            layers += [BasicBlock(cin, cout, stride), BasicBlock(cout, cout, 1)]
            cin = cout
        self.layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(512, classes)

    def forward(self, x):
        return self.head(torch.flatten(self.pool(self.layers(self.stem(x))), 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--no-cudnn", action="store_true")
    args = parser.parse_args()
    torch.backends.cudnn.enabled = not args.no_cudnn
    torch.backends.cudnn.deterministic = True

    torch.manual_seed(0)
    images = torch.randn(256, 3, 32, 32)
    # Learnable labels: a fixed random projection of a downsampled image.
    labels = (nn.functional.avg_pool2d(images, 8).flatten(1) @ torch.randn(48, 10)).argmax(dim=1)

    device = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name(0)}, cudnn: {torch.backends.cudnn.enabled}", flush=True)
    x, y = images.to(device), labels.to(device)
    model = ResNet18().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    losses = []
    torch.cuda.synchronize()
    start = time.perf_counter()
    for step in range(args.steps):
        i = (step * args.batch) % 256
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x[i:i + args.batch]), y[i:i + args.batch])
        loss.backward()
        optimizer.step()
        if step % 5 == 0 or step == args.steps - 1:
            losses.append((step, loss.item()))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    for step, value in losses:
        print(f"loss step={step} value={value!r}", flush=True)
    print(f"time: {elapsed:.3f} s for {args.steps} steps ({1000 * elapsed / args.steps:.1f} ms/step)", flush=True)
    ok = all(v == v for _, v in losses) and losses[-1][1] < losses[0][1]
    print("PASS" if ok else "FAIL: the loss did not drop", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
