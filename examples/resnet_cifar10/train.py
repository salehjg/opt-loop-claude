#!/usr/bin/env python3
"""
train.py — DO NOT MODIFY. Trains ResNet18 on CIFAR-10 using the settings in
hyperparams.json5, records per-epoch train/val/test loss + accuracy, writes
metrics.json and a self-contained Plotly plot.html (served live by
dashboard.py), and prints a summary. Tune hyperparams.json5 to reach the
highest test accuracy in the least wall-clock time.

Requires: torch, torchvision, plotly.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from claude_opt import load_json5            # reuse the JSON5 reader (symlinked here)

HP = load_json5(HERE / "hyperparams.json5")  # the config Claude edits
DATA = HERE / "data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def make_model() -> nn.Module:
    m = torchvision.models.resnet18(weights=None, num_classes=10)
    m.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)  # CIFAR stem
    m.maxpool = nn.Identity()
    init = HP["init"]
    if init in ("kaiming", "xavier"):
        for mod in m.modules():
            if isinstance(mod, (nn.Conv2d, nn.Linear)):
                if init == "kaiming":
                    nn.init.kaiming_normal_(mod.weight, mode="fan_out", nonlinearity="relu")
                else:
                    nn.init.xavier_normal_(mod.weight)
                if getattr(mod, "bias", None) is not None:
                    nn.init.zeros_(mod.bias)
    return m.to(DEVICE)


def make_loaders():
    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                          T.ToTensor(), T.Normalize(MEAN, STD)])
    eval_tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    g = torch.Generator().manual_seed(int(HP["seed"]))
    perm = torch.randperm(50000, generator=g).tolist()
    val_idx, train_idx = perm[:5000], perm[5000:]
    tr = Subset(torchvision.datasets.CIFAR10(DATA, True, download=True, transform=train_tf), train_idx)
    va = Subset(torchvision.datasets.CIFAR10(DATA, True, download=True, transform=eval_tf), val_idx)
    te = torchvision.datasets.CIFAR10(DATA, False, download=True, transform=eval_tf)
    bs, pin = int(HP["batch_size"]), DEVICE == "cuda"
    mk = lambda ds, sh: DataLoader(ds, batch_size=bs, shuffle=sh, num_workers=2, pin_memory=pin)
    return mk(tr, True), mk(va, False), mk(te, False)


def build_optimizer(model):
    o = HP["optimizer"].lower()
    lr, wd = float(HP["learning_rate"]), float(HP["weight_decay"])
    if o == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=float(HP["momentum"]),
                               weight_decay=wd, nesterov=bool(HP["nesterov"]))
    if o == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if o == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    raise ValueError(f"unknown optimizer {HP['optimizer']!r}")


def build_scheduler(opt):
    s = HP["lr_schedule"].lower()
    epochs = int(HP["epochs"])
    if s == "constant":
        return None
    if s == "step":
        return torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, epochs // 3), gamma=0.1)
    if s == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    raise ValueError(f"unknown lr_schedule {HP['lr_schedule']!r}")


def run_epoch(model, loader, crit, opt=None):
    train = opt is not None
    model.train(train)
    loss_sum, correct, tot = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            out = model(x)
            loss = crit(out, y)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            loss_sum += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            tot += y.size(0)
    return loss_sum / tot, 100.0 * correct / tot


def write_outputs(m):
    (HERE / "metrics.json").write_text(json.dumps(m, indent=2))
    e = m["epochs"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss", "Accuracy (%)"))
    colors = {"train": "#2563eb", "val": "#16a34a", "test": "#dc2626"}
    for s in ("train", "val", "test"):
        fig.add_trace(go.Scatter(x=e, y=m[f"{s}_loss"], name=f"{s} loss",
                                 line=dict(color=colors[s])), 1, 1)
        fig.add_trace(go.Scatter(x=e, y=m[f"{s}_acc"], name=f"{s} acc",
                                 line=dict(color=colors[s], dash="dot")), 1, 2)
    h = m["hyperparams"]
    fig.update_layout(
        template="plotly_white", height=540,
        title=(f"ResNet18 / CIFAR-10 — best test acc {m['best_test_acc']:.2f}% "
               f"in {m['total_seconds']:.0f}s ({m['device']})<br>"
               f"<sub>{h['optimizer']} lr={h['learning_rate']} mom={h['momentum']} "
               f"wd={h['weight_decay']} sched={h['lr_schedule']} init={h['init']} "
               f"bs={h['batch_size']} ep={h['epochs']}</sub>"))
    fig.write_html(HERE / "plot.html", include_plotlyjs=True)


def main():
    torch.manual_seed(int(HP["seed"]))
    model = make_model()
    tr, va, te = make_loaders()
    crit = nn.CrossEntropyLoss()
    opt = build_optimizer(model)
    sched = build_scheduler(opt)

    m = {"hyperparams": HP, "device": DEVICE, "epochs": [],
         "train_loss": [], "val_loss": [], "test_loss": [],
         "train_acc": [], "val_acc": [], "test_acc": [],
         "best_test_acc": 0.0, "total_seconds": 0.0}
    start = time.time()
    for ep in range(1, int(HP["epochs"]) + 1):
        trl, tra = run_epoch(model, tr, crit, opt)
        vrl, vra = run_epoch(model, va, crit)
        tel, tea = run_epoch(model, te, crit)
        if sched:
            sched.step()
        m["epochs"].append(ep)
        m["train_loss"].append(round(trl, 4)); m["train_acc"].append(round(tra, 2))
        m["val_loss"].append(round(vrl, 4));   m["val_acc"].append(round(vra, 2))
        m["test_loss"].append(round(tel, 4));  m["test_acc"].append(round(tea, 2))
        m["best_test_acc"] = round(max(m["best_test_acc"], tea), 2)
        m["total_seconds"] = round(time.time() - start, 1)
        write_outputs(m)                       # live dashboard update each epoch
        print(f"epoch {ep:3d}/{HP['epochs']}  train {tra:5.2f}%  val {vra:5.2f}%  "
              f"test {tea:5.2f}%  ({m['total_seconds']:.0f}s)", flush=True)

    print(f"best_test_acc: {m['best_test_acc']:.2f}")
    print(f"final_test_acc: {m['test_acc'][-1]:.2f}")
    print(f"total_seconds: {m['total_seconds']:.1f}")
    print(f"device: {DEVICE}")
    print(f"wrote {HERE / 'plot.html'}")


if __name__ == "__main__":
    main()
