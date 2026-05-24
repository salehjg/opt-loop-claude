#!/usr/bin/env python3
"""
train.py — DO NOT MODIFY. Runs gradient descent with the hyperparameters in
hyperparams.py, then renders a convergence plot to plot.html (self-contained
SVG, no dependencies) and prints a one-line summary to stdout.

claude_opt.py (screenshot mode, with `pre_command: python train.py`) runs this
to regenerate the plot and then screenshots plot.html. Claude reads the
screenshot to see the *shape* of the convergence curve — oscillation,
divergence, slow decay, or fast convergence — and tunes hyperparams.py.

Problem: minimize f(w) = 0.5 * (w0^2 + 100*w1^2) from w=[1,1] (condition
number 100). Heavy-ball momentum:  v = mu*v - lr*grad ;  w = w + v.
"""

from __future__ import annotations

import math
from pathlib import Path

import hyperparams as hp

TARGET_LOSS = 1e-8       # converged once loss <= this
DIVERGE_LOSS = 1e12      # treated as blown up past this
CURV = (1.0, 100.0)      # per-axis curvature → condition number 100
HERE = Path(__file__).resolve().parent


def f(w):
    return 0.5 * (CURV[0] * w[0] ** 2 + CURV[1] * w[1] ** 2)


def grad(w):
    return [CURV[0] * w[0], CURV[1] * w[1]]


def run():
    lr = float(hp.LEARNING_RATE)
    mu = float(hp.MOMENTUM)
    iters = int(hp.NUM_ITERS)

    w = [1.0, 1.0]
    v = [0.0, 0.0]
    hist = [f(w)]                 # loss at iteration 0
    status = "did_not_converge"
    converge_iter = None

    for t in range(1, iters + 1):
        g = grad(w)
        v[0] = mu * v[0] - lr * g[0]
        v[1] = mu * v[1] - lr * g[1]
        w[0] += v[0]
        w[1] += v[1]
        loss = f(w)

        if not math.isfinite(loss) or loss > DIVERGE_LOSS:
            hist.append(DIVERGE_LOSS)
            status = "diverged"
            converge_iter = t          # iteration where it blew up
            break

        hist.append(loss)
        if loss <= TARGET_LOSS:
            status = "converged"
            converge_iter = t
            break

    info = {
        "lr": lr, "mu": mu, "num_iters": iters,
        "status": status,
        "converge_iter": converge_iter,
        "final_loss": hist[-1],
        "best_loss": min(hist),
        "target": TARGET_LOSS,
    }
    return hist, info


# ── self-contained SVG convergence plot ─────────────────────────────────────

def make_svg(hist, info):
    W, H = 920, 520
    L, R, T, B = 75, 35, 110, 55
    pw, ph = W - L - R, H - T - B

    eps = 1e-12
    logs = [math.log10(max(v, eps)) for v in hist]
    lo, hi = math.floor(min(logs)), math.ceil(max(logs))
    if hi <= lo:
        hi = lo + 1
    n = len(hist)

    def X(i):
        return L + (i / (n - 1) * pw if n > 1 else pw / 2)

    def Y(val):
        ly = math.log10(max(val, eps))
        return T + (hi - ly) / (hi - lo) * ph

    color = {"converged": "#16a34a",
             "diverged": "#dc2626",
             "did_not_converge": "#2563eb"}[info["status"]]

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="ui-monospace,Menlo,monospace">']
    parts.append(f'<rect width="{W}" height="{H}" fill="#0b1020"/>')

    # decade gridlines + y labels
    for d in range(lo, hi + 1):
        y = T + (hi - d) / (hi - lo) * ph
        parts.append(f'<line x1="{L}" y1="{y:.1f}" x2="{L+pw}" y2="{y:.1f}" '
                     f'stroke="#1e293b" stroke-width="1"/>')
        parts.append(f'<text x="{L-10}" y="{y+4:.1f}" fill="#64748b" '
                     f'font-size="12" text-anchor="end">1e{d}</text>')

    # target line
    yt = Y(info["target"])
    if T <= yt <= T + ph:
        parts.append(f'<line x1="{L}" y1="{yt:.1f}" x2="{L+pw}" y2="{yt:.1f}" '
                     f'stroke="#22d3ee" stroke-width="1.5" stroke-dasharray="6 4"/>')
        parts.append(f'<text x="{L+pw}" y="{yt-6:.1f}" fill="#22d3ee" '
                     f'font-size="12" text-anchor="end">target {info["target"]:.0e}</text>')

    # axes
    parts.append(f'<line x1="{L}" y1="{T}" x2="{L}" y2="{T+ph}" stroke="#475569"/>')
    parts.append(f'<line x1="{L}" y1="{T+ph}" x2="{L+pw}" y2="{T+ph}" stroke="#475569"/>')
    parts.append(f'<text x="{L+pw/2:.0f}" y="{H-15}" fill="#94a3b8" '
                 f'font-size="13" text-anchor="middle">iteration</text>')
    # x end label
    parts.append(f'<text x="{L+pw:.0f}" y="{T+ph+20:.0f}" fill="#64748b" '
                 f'font-size="12" text-anchor="end">{n-1}</text>')
    parts.append(f'<text x="{L}" y="{T+ph+20:.0f}" fill="#64748b" '
                 f'font-size="12" text-anchor="start">0</text>')

    # the curve
    pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(hist))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" '
                 f'stroke-width="2.5"/>')

    # header text
    title = "Gradient-Descent Convergence  f(w)=0.5(w0^2 + 100 w1^2)"
    parts.append(f'<text x="{L}" y="34" fill="#e2e8f0" font-size="20" '
                 f'font-weight="bold">{title}</text>')
    hp_line = (f"lr={info['lr']:g}   momentum={info['mu']:g}   "
               f"num_iters={info['num_iters']}")
    parts.append(f'<text x="{L}" y="60" fill="#cbd5e1" font-size="15">{hp_line}</text>')

    if info["status"] == "converged":
        st = f"CONVERGED at iteration {info['converge_iter']}  (fewer is better)"
    elif info["status"] == "diverged":
        st = f"DIVERGED at iteration {info['converge_iter']}  — lower the learning rate"
    else:
        st = (f"DID NOT CONVERGE in {info['num_iters']} iters  "
              f"(final loss {info['final_loss']:.2e})")
    parts.append(f'<text x="{L}" y="84" fill="{color}" font-size="16" '
                 f'font-weight="bold">{st}</text>')

    parts.append('</svg>')
    return "\n".join(parts)


def write_html(hist, info):
    svg = make_svg(hist, info)
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>convergence</title>"
        "<style>body{margin:0;background:#0b1020;display:flex;"
        "justify-content:center;padding:24px}</style></head>"
        f"<body>{svg}</body></html>"
    )
    (HERE / "plot.html").write_text(html)


def main():
    hist, info = run()
    write_html(hist, info)
    ci = info["converge_iter"]
    print(f"status: {info['status']}")
    print(f"iters_to_target: {ci if (info['status']=='converged') else '-'}")
    print(f"final_loss: {info['final_loss']:.3e}   best_loss: {info['best_loss']:.3e}")
    print(f"hyperparams: lr={info['lr']:g} momentum={info['mu']:g} "
          f"num_iters={info['num_iters']}")
    print(f"wrote {HERE / 'plot.html'}")


if __name__ == "__main__":
    main()
