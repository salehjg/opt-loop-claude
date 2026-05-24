# Example: screenshot — hyperparameter tuning (in place, screenshot-mode)

Claude tunes the hyperparameters in **`hyperparams.py`** (in place) so gradient
descent on an ill-conditioned quadratic `f(w)=0.5(w0² + 100·w1²)` reaches the
target loss (1e-8) in as **few iterations as possible**, without diverging.

Each measurement runs `train.py` (regenerating `plot.html`) and screenshots it.
Claude reads the **shape** of the loss-vs-iteration curve — diverging (curve
shoots up, red), too slow (nearly flat), or fast (steep drop to the target
line) — and adjusts. Raising the learning rate then adding momentum is the path
to far fewer iterations.

## Requirements
Python 3.10+, the `claude` CLI, and Playwright:
```bash
pip install playwright && playwright install chromium
```

## Run
```bash
cd examples/screenshot
python claude_opt.py validate   # check the config
python claude_opt.py run        # start the infinite loop
python claude_opt.py fetch      # (what Claude calls) run train.py + screenshot
```
`claude_opt.py` here is a symlink to the library two levels up.
While running: type text to steer • `/pause` `/resume` `/stop` `/status` `/cost`.

## Files (the portable 3 + the project)
- `claude_opt.py`, `config.json5`, `plan.json5` — the utility you copy anywhere.
- `hyperparams.py` — **the target**: `LEARNING_RATE`, `MOMENTUM`, `NUM_ITERS`.
- `train.py` — runs gradient descent, writes a self-contained `plot.html`
  (no dependencies). Left untouched by Claude.

`config.json5` uses `pre_command: "python train.py"` then screenshots
`plot.html`. Screenshots accumulate as
`opt_runs/screenshots/screenshot_0001.png`, `…0002.png`, … so you can scrub the
optimization history visually. The history log is in `opt_runs/history.md`.

Tip: for a from-scratch demo, set `LEARNING_RATE = 0.001`, `MOMENTUM = 0.0` in
`hyperparams.py` — it starts slow and far from converged.

## Reset
```bash
git checkout hyperparams.py
rm -rf opt_runs plot.html
```
