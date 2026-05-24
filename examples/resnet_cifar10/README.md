# Example: resnet_cifar10 — hyperparameter search (in place, screenshot-mode)

Claude tunes **`hyperparams.json5`** to train **ResNet18 on CIFAR-10** to the
highest **test accuracy** in the least **wall-clock time**. Each measurement
trains from scratch with the current settings and screenshots a **live Plotly
web server** showing loss and accuracy for the train / val / test sets.

Claude reads the dashboard (curves climbing fast = good; loss spiking/NaN = lr
too high; big train↔test gap = overfitting) plus the printed `best_test_acc`
and `total_seconds`, and adjusts the config.

## Requirements
Python 3.10+, the `claude` CLI, and:
```bash
pip install torch torchvision plotly playwright && playwright install chromium
```
A CUDA GPU is strongly recommended — ResNet18/CIFAR-10 on CPU is slow. CIFAR-10
auto-downloads to `./data` on the first run.

## Run
```bash
cd examples/resnet_cifar10
python claude_opt.py validate   # check the config
python claude_opt.py run        # start the infinite loop
python claude_opt.py fetch      # (what Claude calls) train once + screenshot
```
`claude_opt.py` here is a symlink to the library two levels up.
Open <http://127.0.0.1:8050> in a browser to watch the dashboard update live.
While running: type text to steer • `/pause` `/resume` `/stop` `/status` `/cost`.

## Files (the portable 3 + the project)
- `claude_opt.py`, `config.json5`, `plan.json5` — the utility you copy anywhere.
- `hyperparams.json5` — **the target** Claude edits (optimizer, lr, momentum,
  weight decay, schedule, init, batch size, epochs, …). Read by `train.py` via
  the library's JSON5 reader.
- `train.py` — trains, writes `metrics.json` + a self-contained Plotly
  `plot.html` each epoch, prints the summary. (Do not modify.)
- `dashboard.py` — minimal stdlib web server serving `plot.html` at `:8050`.
  Auto-started by the measurement command. (Do not modify.)

Screenshots accumulate as `opt_runs/screenshots/screenshot_0001.png`, … and the
history log is `opt_runs/history.md`.

## Reset
```bash
git checkout hyperparams.json5
rm -rf opt_runs data plot.html metrics.json
```
