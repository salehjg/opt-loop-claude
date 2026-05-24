# opt-loop-claude

**Drop a self-improving optimization loop into any project, then deploy Claude to make it faster.**

### Get the files

**One-time fetch** — grab the three files into your project root:

```bash
curl -fsSLO https://raw.githubusercontent.com/salehjg/opt-loop-claude/main/claude_opt.py \
     -fsSLO https://raw.githubusercontent.com/salehjg/opt-loop-claude/main/config.json5 \
     -fsSLO https://raw.githubusercontent.com/salehjg/opt-loop-claude/main/plan.json5
```

**Or install a self-updating fetcher** — write a `claude_opt.sh` you can commit to your
project, so you (or anyone using it) can re-run it anytime to pull the latest code +
templates from GitHub:

```bash
cat > claude_opt.sh <<'EOF'
#!/usr/bin/env bash
# claude_opt.sh — fetch the latest claude_opt.py + config.json5 + plan.json5 from
# https://github.com/salehjg/opt-loop-claude  (overwrites the local copies).
set -euo pipefail
BASE="https://raw.githubusercontent.com/salehjg/opt-loop-claude/main"
for f in claude_opt.py config.json5 plan.json5; do
  echo "Fetching $f ..."
  curl -fsSL "$BASE/$f" -o "$f"
done
echo "Done. Edit plan.json5 + config.json5, then: python claude_opt.py run"
EOF
chmod +x claude_opt.sh
```

Commit `claude_opt.sh`, then run `./claude_opt.sh` whenever you want the latest version.

### Run it

Edit `plan.json5` (what to optimize) and `config.json5` (model + how to measure), then:

```bash
python claude_opt.py validate   # check the config
python claude_opt.py run        # start the infinite optimization loop
```

---

## What it is

`claude_opt.py` is a single-file, config-driven **infinite optimization loop** for the
[Claude Code](https://claude.com/claude-code) CLI. It is a portable **3-file utility** —
`claude_opt.py` + `config.json5` + `plan.json5` — that you drop into ANY project to keep
optimizing it **in place**.

There is no separate workspace and no baseline copy: Claude edits the real project file(s)
named in `plan.json5`, measures the result, records the outcome in a history log, and
iterates — over and over, relaunching itself across rate limits and crashes.

## How it works

Claude runs the optimization loop *internally* — many `edit → measure → log` iterations
inside ONE `claude -p` invocation, with its cwd set to your `project_dir`. When Claude
exits (rate limit, natural end, or crash) the script **relaunches** it, preserving context
via `--resume`.

- **`plan.json5`** — describes only the *task*: what to optimize, what not to touch, what
  "correct" means, and optional starter ideas.
- **`config.json5`** — the model/effort, permissions, paths, loop timing, and the
  *measurement* (how a change is scored).
- **`claude_opt.py`** — injects into the prompt the exact measurement command and the loop
  mechanics (never stop, only edit the named targets, record every test in the history
  log, how to resume), then drives the outer relaunch loop.

The **history log** (e.g. `opt_runs/history.md`) is Claude's memory across launches: every
tested change, its metric, whether it was correct, how long the test took, and whether it
was kept or reverted. On restart Claude reads it first so it never repeats a failed idea.

## Three modes (same file)

```bash
python claude_opt.py run       --config config.json5   # the infinite loop (default)
python claude_opt.py fetch     --config config.json5   # measure once (Claude calls this)
python claude_opt.py validate  --config config.json5   # check the config, then exit
```

## Measurement modes

`config.json5`'s `result_fetch.mode` decides how a change is scored:

- **`command`** — run a shell command (build/bench) in `project_dir`; its stdout is the
  result, timed so the test duration is recorded.
- **`screenshot`** — optional `pre_command`, then Playwright captures a URL into
  `screenshot_dir` with incremental names; Claude *reads the image* to judge the result.
- **`none`** — no automated measurement; Claude measures manually per the plan.

## Live control

While `run` is going, the terminal is interactive — type free text to **steer** the next
launch, or use a command:

```
/pause   /resume   /stop   /status   /cost
```

## Requirements

- Python 3.10+
- The [`claude`](https://claude.com/claude-code) CLI, authenticated
- `run` / `validate` / `fetch` (command mode) need only the standard library. JSON5 is read
  with a built-in fallback parser (comments + trailing commas); install the optional
  `json5` / `pyjson5` package for full JSON5 support.
- `fetch` in **screenshot** mode additionally needs Playwright:
  ```bash
  pip install playwright && playwright install chromium
  ```

## Configuration

Every path is yours to set in `config.json5` — nothing is hardcoded. Key fields:

| Key | Meaning |
| --- | --- |
| `model` / `effort` | Claude model alias/id and reasoning effort (`low`…`max`). |
| `permission_mode` | e.g. `bypassPermissions`, `acceptEdits`, `plan`. |
| `allowed_tools` | Tools Claude may use (`Bash`, `Read`, `Edit`, …). |
| `plan_file` | Path to your `plan.json5`. |
| `project_dir` | Your project — edited **in place** (Claude's cwd). |
| `results_dir` / `history_file` | Where dated reports and the history log are written. |
| `infinite_loop_delay` / `opt_loop_delay` | Seconds between relaunches / between measurements. |
| `rate_limit_*` / `max_budget_per_launch` | Backoff timing and an optional USD cap per launch. |
| `result_fetch` | The measurement: `mode` plus its mode-specific fields. |

`plan.json5` is free-form — every key/value is rendered into a readable outline that gets
injected into the prompt. `title` is special-cased as the heading. You describe **only the
task**; the measurement command and loop mechanics are injected for you.

## Examples

Self-contained, runnable demos live in [`examples/`](examples/) (each symlinks
`claude_opt.py` from the repo root):

- **[`simple`](examples/simple/)** — command-mode. Claude speeds up a 512×512 matmul by
  editing `optimized_matmul()` in `matmul.cpp`; the harness reports speedup and whether the
  result still matches the untouched reference.
- **[`screenshot`](examples/screenshot/)** — screenshot-mode. Claude tunes the
  hyperparameters in `hyperparams.py` to reach a target loss in as few iterations as
  possible, reading the shape of the loss curve from screenshots of a generated `plot.html`.
- **[`resnet_cifar10`](examples/resnet_cifar10/)** — screenshot-mode. Claude tunes
  `hyperparams.json5` to train ResNet18 on CIFAR-10 to the highest test accuracy in the
  least wall-clock time, watching a live Plotly dashboard.

## Reset

The loop edits your real files. Use version control so you can always roll back:

```bash
git checkout <your target file(s)>
rm -rf opt_runs
```
