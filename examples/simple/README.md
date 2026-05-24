# Example: simple — matmul (in place, command-mode)

Claude iteratively speeds up a 512×512 dense matrix multiply by editing
**`optimized_matmul()` in `matmul.cpp`** — in place, one file. The same file
holds `gold_matmul()`, the untouched reference: every run executes both, so the
program reports the speedup **and** whether the optimized result still matches
the reference (`correct: yes/no`). No baseline copy, no checksum bookkeeping.

## Requirements
Python 3.10+, the `claude` CLI, and `g++` (C++17, AVX2).

## Run
```bash
cd examples/simple
python claude_opt.py validate   # check the config
python claude_opt.py run        # start the infinite loop
python claude_opt.py fetch      # (what Claude calls) compile + run, print result
```
`claude_opt.py` here is a symlink to the library two levels up.
While running: type text to steer • `/pause` `/resume` `/stop` `/status` `/cost`.

## Files (the portable 3 + the project)
- `claude_opt.py`, `config.json5`, `plan.json5` — the utility you copy anywhere.
- `matmul.cpp` — **the project**: edit `optimized_matmul()`, leave `gold_matmul()` alone.

The measurement command (in `config.json5`) compiles + runs `matmul.cpp`; the
binary prints `gold_ms / opt_ms / speedup / rel_err / correct`. Runtime outputs
go to `./opt_runs/` (set by `results_dir` / `history_file` in the config) — the
history log there is what Claude reads to know what's been tried and how long
tests take.

## Reset
```bash
git checkout matmul.cpp
rm -rf opt_runs
```
