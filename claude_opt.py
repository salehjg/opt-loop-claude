#!/usr/bin/env python3
"""
claude_opt.py — single-file, config-driven infinite optimization loop for
Claude Code.

A portable 3-file utility — claude_opt.py + config.json5 + plan.json5 — that you
drop into ANY project to keep optimizing it IN PLACE. There is no separate
workspace and no baseline copy: Claude edits the real project file(s) named in
plan.json5, measures, records the outcome in a history log, and iterates.

Invoked in three modes (same file):

    python claude_opt.py run      --config config.json5   # the infinite loop (default)
    python claude_opt.py fetch    --config config.json5   # measure once (Claude calls this)
    python claude_opt.py validate --config config.json5   # check the config, then exit

How it works
------------
Claude runs the optimization loop *internally* — many edit / measure / log
iterations inside ONE ``claude -p`` invocation, with its cwd set to your
``project_dir``. When Claude exits (rate limit, natural end, or crash) this
script relaunches it, preserving context via ``--resume``. plan.json5 describes
the task; this script *injects* into the prompt the exact measurement command
(``claude_opt.py fetch``) and the loop mechanics (never stop, only edit the
named targets, record every test in the history log). ``fetch`` either runs a
shell command, or takes a Playwright screenshot of a URL.

All paths (project_dir, results_dir, history_file, screenshot_dir, ...) are set
by you in config.json5 — nothing is hardcoded.

Dependencies
------------
``run`` / ``validate`` / ``fetch`` (command mode) need only the standard library
and the ``claude`` CLI. JSON5 is read with a built-in fallback parser (comments
+ trailing commas) if the optional ``json5`` / ``pyjson5`` package is absent.
``fetch`` in screenshot mode additionally needs Playwright:

    pip install playwright && playwright install chromium
"""

from __future__ import annotations

import argparse
import json
import logging
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


# ===========================================================================
# JSON5 reader  (no hard dependency)
# ===========================================================================

def _strip_json5(text: str) -> str:
    """Strip ``//`` and ``/* */`` comments and trailing commas from JSON5,
    leaving valid JSON. String-aware so delimiters inside strings survive.

    This is the practically useful subset of JSON5 (comments + trailing
    commas). For full JSON5 install the ``json5`` package — it is preferred
    automatically when present.
    """
    out: list[str] = []
    i, n = 0, len(text)
    in_str = False
    quote = ""
    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            if c == "\\" and i + 1 < n:        # keep escaped char verbatim
                out.append(text[i + 1])
                i += 2
                continue
            if c == quote:
                in_str = False
            i += 1
            continue
        if c in ('"', "'"):
            in_str = True
            quote = c
            out.append(c)
            i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            i += 2
            while i < n and text[i] != "\n":
                i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2
            continue
        out.append(c)
        i += 1

    no_comments = "".join(out)
    no_trailing = re.sub(r",(\s*[}\]])", r"\1", no_comments)
    return no_trailing


def load_json5(path: Path) -> dict:
    """Load a JSON5 file as a dict, preferring a real json5 lib if installed."""
    text = Path(path).read_text()
    for mod_name in ("json5", "pyjson5"):
        try:
            mod = __import__(mod_name)
        except ImportError:
            continue
        loads = getattr(mod, "loads", None) or getattr(mod, "decode", None)
        if loads:
            return loads(text)
    try:
        return json.loads(_strip_json5(text))
    except json.JSONDecodeError as e:
        raise ConfigError(
            f"{path}: not valid JSON5 (built-in parser handles comments + "
            f"trailing commas only; install `json5` for full support): {e}"
        ) from e


class ConfigError(Exception):
    """Raised when the config file is missing keys or has wrong types."""


# ===========================================================================
# Stream-JSON parser  (Claude Code --output-format stream-json)
# ===========================================================================

class StreamParser:
    """Parses Claude Code ``--output-format stream-json`` events into a live
    text stream, tracking session id, cost, tool calls and rate-limit state."""

    def __init__(self):
        self.session_id: Optional[str] = None
        self.total_cost: float = 0.0
        self.rate_limit_info: Optional[dict] = None
        self.result_text: str = ""
        self.is_error: bool = False

        self._text_parts: list[str] = []
        self._tool_uses: list[dict] = []
        self._current_tool: Optional[dict] = None
        self._seen_stream_events: bool = False
        self._trailing_newlines: int = 0

    @property
    def full_text(self) -> str:
        return "".join(self._text_parts)

    def _emit(self, text: str) -> str:
        """Append ``text``, capping consecutive newlines at 2. Returns what
        was actually appended so console and transcript stay in sync."""
        if not text:
            return ""
        leading = len(text) - len(text.lstrip("\n"))
        allowed = max(0, 2 - self._trailing_newlines)
        if leading > allowed:
            text = "\n" * allowed + text[leading:]
        if not text:
            return ""
        if text.strip("\n") == "":
            self._trailing_newlines += len(text)
        else:
            self._trailing_newlines = len(text) - len(text.rstrip("\n"))
        self._text_parts.append(text)
        return text

    def feed(self, line: str) -> Optional[str]:
        """Feed one JSON line. Returns a displayable text chunk or None."""
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return line  # non-JSON passthrough

        etype = event.get("type", "")

        if etype == "system" and event.get("subtype") == "init":
            self.session_id = event.get("session_id")
            return None
        if etype == "stream_event":
            self._seen_stream_events = True
            return self._handle_stream_event(event.get("event", {}))
        if etype == "assistant":
            if self._seen_stream_events:
                return None
            return self._handle_assistant(event.get("message", {}))
        if etype == "user":
            msg = event.get("message", {})
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if block.get("type") == "tool_result":
                        rt = self._extract_tool_result_text(block.get("content", ""))
                        if rt:
                            self._emit(f"[output: {rt[:500]}]\n")
            return None
        if etype == "rate_limit_event":
            self.rate_limit_info = event.get("rate_limit_info", {})
            return None
        if etype == "result":
            self.result_text = event.get("result", "")
            self.is_error = event.get("is_error", False)
            self.total_cost += event.get("total_cost_usd", 0)
            sid = event.get("session_id")
            if sid:
                self.session_id = sid
            return None
        return None

    @staticmethod
    def _extract_tool_result_text(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return "".join(parts)
        return ""

    def is_rate_limited(self) -> bool:
        if not self.rate_limit_info:
            return False
        return self.rate_limit_info.get("status") != "allowed"

    def rate_limit_resets_at(self) -> Optional[int]:
        if self.rate_limit_info:
            return self.rate_limit_info.get("resetsAt")
        return None

    @property
    def tool_count(self) -> int:
        return len(self._tool_uses)

    def _handle_stream_event(self, ev: dict) -> Optional[str]:
        inner = ev.get("type", "")
        if inner == "content_block_delta":
            delta = ev.get("delta", {})
            if delta.get("type") == "text_delta":
                return self._emit(delta.get("text", "")) or None
            return None
        if inner == "content_block_start":
            block = ev.get("content_block", {})
            if block.get("type") == "tool_use":
                self._current_tool = {"name": block.get("name"), "id": block.get("id")}
                return self._emit(f"\n[tool: {block.get('name')}]\n") or None
            return None
        if inner == "content_block_stop":
            if self._current_tool:
                self._tool_uses.append(self._current_tool)
                self._current_tool = None
                return None
            return self._emit("\n") or None
        return None

    def _handle_assistant(self, msg: dict) -> Optional[str]:
        content = msg.get("content", [])
        if isinstance(content, str):
            return self._emit(content) or None
        texts = []
        for block in content:
            if block.get("type") == "text":
                emitted = self._emit(block.get("text", ""))
                if emitted:
                    texts.append(emitted)
        return "".join(texts) if texts else None


# ===========================================================================
# ResultFetcher — the `fetch` subcommand (the measurement)
# ===========================================================================

class ResultFetcher:
    """Measures the current state once and prints a ``RESULT_*`` contract line
    that Claude parses. Modes, selected by ``result_fetch.mode``:

      * ``command``    — run a shell command in ``project_dir`` and capture its
                         stdout (timed, so the test duration is reported).
      * ``screenshot`` — optional ``pre_command`` then a Playwright capture of a
                         URL to an incrementally-named file in ``screenshot_dir``.
      * ``none``       — no automated measurement.

    A ``opt_loop_delay`` cooldown is enforced between successive fetches using a
    timestamp file under ``results_dir``.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        rf = cfg["result_fetch"]
        self.mode: str = rf["mode"]
        self.project_dir = Path(cfg["project_dir"]).resolve()
        self.results_dir = Path(cfg["results_dir"]).resolve()
        self.opt_loop_delay = float(cfg.get("opt_loop_delay", 0))
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._cooldown_file = self.results_dir / ".last_fetch"

    # ── cooldown (the opt_loop_delay) ──────────────────────────────────────

    def _enforce_cooldown(self) -> None:
        if self.opt_loop_delay <= 0:
            return
        try:
            last = float(self._cooldown_file.read_text().strip())
        except (OSError, ValueError):
            last = 0.0
        wait = self.opt_loop_delay - (time.time() - last)
        if wait > 0:
            print(f"[fetch] opt_loop_delay cooldown — waiting {wait:.1f}s",
                  file=sys.stderr)
            time.sleep(wait)

    def _touch_cooldown(self) -> None:
        try:
            self._cooldown_file.write_text(str(time.time()))
        except OSError:
            pass

    # ── command mode ────────────────────────────────────────────────────────

    def _run_command(self) -> tuple[int, str, Path, float]:
        cmd = self.cfg["result_fetch"]["command"]
        start = time.monotonic()
        proc = subprocess.run(cmd, shell=True, cwd=str(self.project_dir),
                              capture_output=True, text=True)
        secs = time.monotonic() - start
        combined = proc.stdout + (f"\n[stderr]\n{proc.stderr}" if proc.stderr else "")
        result_path = self.results_dir / f"result_{self.run_timestamp}.txt"
        result_path.write_text(combined if combined.endswith("\n") else combined + "\n")
        (self.results_dir / f"fetch_{self.run_timestamp}.log").write_text(
            f"$ {cmd}\n(returncode {proc.returncode}, {secs:.2f}s)\n\n{combined}\n"
        )
        return proc.returncode, combined, result_path, secs

    # ── screenshot mode ──────────────────────────────────────────────────────

    @staticmethod
    def _next_index(folder: Path, prefix: str, suffix: str) -> int:
        pat = re.compile(rf"^{re.escape(prefix)}_(\d+){re.escape(suffix)}$")
        highest = 0
        if folder.is_dir():
            for p in folder.iterdir():
                m = pat.match(p.name)
                if m:
                    highest = max(highest, int(m.group(1)))
        return highest + 1

    def _resolve_url(self, url: str) -> str:
        """A url with no scheme is treated as a path relative to project_dir and
        turned into a file:// URL — keeps configs machine-independent."""
        if "://" in url:
            return url
        p = Path(url)
        if not p.is_absolute():
            p = self.project_dir / p
        return p.resolve().as_uri()

    def _screenshot(self) -> tuple[Path, str, str]:
        """Returns (image_path, resolved_url, pre_command_note)."""
        rf = self.cfg["result_fetch"]

        pre_note = ""
        pre = rf.get("pre_command")
        if pre:
            p = subprocess.run(pre, shell=True, cwd=str(self.project_dir),
                               capture_output=True, text=True)
            pre_note = f"$ {pre}  (rc={p.returncode})\n" + (p.stdout or "")
            if p.stderr:
                pre_note += f"[stderr] {p.stderr}"
            if not pre_note.endswith("\n"):
                pre_note += "\n"

        folder = Path(rf["screenshot_dir"]).resolve()
        folder.mkdir(parents=True, exist_ok=True)
        prefix = rf.get("screenshot_prefix", "screenshot")
        n = self._next_index(folder, prefix, ".png")
        path = folder / f"{prefix}_{n:04d}.png"

        try:
            from playwright.sync_api import sync_playwright
        except ImportError as e:
            raise RuntimeError(
                "screenshot mode requires Playwright. Install with:\n"
                "    pip install playwright && playwright install chromium"
            ) from e

        url = self._resolve_url(rf["url"])
        w, h = rf.get("viewport", [1920, 1080])
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": int(w), "height": int(h)})
            page.goto(url, wait_until="load")
            page.wait_for_timeout(int(rf.get("wait_ms", 3000)))
            page.screenshot(path=str(path), full_page=bool(rf.get("full_page", True)))
            browser.close()
        return path, url, pre_note

    # ── entry point ─────────────────────────────────────────────────────────

    def main(self) -> int:
        self._enforce_cooldown()
        try:
            if self.mode == "command":
                rc, out, path, secs = self._run_command()
                print(f"RESULT_FILE: {path}")
                print("---")
                sys.stdout.write(out if out.endswith("\n") else out + "\n")
                print(f"measurement_seconds: {secs:.2f}")
                if rc != 0:
                    print(f">>> MEASUREMENT COMMAND FAILED (returncode {rc}) <<<")
            elif self.mode == "screenshot":
                path, url, pre_note = self._screenshot()
                print(f"RESULT_IMAGE: {path}")
                print("---")
                if pre_note:
                    sys.stdout.write(pre_note)
                print(f"screenshot of {url} captured at {self.run_timestamp}")
                rc = 0
            elif self.mode == "none":
                print("RESULT_STDOUT:")
                print("---")
                print("No automated measurement configured (result_fetch.mode "
                      "== 'none'). Measure manually per the plan.")
                rc = 0
            else:
                raise ConfigError(f"unknown result_fetch.mode: {self.mode!r}")
        finally:
            self._touch_cooldown()
        return rc


# ===========================================================================
# ClaudeOptLoop — the orchestrator (the `run` subcommand)
# ===========================================================================

class ClaudeOptLoop:
    """Infinite outer loop around the Claude Code CLI, driven by a JSON5 config.

    Public surface mandated by the design goals:
      * :meth:`validate_config`  — verify every required key exists / typechecks.
      * :meth:`build_dirs`       — create the directories named in the config.
      * :meth:`run`              — the infinite relaunch loop.
    """

    # Nested schema of required keys → expected type(s). ``result_fetch`` keys
    # are mode-dependent and validated separately below.
    REQUIRED: dict[str, Any] = {
        "model": str,
        "effort": str,
        "permission_mode": str,
        "allowed_tools": list,
        "plan_file": str,
        "project_dir": str,
        "results_dir": str,
        "history_file": str,
        "stderr_log_file": (str, type(None)),
        "infinite_loop_delay": (int, float),
        "opt_loop_delay": (int, float),
        "rate_limit_base_wait": (int, float),
        "rate_limit_max_wait": (int, float),
        "max_budget_per_launch": (int, float, type(None)),
        "result_fetch": {"mode": str},
    }

    def __init__(self, cfg: dict, config_path: Path):
        self.cfg = cfg
        self.config_path = Path(config_path).resolve()
        self.self_path = Path(__file__).resolve()

        self.validate_config()  # raises ConfigError listing every problem

        self.project_dir = Path(cfg["project_dir"]).resolve()
        self.results_dir = Path(cfg["results_dir"]).resolve()
        self.history_path = Path(cfg["history_file"]).resolve()
        self.plan_path = Path(cfg["plan_file"]).resolve()
        if not self.project_dir.is_dir():
            raise ConfigError(f"project_dir does not exist: {self.project_dir}")
        if not self.plan_path.is_file():
            raise ConfigError(f"plan_file not found: {self.plan_path}")

        # Runtime state
        self.launch_count = 0
        self.session_id: Optional[str] = None
        self.total_cost_usd = 0.0
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.pause_event.set()  # start un-paused
        self.steering_queue: "queue.Queue[str]" = queue.Queue()
        self._current_proc: Optional[subprocess.Popen] = None

        # Heavy setup deferred to prepare() so `validate` has no side effects.
        self.plan: dict = {}
        self._system_prompt: str = ""

    def prepare(self) -> None:
        """Create dirs, set up logging, load the plan and assemble the prompt.
        Called by run(); kept separate so validate has no side effects."""
        self.build_dirs()
        self._setup_logging()
        self.plan = load_json5(self.plan_path)
        self._system_prompt = self._build_system_prompt()
        self._write_assembled_plan()

    # ── validation ───────────────────────────────────────────────────────────

    def validate_config(self) -> None:
        """Verify every required key exists with the right type. Raises
        :class:`ConfigError` listing *all* problems found (not just the first)."""
        errors: list[str] = []
        self._validate_node(self.REQUIRED, self.cfg, "", errors)

        rf = self.cfg.get("result_fetch")
        if isinstance(rf, dict):
            mode = rf.get("mode")
            extra: dict[str, Any] = {}
            if mode == "command":
                extra = {"command": str}
            elif mode == "screenshot":
                extra = {"url": str, "screenshot_dir": str,
                         "viewport": list, "wait_ms": (int, float), "full_page": bool}
            elif mode not in ("none", None):
                errors.append(
                    f"result_fetch.mode: expected one of "
                    f"'command'|'screenshot'|'none', got {mode!r}")
            self._validate_node(extra, rf, "result_fetch", errors)

        if errors:
            raise ConfigError(
                f"config {self.config_path} is invalid:\n  - " + "\n  - ".join(errors))

    def _validate_node(self, schema: dict, data: Any, path: str,
                       errors: list[str]) -> None:
        if not isinstance(data, dict):
            errors.append(f"{path or '<root>'}: expected object")
            return
        for key, expected in schema.items():
            full = f"{path}.{key}" if path else key
            if key not in data:
                errors.append(f"missing key: {full}")
                continue
            val = data[key]
            if isinstance(expected, dict):
                self._validate_node(expected, val, full, errors)
            else:
                types = expected if isinstance(expected, tuple) else (expected,)
                bad_bool = isinstance(val, bool) and bool not in types
                if not isinstance(val, types) or bad_bool:
                    names = "|".join(t.__name__ for t in types)
                    errors.append(f"{full}: expected {names}, got {type(val).__name__}")

    # ── directory construction ────────────────────────────────────────────────

    def build_dirs(self) -> None:
        """Create the directories named in the config (never project_dir — that
        is your existing project)."""
        dirs = [Path(self.cfg["results_dir"]),
                Path(self.cfg["history_file"]).parent]
        if self.cfg.get("stderr_log_file"):
            dirs.append(Path(self.cfg["stderr_log_file"]).parent)
        rf = self.cfg.get("result_fetch", {})
        if rf.get("mode") == "screenshot" and rf.get("screenshot_dir"):
            dirs.append(Path(rf["screenshot_dir"]))
        for d in dirs:
            d.resolve().mkdir(parents=True, exist_ok=True)

    # ── logging ────────────────────────────────────────────────────────────────

    def _setup_logging(self) -> None:
        self.logger = logging.getLogger("ClaudeOpt")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        ch = logging.StreamHandler(sys.stderr)
        ch.setFormatter(
            logging.Formatter("\033[90m%(asctime)s [%(levelname)s]\033[0m %(message)s"))
        self.logger.addHandler(ch)
        stderr_log = self.cfg.get("stderr_log_file")
        if stderr_log:
            fh = logging.FileHandler(Path(stderr_log).resolve())
            fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            self.logger.addHandler(fh)

    # ── plan assembly + injection ──────────────────────────────────────────────

    def _render_plan(self) -> str:
        """Render the hierarchical plan.json5 into a readable outline."""
        lines: list[str] = []

        def walk(key: Optional[str], val: Any, depth: int):
            indent = "  " * depth
            if isinstance(val, dict):
                if key is not None:
                    lines.append(f"{indent}- **{key}**:")
                for k, v in val.items():
                    walk(k, v, depth + 1 if key is not None else depth)
            elif isinstance(val, list):
                lines.append(f"{indent}- **{key}**:")
                for item in val:
                    if isinstance(item, (dict, list)):
                        walk("·", item, depth + 1)
                    else:
                        lines.append(f"{indent}  - {item}")
            else:
                lines.append(f"{indent}- **{key}**: {val}")

        title = self.plan.get("title")
        body = {k: v for k, v in self.plan.items() if k != "title"}
        header = f"# {title}\n\n" if title else ""
        walk(None, body, 0)
        return header + "\n".join(lines)

    def _fetch_instructions(self) -> str:
        rf = self.cfg["result_fetch"]
        mode = rf["mode"]
        cmd = f"python {self.self_path} fetch --config {self.config_path}"
        if mode == "command":
            what = (
                "It runs your measurement command in the project dir and saves the\n"
                "output to a dated report. Its first stdout line is\n"
                "`RESULT_FILE: <path>`; the text after `---` is the full report,\n"
                "ending with `measurement_seconds:` — the test duration, which you\n"
                "MUST record in the history entry.")
        elif mode == "screenshot":
            what = (
                f"It regenerates and screenshots {rf['url']} into\n"
                f"{Path(rf['screenshot_dir']).resolve()} with an incremental name\n"
                "(screenshot_0001.png, screenshot_0002.png, ...). Its first stdout\n"
                "line is `RESULT_IMAGE: <path>` — read that image to judge the result.")
        else:
            return (
                "=== MEASUREMENT ===\n"
                "No automated measurement is configured. Measure manually as the\n"
                "plan describes, and record outcomes in the history log.")
        return (
            "=== HOW TO MEASURE (run this after every change) ===\n"
            f"    {cmd}\n\n{what}\n"
            f"A cooldown of {self.cfg['opt_loop_delay']}s is enforced between calls.")

    def _loop_rules(self) -> str:
        hist = self.history_path
        return (
            "=== INFINITE OPTIMIZATION LOOP — ABSOLUTE RULES ===\n"
            "You are inside an INFINITE optimization loop managed by a Python\n"
            "orchestrator that RELAUNCHES you every time you exit. There is no end.\n\n"
            "1. NEVER say the task is complete, finished, or done; NEVER suggest\n"
            "   stopping — there are always more improvements.\n"
            "2. In ONE session do AS MANY iterations as you can. Each iteration =\n"
            "   pick an idea -> edit the target file(s) -> MEASURE (see below) ->\n"
            "   record the outcome in the history log -> next iteration.\n"
            "3. Keep a change ONLY if the measurement shows it is CORRECT and\n"
            "   BETTER than before; otherwise revert it.\n"
            "4. You are editing the REAL project in place. ONLY modify the file(s)\n"
            "   named as targets in the plan. NEVER modify claude_opt.py,\n"
            "   config.json5, plan.json5, or anything under the results dir.\n"
            "5. Return only on a rate limit or CLI termination; you will be\n"
            "   relaunched with the same session context.\n\n"
            f"THE HISTORY LOG IS YOUR MEMORY: {hist}\n"
            "After every change you test, append a brief entry with bash `>>`\n"
            "(atomic for writes <= 4 KB) — DO NOT use Write/Edit on it. Format:\n"
            f"    cat <<'EOF' >> {hist}\n"
            "\n"
            "    ### iteration N — $(date -Iseconds)\n"
            "    - Tested: <approach / what you changed>\n"
            "    - Result: <metric vs previous/baseline>; correct: <yes/no>\n"
            "    - Test took: <seconds, from measurement_seconds>\n"
            "    - Kept / reverted: <which> — <why>\n"
            "    - Next: <next idea>\n"
            "    EOF\n\n"
            "ON RESTART — read the history FIRST, before editing anything:\n"
            f"   1. tail -300 {hist}   (what's been tried + results; the\n"
            "      `## LAUNCH N` markers show how long each test/session ran —\n"
            "      use them to budget what you attempt this session)\n"
            f"   2. ls -t {self.results_dir} | head -3   (newest report — read it)\n"
            "   3. read the current target file(s).\n"
            "Do NOT retry ideas the history records as already failed.\n"
            f"\nProject dir: {self.project_dir}"
            f"\nResults dir: {self.results_dir}"
            f"\nHistory log: {hist}\n"
        )

    def _build_system_prompt(self) -> str:
        return "\n\n".join([
            self._render_plan(),
            self._fetch_instructions(),
            self._loop_rules(),
        ])

    def _write_assembled_plan(self) -> None:
        """Persist the fully-assembled prompt as an artefact (plan.json5 is
        never mutated)."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            (self.results_dir / f"assembled_plan_{ts}.md").write_text(
                self._system_prompt + "\n")
        except OSError:
            pass

    def _build_launch_prompt(self) -> str:
        parts: list[str] = []
        if self.session_id is None:
            parts.append(
                "Start (or resume) the infinite optimization loop. First catch up "
                f"on prior state (tail {self.history_path}; newest report in "
                f"{self.results_dir}; the current target file(s)), then iterate per "
                "the plan. Make as many iterations as this session allows; exit "
                "only on a rate limit.")
        else:
            parts.append(
                "Continue the infinite optimization loop. Quickly verify on-disk "
                f"state (tail {self.history_path}; newest report; current target "
                "file(s)), then keep iterating per the plan. Exit only on a rate limit.")
        steering: list[str] = []
        while True:
            try:
                steering.append(self.steering_queue.get_nowait())
            except queue.Empty:
                break
        if steering:
            parts.append("=== USER GUIDANCE (apply now) ===")
            parts.extend(steering)
        return "\n\n".join(parts)

    # ── CLI command ──────────────────────────────────────────────────────────

    def _build_cmd(self, prompt: str) -> list[str]:
        cmd = [
            "claude", "-p", prompt,
            "--verbose",
            "--output-format", "stream-json",
            "--include-partial-messages",
            "--model", self.cfg["model"],
            "--effort", self.cfg["effort"],
            "--permission-mode", self.cfg["permission_mode"],
            "--append-system-prompt", self._system_prompt,
        ]
        if self.cfg["allowed_tools"]:
            cmd.extend(["--allowed-tools", " ".join(self.cfg["allowed_tools"])])
        if self.cfg.get("max_budget_per_launch") is not None:
            cmd.extend(["--max-budget-usd", str(self.cfg["max_budget_per_launch"])])
        if self.session_id:
            cmd.extend(["--resume", self.session_id])
        return cmd

    # ── run one Claude launch ──────────────────────────────────────────────────

    def _run_claude(self, prompt: str) -> tuple[StreamParser, bool]:
        cmd = self._build_cmd(prompt)
        parser = StreamParser()
        rate_limited = False
        started = datetime.now().astimezone()
        started_mono = time.monotonic()

        with open(self.history_path, "a", buffering=1) as log_f:
            log_f.write(self._launch_header(started))
            try:
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    cwd=str(self.project_dir), text=True, bufsize=1)
                self._current_proc = proc
                for raw_line in proc.stdout:
                    line = raw_line.rstrip("\n")
                    if not line:
                        continue
                    display = parser.feed(line)
                    if display:
                        sys.stdout.write(display)
                        sys.stdout.flush()
                        log_f.write(display)
                    if parser.session_id and self.session_id is None:
                        self.session_id = parser.session_id
                proc.wait(timeout=30)
                stderr_out = proc.stderr.read()
                if stderr_out:
                    self.logger.debug(f"stderr: {stderr_out[:500]}")
                    if any(k in stderr_out.lower()
                           for k in ("rate limit", "429", "overloaded")):
                        rate_limited = True
                if proc.returncode not in (0, None):
                    self.logger.warning(f"Claude exited with code {proc.returncode}")
            except subprocess.TimeoutExpired:
                proc.kill()
                self.logger.warning("Claude wait timed out — killed")
            except Exception as e:
                self.logger.error(f"Error running Claude: {e}")
            finally:
                self._current_proc = None
                sys.stdout.write("\n")
                sys.stdout.flush()
                if parser.session_id:
                    self.session_id = parser.session_id
                self.total_cost_usd += parser.total_cost
                if parser.is_rate_limited():
                    rate_limited = True
                duration_s = time.monotonic() - started_mono
                ended = datetime.now().astimezone()
                log_f.write(self._launch_footer(ended, duration_s, parser, rate_limited))
        return parser, rate_limited

    # ── history cycle markers ────────────────────────────────────────────────

    @staticmethod
    def _format_duration(seconds: float) -> str:
        s = int(seconds)
        h, rem = divmod(s, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h}h {m}m {s}s"
        if m:
            return f"{m}m {s}s"
        return f"{s}s"

    def _launch_header(self, started: datetime) -> str:
        bar = "=" * 70
        sid = self.session_id or "(new session)"
        return (
            f"\n\n{bar}\n"
            f"## LAUNCH {self.launch_count} — START "
            f"{started.isoformat(timespec='seconds')}\n"
            f"- purpose: continue optimization per {self.plan_path.name}\n"
            f"- session: {sid}\n"
            f"- model: {self.cfg['model']}  effort: {self.cfg['effort']}\n"
            f"{bar}\n\n")

    def _launch_footer(self, ended, duration_s, parser, rate_limited) -> str:
        bar = "=" * 70
        if rate_limited:
            resets_at = parser.rate_limit_resets_at()
            if resets_at:
                iso = datetime.fromtimestamp(resets_at).astimezone().isoformat(timespec="seconds")
                exit_reason = f"rate_limited (resets at {iso})"
            else:
                exit_reason = "rate_limited"
        elif parser.is_error:
            exit_reason = "error"
        else:
            exit_reason = "clean"
        return (
            f"\n\n{bar}\n"
            f"## LAUNCH {self.launch_count} — END "
            f"{ended.isoformat(timespec='seconds')}  "
            f"(duration: {self._format_duration(duration_s)})\n"
            f"- exit: {exit_reason}\n"
            f"- cost: ${parser.total_cost:.4f}  cumulative: ${self.total_cost_usd:.4f}\n"
            f"- tool calls: {parser.tool_count}\n"
            f"{bar}\n")

    # ── rate-limit backoff ──────────────────────────────────────────────────

    def _wait_rate_limit(self, parser: StreamParser, attempt: int) -> None:
        resets_at = parser.rate_limit_resets_at()
        if resets_at:
            wait = max(0, resets_at - int(time.time())) + 10
            wait = min(wait, self.cfg["rate_limit_max_wait"])
        else:
            wait = min(self.cfg["rate_limit_base_wait"] * (2 ** attempt),
                       self.cfg["rate_limit_max_wait"])
        self.logger.warning(f"Rate limited — waiting {wait}s (attempt {attempt + 1})")
        elapsed = 0
        while elapsed < wait and not self.stop_event.is_set():
            time.sleep(min(5, wait - elapsed))
            elapsed += 5

    # ── realtime steering ──────────────────────────────────────────────────────

    def _input_listener(self) -> None:
        if not sys.stdin.isatty():
            return
        while not self.stop_event.is_set():
            try:
                line = input()
            except (EOFError, KeyboardInterrupt):
                self.stop()
                break
            s = line.strip()
            if not s:
                continue
            if s.startswith("/"):
                cmd = s[1:].lower()
                if cmd in ("stop", "quit", "exit"):
                    self.stop()
                    break
                elif cmd == "pause":
                    self.pause()
                elif cmd == "resume":
                    self.unpause()
                elif cmd == "status":
                    self._print_status()
                elif cmd == "cost":
                    print(f"\033[36mTotal cost: ${self.total_cost_usd:.4f}\033[0m",
                          file=sys.stderr)
                else:
                    print(f"\033[33mUnknown command: /{cmd}\033[0m", file=sys.stderr)
            else:
                self.steer(s)
                print("\033[90m[steering queued for next launch]\033[0m", file=sys.stderr)

    def _print_status(self) -> None:
        paused = "YES" if not self.pause_event.is_set() else "no"
        print(
            f"\n\033[36m── status ─────────────────────────\033[0m\n"
            f"  Launches   : {self.launch_count}\n"
            f"  Session    : {(self.session_id or 'n/a')[:12]}…\n"
            f"  Paused     : {paused}\n"
            f"  Steering Q : {self.steering_queue.qsize()} msgs\n"
            f"  Cost       : ${self.total_cost_usd:.4f}\n"
            f"\033[36m──────────────────────────────────\033[0m",
            file=sys.stderr)

    # ── public control ──────────────────────────────────────────────────────────

    def steer(self, message: str) -> None:
        self.steering_queue.put(message)
        self.logger.info(f"Steering: {message[:80]}")

    def stop(self) -> None:
        self.stop_event.set()
        proc = self._current_proc
        if proc and proc.poll() is None:
            proc.terminate()
        self.logger.info("Stop requested")

    def pause(self) -> None:
        self.pause_event.clear()
        self.logger.info("Paused — type '/resume' to continue")

    def unpause(self) -> None:
        self.pause_event.set()
        self.logger.info("Resumed")

    # ── the infinite loop ──────────────────────────────────────────────────────

    def run(self) -> None:
        self.prepare()
        self.logger.info("=" * 60)
        self.logger.info("INFINITE OPTIMIZATION LOOP (claude_opt.py)")
        self.logger.info(f"  Config    : {self.config_path}")
        self.logger.info(f"  Plan      : {self.plan_path}")
        self.logger.info(f"  Project   : {self.project_dir}")
        self.logger.info(f"  Results   : {self.results_dir}")
        self.logger.info(f"  History   : {self.history_path}")
        self.logger.info(f"  Model     : {self.cfg['model']}  effort: {self.cfg['effort']}")
        self.logger.info(f"  Fetch     : {self.cfg['result_fetch']['mode']}")
        self.logger.info("  Commands  : type text to steer  •  /pause /resume /stop /status /cost")
        self.logger.info("=" * 60)

        threading.Thread(target=self._input_listener, daemon=True).start()
        prev_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, lambda *_: self.stop())

        try:
            while not self.stop_event.is_set():
                while not self.pause_event.is_set() and not self.stop_event.is_set():
                    time.sleep(0.5)
                if self.stop_event.is_set():
                    break

                self.launch_count += 1
                self.logger.info(
                    f"\033[1m{'─' * 22} LAUNCH {self.launch_count} {'─' * 22}\033[0m")
                prompt = self._build_launch_prompt()

                attempt = 0
                parser: Optional[StreamParser] = None
                while not self.stop_event.is_set():
                    parser, rate_limited = self._run_claude(prompt)
                    if not rate_limited:
                        break
                    self._wait_rate_limit(parser, attempt)
                    attempt += 1

                if parser is None:
                    break

                self.logger.info(
                    f"Launch {self.launch_count} ended  "
                    f"(${parser.total_cost:.4f} this launch, "
                    f"${self.total_cost_usd:.4f} total)")
                if self.stop_event.is_set():
                    break
                time.sleep(self.cfg["infinite_loop_delay"])
        except Exception as e:
            self.logger.error(f"Fatal: {e}", exc_info=True)
            raise
        finally:
            signal.signal(signal.SIGINT, prev_handler)
            self.logger.info(
                f"Loop ended after {self.launch_count} launches  "
                f"(${self.total_cost_usd:.4f} total)")


# ===========================================================================
# CLI
# ===========================================================================

def _add_config_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", "-c", default="config.json5",
                   help="Path to the JSON5 config (default: ./config.json5)")


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or (argv[0] not in ("run", "fetch", "validate", "-h", "--help")
                    and argv[0].startswith("-")):
        argv = ["run", *argv]

    parser = argparse.ArgumentParser(
        prog="claude_opt.py",
        description="Single-file, config-driven infinite optimization loop for Claude Code.")
    sub = parser.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="Run the infinite optimization loop (default).")
    _add_config_arg(p_run)
    p_fetch = sub.add_parser("fetch", help="Measure once and print the RESULT_* contract.")
    _add_config_arg(p_fetch)
    p_val = sub.add_parser("validate", help="Validate the config file and exit.")
    _add_config_arg(p_val)

    args = parser.parse_args(argv)
    cmd = args.cmd or "run"
    config_path = Path(args.config).resolve()

    if not config_path.is_file():
        print(f"error: config file not found: {config_path}", file=sys.stderr)
        return 2

    cfg = load_json5(config_path)

    if cmd == "fetch":
        try:
            return ResultFetcher(cfg).main()
        except (RuntimeError, ConfigError) as e:
            print(f"FETCH ERROR: {e}", file=sys.stderr)
            return 1

    if cmd == "validate":
        try:
            ClaudeOptLoop(cfg, config_path)  # __init__ validates
        except ConfigError as e:
            print(f"INVALID:\n{e}", file=sys.stderr)
            return 1
        print("OK: config is valid.")
        return 0

    try:
        loop = ClaudeOptLoop(cfg, config_path)
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    loop.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
