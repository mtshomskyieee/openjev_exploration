#!/usr/bin/env bash
# data-collection-harness.sh -- regenerate the CSVs in data/ by replaying every
# run, one after another, headless.
#
#   run 1  previous_unguided  initial_custom_jev_solution.py, plain board prompt
#   run 2  optimal_guided     optimal script, ChainOfThought_GUIDE (sample games)
#   run 3  logic_guide        optimal script, GUIDE (four rules)
#   run 4  decision_facts     optimal script, Pong.decision_facts() (as shipped)
#
# Each run plays the heuristic control and then the model on the same seed. The
# summary files are cumulative, like the originals: run2 = runs 1-2, run3 = runs
# 1-3, run4 = runs 1-4. The per-decision log is written for run 2.
#
# The scripts only print totals, so the game loop from their main() is replayed
# here with the same order of RNG calls, recording every decision. Nothing
# renders; progress goes to stderr.
#
# Usage:
#   ./data-collection-harness.sh                  # everything; expect about an hour on CPU
#   BACKENDS=heuristic ./data-collection-harness.sh   # quick check, no model
#
# Settings (environment variables):
#   OUT_DIR=data  FRAMES=400  DECIDE_EVERY=2  THRESHOLD=0.55  SEED=7
#   OPPONENT_SKILL=0.62  BACKENDS="heuristic llama"  MODEL_LABEL=qwen3-0.6b
#   REPO_ID=Qwen/Qwen3-0.6B-GGUF  FILENAME=Qwen3-0.6B-Q8_0.gguf
#
# Existing CSVs in OUT_DIR are copied to OUT_DIR/.bak/<timestamp>/ first.
#
# Runs 2 and 3 pair their guide with the initial script's system prompt
# ("Choose the move that best intercepts the ball..."). For run 2 this
# reproduces the original log's probabilities exactly. Run 3 has no per-decision
# log to check against, so compare its summary row. Heuristic rows ignore the
# prompt and match the originals except for latency_ms. The chose_* columns,
# blank in some original rows, are always filled here.

set -euo pipefail
cd "$(dirname "$0")"

export OUT_DIR="${OUT_DIR:-data}"
export FRAMES="${FRAMES:-400}"
export DECIDE_EVERY="${DECIDE_EVERY:-2}"
export THRESHOLD="${THRESHOLD:-0.55}"
export SEED="${SEED:-7}"
export OPPONENT_SKILL="${OPPONENT_SKILL:-0.62}"
export BACKENDS="${BACKENDS:-heuristic llama}"
export MODEL_LABEL="${MODEL_LABEL:-qwen3-0.6b}"
export REPO_ID="${REPO_ID:-Qwen/Qwen3-0.6B-GGUF}"
export FILENAME="${FILENAME:-Qwen3-0.6B-Q8_0.gguf}"

command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }

if [[ " $BACKENDS " == *" llama "* ]]; then
    uv sync --extra llama --inexact --quiet
else
    uv sync --inexact --quiet
fi

mkdir -p "$OUT_DIR"
if compgen -G "$OUT_DIR/*.csv" >/dev/null; then
    backup="$OUT_DIR/.bak/$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$backup"
    cp "$OUT_DIR"/*.csv "$backup"/
    echo "backed up existing CSVs to $backup" >&2
fi

uv run --no-sync python - <<'PY'
import csv
import importlib.util
import os
import sys
import time

OUT_DIR = os.environ["OUT_DIR"]
FRAMES = int(os.environ["FRAMES"])
DECIDE_EVERY = int(os.environ["DECIDE_EVERY"])
THRESHOLD = float(os.environ["THRESHOLD"])
SEED = int(os.environ["SEED"])
OPPONENT_SKILL = float(os.environ["OPPONENT_SKILL"])
BACKENDS = os.environ["BACKENDS"].split()
MODEL_LABEL = os.environ["MODEL_LABEL"]
REPO_ID = os.environ["REPO_ID"]
FILENAME = os.environ["FILENAME"]
WIDTH = 56  # the scripts size the board to the terminal; pin the full width

SUMMARY_COLS = ["run", "backend", "decisions", "sat_out", "sat_out_rate", "acted",
                "correct", "accuracy", "chose_up", "chose_down", "chose_stay",
                "ai_score", "opponent_score"]
LOG_COLS = ["backend", "decision", "frame", "model_move", "confidence", "p_up",
            "p_down", "p_stay", "perfect_move", "used_fallback", "played_move",
            "agreed_when_acted", "latency_ms", "ai_score", "opponent_score"]


def load(name):
    spec = importlib.util.spec_from_file_location(name, f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses look the module up while it loads
    spec.loader.exec_module(mod)
    return mod


initial = load("initial_custom_jev_solution")
optimal = load("optimal_custom_jev_solution")


def guided_prompt(mod, guide, system):
    """A _prompt() that puts a saved guide in front of the plain board text."""
    def _prompt(self, game):
        options = "\n".join(f"{lab}. {mv}" for lab, mv in zip(mod.LABELS, mod.MOVES, strict=True))
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{guide}{game.describe()}\n\n{options}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
    return _prompt


# (run label, module, prompt override or None, file stem written after this run)
RUNS = [
    ("previous_unguided", initial, None, None),
    ("optimal_guided", optimal,
     guided_prompt(optimal, optimal.LlamaChoiceBackend.ChainOfThought_GUIDE,
                   initial.LlamaChoiceBackend.SYSTEM),
     "run2_sample_games"),
    ("logic_guide", optimal,
     guided_prompt(optimal, optimal.LlamaChoiceBackend.GUIDE,
                   initial.LlamaChoiceBackend.SYSTEM),
     "run3_four_rules"),
    ("decision_facts", optimal, None, "run4_decision_facts"),
]
LOGGED_RUN = "optimal_guided"


def play(mod, backend, label):
    """The headless game loop from main(), recording each decision."""
    game = mod.Pong(width=WIDTH, seed=SEED, opponent_skill=OPPONENT_SKILL)
    rows = []
    move = "STAY"
    started = time.monotonic()
    for frame in range(FRAMES):
        if frame % DECIDE_EVERY == 0:
            truth = game.perfect_move()
            choice = backend.choose(game)
            fallback = choice.confidence < THRESHOLD
            move = game.perfect_move() if fallback else choice.value
            p = choice.probabilities
            rows.append({
                "backend": backend.name, "decision": len(rows) + 1, "frame": frame,
                "model_move": choice.value, "confidence": round(choice.confidence, 4),
                "p_up": p["UP"], "p_down": p["DOWN"], "p_stay": p["STAY"],
                "perfect_move": truth, "used_fallback": int(fallback), "played_move": move,
                "agreed_when_acted": int(not fallback and choice.value == truth),
                "latency_ms": round(choice.latency_ms, 1),
                "ai_score": game.right_score, "opponent_score": game.left_score,
            })
            n = len(rows)
            if n % 20 == 0:
                print(f"  {label:<18} {backend.name:<9} {n:>4} decisions  "
                      f"{time.monotonic() - started:6.0f}s", file=sys.stderr)
        game.apply(move, "right")
        game.step_opponent()
        game.step_ball()
    return rows, game


def summarize(run, label, rows, game):
    acted = [r for r in rows if not r["used_fallback"]]
    correct = sum(r["agreed_when_acted"] for r in acted)
    chose = {m: sum(r["model_move"] == m for r in acted) for m in ("UP", "DOWN", "STAY")}
    return {
        "run": run, "backend": label, "decisions": len(rows),
        "sat_out": len(rows) - len(acted),
        "sat_out_rate": round((len(rows) - len(acted)) / len(rows), 4),
        "acted": len(acted), "correct": correct,
        "accuracy": round(correct / len(acted), 4) if acted else "",
        "chose_up": chose["UP"], "chose_down": chose["DOWN"], "chose_stay": chose["STAY"],
        "ai_score": game.right_score, "opponent_score": game.left_score,
    }


def write(path, cols, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)  # CRLF, like the original logs
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)", file=sys.stderr)


models = {}  # one loaded Llama per module, reused across runs
summary = []
for run, mod, prompt, stem in RUNS:
    log_rows = []
    for kind in BACKENDS:
        if kind == "heuristic":
            backend, label = mod.HeuristicBackend(seed=SEED), "heuristic"
        elif kind == "llama":
            if mod.__name__ not in models:
                print(f"loading {REPO_ID}/{FILENAME} for {mod.__name__} ...", file=sys.stderr)
                models[mod.__name__] = mod.LlamaChoiceBackend(REPO_ID, FILENAME)
            backend, label = models[mod.__name__], MODEL_LABEL
            # Instance attribute, so it never leaks into another run.
            if prompt is not None:
                backend._prompt = prompt.__get__(backend)
            else:
                backend.__dict__.pop("_prompt", None)
        else:
            sys.exit(f"unknown backend {kind!r}; use heuristic and/or llama")
        rows, game = play(mod, backend, run)
        summary.append(summarize(run, label, rows, game))
        log_rows += rows
    if run == LOGGED_RUN:
        write(os.path.join(OUT_DIR, "run2_sample_games_decision_log.csv"), LOG_COLS, log_rows)
    if stem:
        write(os.path.join(OUT_DIR, f"{stem}_vs_heuristic.csv"), SUMMARY_COLS, summary)
PY
