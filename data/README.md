# Data

These files are logs from exploratory runs, not a curated benchmark. They were written while working out whether each change to the prompt helped. Read them as a lab notebook: the rows are kept as they came out of the runs, and only the formatting was fixed (see [Changes from the raw logs](#changes-from-the-raw-logs)).

## How the runs were made

Every run used the same setup:

| Setting | Value |
|---|---|
| Game | 400 frames, one decision every 2 frames → **200 decisions** per backend |
| Confidence gate | `--threshold 0.55` (below it, the closed-form controller plays and the decision is excluded from accuracy) |
| Seed | `7` (ball serves and heuristic noise are identical across runs) |
| Model | Qwen3-0.6B, `Qwen3-0.6B-Q8_0.gguf`, via `llama-cpp-python` |
| Hardware | 8-core CPU, no GPU |
| Opponent | `--opponent-skill 0.62` |

Each run pairs the model with the `heuristic` backend playing its own game with the same seed, as a control.

The scripts themselves only print totals. The original files were assembled from exploratory runs.

### Regenerating

[`../data-collection-harness.sh`](../data-collection-harness.sh) replays all four runs one after another, headless, and rewrites every CSV here. Before it writes, it copies the existing files to `data/.bak/<timestamp>/`.

```bash
./data-collection-harness.sh                                    # all runs, heuristic + model: about an hour on CPU
BACKENDS=heuristic ./data-collection-harness.sh                 # control only, seconds
OUT_DIR=/tmp/regen ./data-collection-harness.sh                 # write somewhere else
```

Other settings are environment variables: `FRAMES`, `DECIDE_EVERY`, `THRESHOLD`, `SEED`, `OPPONENT_SKILL`, `REPO_ID`, `FILENAME`, `MODEL_LABEL`. The header of the script lists them all. How a regenerated file compares with the originals:

- **Heuristic rows:** identical, except `latency_ms`, which is wall-clock time.
- **Run 2 model rows:** probabilities identical to the original log (spot-checked).
- **Run 3:** has no per-decision log to check against.
- **`chose_*`:** always filled in, where some original rows left it blank.
- **The model runs** are deterministic in principle, but different llama.cpp builds or CPUs can shift logits slightly.

## The runs

The files are numbered by run. Run 1, the plain prompt, has no file of its own; its rows appear as `previous_unguided` in every summary.

| `run` label | Run | Prompt the model saw | Script |
|---|---|---|---|
| `previous_unguided` | 1 | The board in words: rounded ball column/row, "toward/away", "upward/downward", paddle rows | `initial_custom_jev_solution.py` |
| `optimal_guided` | 2 | Five finished sample games (board → correct letter) before the live board | `optimal_custom_jev_solution.py`, `ChainOfThought_GUIDE` |
| `logic_guide` | 3 | Four ordered rules, no samples, before the live board | `optimal_custom_jev_solution.py`, `GUIDE` |
| `decision_facts` | 4 | Board plus speeds, the row where the ball meets the right edge, the paddle centre, and one sentence saying which way that is | `optimal_custom_jev_solution.py`, `Pong.decision_facts()` |

`optimal_custom_jev_solution.py` as shipped runs **run 4**. The run 2 and run 3 prompts are kept in the file as `ChainOfThought_GUIDE` and `GUIDE`, but no flag switches to them. To rerun them, swap the prompt in `LlamaChoiceBackend._prompt()`.

## Files

### Summary tables (one row per run × backend)

Each summary file is **cumulative**: it repeats the rows of every earlier run, so you can compare the newest run with the others in a single file. Each one is the comparison table as it stood after that run.

| File | Rows | Comparison |
|---|---|---|
| `run2_sample_games_vs_heuristic.csv` | 4 | Plain prompt vs. sample games |
| `run3_four_rules_vs_heuristic.csv` | 6 | … plus four rules |
| `run4_decision_facts_vs_heuristic.csv` | 8 | … plus the meeting row written out, which is the complete picture |

Columns:

| Column | Meaning |
|---|---|
| `run` | Run label (see above) |
| `backend` | `heuristic` (no model, control) or `qwen3-0.6b` |
| `decisions` | Decisions made (200) |
| `sat_out` | Decisions below the confidence gate; the fallback controller played these |
| `sat_out_rate` | `sat_out / decisions` |
| `acted` | `decisions − sat_out`, the decisions the backend's own answer was played |
| `correct` | Of `acted`, how many matched the perfect-play move |
| `accuracy` | `correct / acted` |
| `chose_up`, `chose_down`, `chose_stay` | Distribution of moves on acted decisions. **Blank = not recorded** for that run (the tally was added later) |
| `ai_score`, `opponent_score` | Final score, right paddle (AI) vs. left paddle |

### Per-decision log

`run2_sample_games_decision_log.csv` has 400 rows (200 `heuristic` + 200 `llama`) from **run 2** (sample games). It is the detail behind the `optimal_guided` rows above: 92 sat out, 28 of 108 correct. On the 108 decisions it acted on, the model chose UP every time.

| Column | Meaning |
|---|---|
| `backend` | `heuristic` or `llama` (the same model as `qwen3-0.6b` in the summaries; the name differs because this is what the script calls its backend) |
| `decision` | 1-based decision index |
| `frame` | Game frame the decision was made on |
| `model_move` | The backend's top choice |
| `confidence` | Probability of `model_move` |
| `p_up`, `p_down`, `p_stay` | Full distribution over the three options |
| `perfect_move` | Closed-form correct move (the answer key) |
| `used_fallback` | `1` if confidence < 0.55 and the fallback played |
| `played_move` | What the paddle actually did |
| `agreed_when_acted` | `1` if the backend acted and `model_move == perfect_move`; `0` otherwise, including every fallback row |
| `latency_ms` | Time for the decision. Mean ≈ 8.7 s for `llama` here, because the five sample games make the prompt long |
| `ai_score`, `opponent_score` | Running score at that decision |

## Caveats

- **The heuristic's 100% is circular.** It scores moves with the same intercept geometry the grader uses. It is a control, not a result.
- **200 decisions, one seed, one machine.** The numbers can separate "broken" from "working" but can't rank close variants.
- **Run 4's 100% is the prompt stating the answer.** `decision_facts()` ends with "Move up/down/Stay". It shows that the model reads a stated fact reliably. It does not show that the model can work out the geometry.
- The OpenJev-library results (`open_jev_solution_example.py`) have no CSV. They are summarized in the top-level README.

## Changes from the raw logs

- Files renamed, from `optimal_jev_decisions.csv`, `optimal_jev_vs_heuristic.csv`, `optimial_jev_logic_vs_heuristic.csv`, and `optimial_jev2_logic_vs_heuristic.csv` respectively.
- In `run2_sample_games_vs_heuristic.csv`, three rows had one extra empty field (14 fields against a 13-column header). The extra field was removed, so those rows now match their copies in the run 3 and run 4 files exactly. No values changed.
