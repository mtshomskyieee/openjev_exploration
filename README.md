# Jev on Pong: choosing instead of generating

A small, runnable experiment with the idea behind TypeSafe AI's **Jev** "System One" models: don't ask a language model to *write* an answer. Hand it a closed set of options defined in code, read the probability it assigns to each one, and let a confidence gate decide whether to trust it.

Pong is the testbed because every decision is tiny (the right paddle goes `UP`, `DOWN` or `STAY`) and because there is an exact answer key: you can calculate where the ball will cross the paddle's column and grade every move against it.

```
  Choice distribution:
    UP    12.0% ██················
    DOWN  81.3% ███████████████···  <
    STAY   6.7% █·················

  confidence 81.3%   threshold 55%
```

No text is parsed and no schema is validated. The move is one of the options you defined, by construction. When confidence falls below `--threshold`, the game ignores the model and plays a deterministic controller instead. In production, that gate is where you would escalate to a human or a bigger model.

## What's in the repo

| File | What it is | Backends | Extra needed for a model |
|---|---|---|---|
| [`initial_custom_jev_solution.py`](initial_custom_jev_solution.py) | A hand-rolled copy of the OpenJev recipe: prefill a prompt on a local GGUF model, read the logits for `A`/`B`/`C` at the last position, softmax. One forward pass, no sampling. The prompt is a plain description of the board. | `heuristic`, `llama` | `llama` |
| [`optimal_custom_jev_solution.py`](optimal_custom_jev_solution.py) | The same game and recipe. Only the prompt changes: `Pong.decision_facts()` adds the ball's speeds, the row where it meets the paddle's side, and which way that is. The earlier prompt attempts are kept in the file as `ChainOfThought_GUIDE` and `GUIDE`. | `heuristic`, `llama` | `llama` |
| [`open_jev_solution_example.py`](open_jev_solution_example.py) | Built on the real [OpenJev](https://github.com/zhangcy122/OpenJev) library (`openjevpro`): `OpenJevProClient.decide_choice()`, an explicit `UNKNOWN` option, `HybridJevGateway` + `TypeSafeJevGuardHarness` as the confidence gate, optional temperature calibration and order-invariant scoring. It starts and stops a local llama-cpp-python server for you. | `openjev`, `mock`, `heuristic` | `openjev` (needed for every backend) |
| [`data/`](data/) | Logs of the runs behind the results below. See [`data/README.md`](data/README.md). | | |
| [`data-collection-harness.sh`](data-collection-harness.sh) | Replays every run headless, one after another, and regenerates the CSVs in `data/`. `BACKENDS=heuristic` gives a quick check with no model. | `heuristic`, `llama` | `llama` (installed for you) |

The `heuristic` backend has no model. It scores the moves geometrically and softmaxes, which gives you a control group and lets the scripts run anywhere.

## Requirements

- [uv](https://docs.astral.sh/uv/getting-started/installation/). It installs Python 3.12 for you if needed.
- For any model backend: a C/C++ compiler and CMake, because `llama-cpp-python` builds from source on first install (about 2–3 minutes on an 8-core CPU).
- A one-time download of about 600 MB (Qwen3-0.6B, Q8 GGUF) into the Hugging Face cache on first model run.
- No GPU needed. Every result below came from an 8-core CPU.

## Quick start

```bash
git clone https://github.com/mtshomskyieee/openjev_exploration.git
cd openjev_exploration

uv sync                                              # numpy only
uv run python initial_custom_jev_solution.py         # heuristic backend, animated in the terminal (Ctrl-C to stop)
```

### With a model

Install the extra you need. `uv sync` makes the environment match exactly what you ask for, so use `--all-extras` to keep both:

```bash
uv sync --all-extras                                 # or: --extra llama / --extra openjev

# The custom recipe (initial and optimal prompts)
uv run python initial_custom_jev_solution.py --backend llama
uv run python optimal_custom_jev_solution.py --backend llama

# The real OpenJev library (--backend auto picks openjev once llama-cpp-python is installed)
uv run python open_jev_solution_example.py
```

### Running a single file without cloning

Each script also has a [PEP 723](https://peps.python.org/pep-0723/) header listing its own dependencies, so any one file runs on its own:

```bash
uv run initial_custom_jev_solution.py --backend llama
```

Note the difference inside the repo: `uv run python file.py` uses the project environment from `uv sync`, while `uv run file.py` reads the script's header and builds a separate cached environment for it. Both work. The first shares one install across all three scripts.

### Headless, for numbers instead of animation

```bash
uv run python initial_custom_jev_solution.py --backend llama --frames 400 --no-render
```

```
backend        llama
frames         400
decisions      200
agreement      23/90 (25.6%)
fallbacks      110 (55.0%)
mean latency   1008.7 ms
final score    AI 0 – 0 opponent
```

`agreement` counts how often the move the model played matched the answer key, **only on decisions where it was confident enough to act**. `fallbacks` (`escalated` in the OpenJev script) counts decisions below the gate, which the closed-form controller played instead. Those are left out of `agreement`.

## Results

Same game, seed 7, 55% gate, Qwen3-0.6B on an 8-core CPU with no GPU. Hand-rolled rows come from the two custom scripts; per-run data is in [`data/`](data/README.md).

| Method | Decisions | Sat out / escalated | Right when it acted | Mean latency |
|---|---|---|---|---|
| Heuristic baseline (circular, see caveats) | 200 | 3 (1.5%) | 197/197 (100%) | ~0 ms |
| Hand-rolled recipe, plain prompt | 200 | 110 (55%) | 23/90 (25.6%) | 1009 ms |
| Hand-rolled, five sample games in prompt | 200 | 92 (46%) | 28/108 (25.9%) | ~8700 ms |
| Hand-rolled, four rules in prompt | 200 | 3 (1.5%) | 4/197 (2.0%) | not recorded |
| Hand-rolled, meeting row written out (`decision_facts`) | 200 | 41 (20.5%) | 159/159 (100%) | not recorded |
| OpenJev, single pass | 200 | 200 (100%) | n/a | 2150 ms |
| OpenJev, `--calibrate 40` | 30 | 30 (100%) | n/a | 2260 ms |
| OpenJev, `--order-invariant` | 20 | 20 (100%) | n/a | 3918 ms |
| OpenJev, `--order-invariant --threshold 0.3` | 20 | 0 (0%) | 10/20 (50%) | 4006 ms |
| OpenJev mock client (no model) | 100 | 0 (0%) | 19/100 (19%) | ~0 ms |

The OpenJev rows after the first use 20–30 decisions. Treat them as indications, not benchmarks.

What the runs showed:

- **The shape of the answer was never the hard part.** A typed move and a probability came back on every run, including the one that played `DOWN` 197 times.
- **The prompt has to contain the fact the choice depends on.** "Moving toward you and downward" hides two different correct moves. Sample games pulled the model to always choose `UP`, and rules pulled it to always choose `DOWN`. Only stating the meeting row made it accurate. That 100% is the model reading a stated answer, not doing the geometry.
- **OpenJev turned "confidently wrong" into "abstains", but part of that was position bias.** In a single pass, `UNKNOWN` (the last letter) took 61–83% of the probability. Scored order-invariantly, it came *last* every time, and the three moves came back nearly flat.
- **Calibration measured the problem instead of hiding it.** Fitting on 40 graded decisions pushed the temperature to OpenJev's upper bound (T = 10), which means the scores are close to noise. With position bias removed and a 0.3 gate, the model beat chance a little (50% vs 33%).
- **The library makes the probability trustworthy. It doesn't supply missing information.** It also costs latency: a single pass took about 2× the hand-rolled recipe, and order invariance about 4×.

## Options

Shared by all three scripts:

| Flag | Default | Notes |
|---|---|---|
| `--frames N` | `0` | `0` runs until Ctrl-C |
| `--fps` | `18` | Render rate |
| `--decide-every N` | `2` | Frames between decisions; raise it for slow backends |
| `--threshold` | `0.55` | Below this confidence, use the fallback controller |
| `--opponent-skill` | `0.62` | 0–1 chance the left paddle reacts on a given tick |
| `--seed` | `7` | Deterministic serves and heuristic noise |
| `--repo-id`, `--filename` | `Qwen/Qwen3-0.6B-GGUF`, `Qwen3-0.6B-Q8_0.gguf` | Any GGUF model on Hugging Face |
| `--no-render` | off | Headless; implies a stats summary |
| `--stats` | off | Print the summary even when rendering |

Custom scripts: `--backend {auto,llama,heuristic}`. `auto` uses `llama` if it can be imported, otherwise `heuristic`.

`open_jev_solution_example.py` only:

| Flag | Default | Notes |
|---|---|---|
| `--backend {auto,openjev,mock,heuristic}` | `auto` | `mock` is OpenJev's keyword-matching CI stub. It doesn't understand Pong. |
| `--base-url` | none | Use an existing OpenAI-compatible server (vLLM, SGLang, llama.cpp) instead of launching one |
| `--model` | `qwen3-0.6b` | Model name to send to that server |
| `--order-invariant` | off | Score each option in its own isolated YES/NO pass (one request per option) |
| `--calibrate N` | `0` | Fit the softmax temperature on N warm-up decisions graded against perfect play |

Lower `--threshold` to watch the model make more of its own mistakes. Raise it to watch the fallback take over.

## Reproducing the results

```bash
# Hand-rolled recipe
uv run python initial_custom_jev_solution.py --backend llama --frames 400 --no-render
uv run python optimal_custom_jev_solution.py --backend llama --frames 400 --no-render     # decision_facts prompt

# OpenJev library
uv run python open_jev_solution_example.py --frames 400 --no-render                        # single pass
uv run python open_jev_solution_example.py --calibrate 40 --frames 60 --no-render
uv run python open_jev_solution_example.py --order-invariant --threshold 0.3 --frames 40 --no-render
uv run python open_jev_solution_example.py --backend mock --frames 200 --no-render
```

To regenerate every CSV in `data/` in one go (about an hour on CPU), run `./data-collection-harness.sh`.

To run the sample-games or four-rules prompt by hand, change the prompt in `LlamaChoiceBackend._prompt()` in `optimal_custom_jev_solution.py` to use `ChainOfThought_GUIDE` or `GUIDE`.

## Caveats

- **The heuristic's 100% is circular.** It scores moves with the same intercept calculation used to grade them. It is a control, not a result.
- **Pong is a bad job for a language model, on purpose.** The closed-form fallback beats any model on accuracy and latency. The point is the *interface*: typed options in, calibrated probabilities out, confidence deciding whether to trust the answer. That interface carries over to problems with no closed form, such as ticket routing or classification.
- **Small samples.** One seed, one machine, 20–200 decisions per row.

## Notes and troubleshooting

- **`openjevpro` is pinned to a git commit** (`68561c3f`). The PyPI release, 0.2.0, lags the repository and has no mock client, order invariance or chat mode.
- **Qwen3 needs chat mode with thinking off.** With raw completions, Qwen3's top-20 tokens were mostly whitespace. The OpenJev script sets `enable_thinking: false` on the server, because the server ignores the client's own setting for it.
- **The `HOST` environment variable.** llama-cpp-python's server reads `HOST`/`PORT` ahead of its command-line flags, and some shells export `HOST` as the machine name, which would bind the server to your LAN. The script pins `127.0.0.1`.
- **If the server fails to start,** the error message points to its log file (`llama_server_*.log` in your temp directory).
- **Slow frame rate with a model?** Raise `--decide-every` (for example, `--decide-every 4 --fps 12`).

## References

- TypeSafe AI, "Introducing System One Models & Jev": https://typesafe.ai/blog/introducing-system-one-models-and-jev
- "Jev (AI model)," Wikipedia: https://en.wikipedia.org/wiki/Jev_(AI_model)
- OpenJev: https://github.com/zhangcy122/OpenJev
- openjev-sglang: https://github.com/ipconfiger/openjev-sglang
- Qwen3-0.6B GGUF: https://huggingface.co/Qwen/Qwen3-0.6B-GGUF

## License

[MIT](LICENSE) for the code in this repository.

The `openjevpro` dependency, used only by `open_jev_solution_example.py`, is licensed separately under the **PolyForm Noncommercial** license. Check its terms before any commercial use.
