#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "openjevpro @ git+https://github.com/zhangcy122/OpenJev@68561c3f4cb86ba9aafb16f336e02b52ab1b3212",
#   "llama-cpp-python[server]",
#   "huggingface-hub",
# ]
# ///
"""
open_jev_solution_example.py -- OpenJev (openjevpro) playing Pong.

This is a worked example of the pattern described in the README, built on
the OpenJev library itself rather than a hand-rolled copy of its recipe. Instead
of asking a language model to *generate* anything, we give it a closed set of
options and read the probability it assigns to each one. That is the Choice
primitive, and it is all this file needs to play a game.

Every few frames the Pong loop hands OpenJev a structured `state` dict and the
candidates UP / DOWN / STAY / UNKNOWN. What comes back is an openjevpro
ChoiceDecision -- a typed value plus a calibrated distribution, never a string
to parse. The OpenJev pieces in play:

  OpenJevProClient      decide_choice() against an OpenAI-compatible server.
                        One max_tokens=1 request, the letter logprobs read off
                        the top-20, softmaxed by a TemperatureCalibrator.
                        --order-invariant switches to OpenJev's isolated
                        per-candidate YES/NO scoring (one pass per option).

  TemperatureCalibrator --calibrate N fits the softmax temperature by NLL on N
                        warm-up decisions graded against perfect play. This is
                        post-hoc calibration: it can make confidence honest, it
                        cannot make a wrong argmax right.

  HybridJevGateway      The confidence gate. A TypeSafeJevGuardHarness applies
                        tau = max(--threshold, 1.25 / K); confident decisions
                        run on the local model (Tier1_Local), everything else --
                        including an explicit UNKNOWN -- escalates to Tier 2.
                        Here Tier 2 is the closed-form controller, standing in
                        for the human or larger model you would use in
                        production.

Backends, all behind the same decide_choice() interface:

  openjev    OpenJevProClient against a local llama-cpp-python server running
             Qwen3-0.6B (launched and torn down for you), or any vLLM / SGLang /
             llama.cpp endpoint via --base-url.
  mock       OpenJevProClient(mock=True), OpenJev's own zero-GPU MockClient.
             It is a keyword-matching CI stub and does not understand Pong --
             useful for seeing the offline path, not for numbers.
  heuristic  No model. Scores the moves geometrically and softmaxes through
             OpenJev's calibrator. A baseline, circular by construction.

The game tracks how often the model agrees with the perfect-play move when it
is confident enough to act -- a crude local stand-in for the kind of decision
benchmark JevBench measures.

Usage:
    uv run open_jev_solution_example.py                           # Qwen3-0.6B via OpenJev, animated
    uv run open_jev_solution_example.py --calibrate 40            # fit temperature first
    uv run open_jev_solution_example.py --frames 400 --no-render  # headless stats
    uv run open_jev_solution_example.py --backend heuristic       # no model download
    uv run open_jev_solution_example.py --base-url http://localhost:8000/v1 --model Qwen/Qwen3-4B
"""

from __future__ import annotations

import argparse
import atexit
import enum
import os
import random
import shutil
import socket
import subprocess
import sys
import tempfile
import time

try:
    from openjevpro import (
        ChoiceDecision,
        HybridJevGateway,
        OpenJevProClient,
        TemperatureCalibrator,
        TypeSafeJevGuardHarness,
    )
except ImportError:  # pragma: no cover - depends on environment
    sys.exit("openjevpro is not installed. Run with `uv run open_jev_solution_example.py`, "
             "which installs it from the script header.")

# --------------------------------------------------------------------------
# The schema
# --------------------------------------------------------------------------


class Move(str, enum.Enum):
    """The closed option set. Defining it here -- in code, not in a prompt --
    is the part of the pattern that pays off regardless of which backend wins."""

    UP = "UP"
    DOWN = "DOWN"
    STAY = "STAY"


# UNKNOWN is passed explicitly rather than left for the client to append, so the
# gateway's guard keeps its probability mass and can abstain on it directly.
CANDIDATES = [m.value for m in Move] + ["UNKNOWN"]

# Per-candidate criteria. OpenJev renders these into the prompt as a rule list.
CRITERIA = {
    "UP": "The ball will reach the right edge above the paddle's centre, "
          "so move the paddle toward row 0.",
    "DOWN": "The ball will reach the right edge below the paddle's centre, "
            "so move the paddle toward the bottom row.",
    "STAY": "The ball will arrive within the rows the paddle already covers, "
            "so hold position.",
    "UNKNOWN": "The state is ambiguous and no move is clearly best.",
}


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------


class OpenJevEngine:
    """Thin adapter over OpenJevProClient.

    HybridJevGateway calls decide_choice() without order_invariant, so the flag
    is forwarded from here. The adapter also keeps the last full ChoiceDecision
    and its latency for the display.
    """

    def __init__(self, client: OpenJevProClient, name: str,
                 order_invariant: bool = False) -> None:
        self.client = client
        self.name = name
        self.order_invariant = order_invariant
        self.last: ChoiceDecision | None = None
        self.latency_ms = 0.0

    @property
    def calibrator(self) -> TemperatureCalibrator:
        # In mock mode the client delegates to a MockClient with its own calibrator.
        mock = getattr(self.client, "_mock_client", None)
        return mock.calibrator if mock is not None else self.client.calibrator

    def decide_choice(self, state, candidates, criteria="", allow_abstain=True) -> ChoiceDecision:
        start = time.perf_counter()
        self.last = self.client.decide_choice(
            state=state,
            candidates=candidates,
            criteria=criteria,
            allow_abstain=allow_abstain,
            order_invariant=self.order_invariant,
        )
        self.latency_ms = (time.perf_counter() - start) * 1000
        return self.last


class HeuristicEngine:
    """Geometric scoring, no model. Same decide_choice() shape as OpenJev.

    `noise` blunts it deliberately so the confidence gate and the escalation
    path actually get exercised in a demo run -- a perfect controller would
    never dip below threshold and you would never see the routing work.
    """

    name = "heuristic"

    def __init__(self, game: "Pong", noise: float = 0.35, seed: int | None = None) -> None:
        self.game = game
        self.noise = noise
        self.rng = random.Random(seed)
        self.calibrator = TemperatureCalibrator(temperature=1.0)
        self.last: ChoiceDecision | None = None
        self.latency_ms = 0.0

    def decide_choice(self, state, candidates, criteria="", allow_abstain=True) -> ChoiceDecision:
        start = time.perf_counter()
        game = self.game
        delta = game.predicted_intercept_y() - (game.right_y + game.paddle_h / 2)
        # Sharpness scales with how far off-centre we are: a big gap is an easy
        # call, a small one is genuinely ambiguous and should score as such.
        scale = min(abs(delta), 6.0)
        logits = {
            "UP": scale if delta < -0.5 else -scale,
            "DOWN": scale if delta > 0.5 else -scale,
            "STAY": 2.0 if abs(delta) <= 0.5 else -scale,
            "UNKNOWN": -3.0,
        }
        logits = {k: v + self.rng.gauss(0, self.noise) for k, v in logits.items()
                  if k in candidates}
        probs = self.calibrator.calibrate(logits)
        best = max(probs, key=probs.get)
        self.last = ChoiceDecision(value=best, probabilities=probs,
                                   confidence=probs[best], raw_logits=logits)
        self.latency_ms = (time.perf_counter() - start) * 1000
        return self.last


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def launch_llama_server(repo_id: str, filename: str, alias: str,
                        timeout_s: float = 180.0) -> str:
    """Start llama-cpp-python's OpenAI-compatible server and return its base URL.

    OpenJev is a client, not a runtime: it needs an endpoint that returns
    logprobs. On a CPU-only box this is the lightest one available.
    """
    import requests
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    port = free_port()
    log = tempfile.NamedTemporaryFile(prefix="llama_server_", suffix=".log", delete=False)
    # The server's settings read HOST/PORT from the environment ahead of the
    # command line, and many shells export HOST as the machine's hostname --
    # which would bind the server to the LAN. Pin both explicitly.
    env = {**os.environ, "HOST": "127.0.0.1", "PORT": str(port)}
    # enable_thinking=false makes Qwen3's chat template emit an empty
    # <think></think> block, so the first generated token is the answer letter.
    # (OpenJev's own chat_template_kwargs travels as a literal `extra_body` key
    # that this server ignores, so it has to be set server-side.)
    proc = subprocess.Popen(
        [sys.executable, "-m", "llama_cpp.server", "--model", model_path,
         "--model_alias", alias, "--host", "127.0.0.1", "--port", str(port),
         "--n_ctx", "2048", "--chat_template_kwargs", '{"enable_thinking": false}'],
        stdout=log, stderr=subprocess.STDOUT, env=env,
    )

    def stop() -> None:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    atexit.register(stop)
    base_url = f"http://127.0.0.1:{port}/v1"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"llama server exited early; see {log.name}")
        try:
            if requests.get(f"{base_url}/models", timeout=1).ok:
                return base_url
        except requests.RequestException:
            pass
        time.sleep(0.5)
    stop()
    raise RuntimeError(f"llama server did not come up in {timeout_s:.0f}s; see {log.name}")


# --------------------------------------------------------------------------
# Pong
# --------------------------------------------------------------------------


class Pong:
    """A deliberately small game. The AI drives the right paddle only."""

    def __init__(self, width: int = 56, height: int = 18, paddle_h: int = 4,
                 seed: int | None = None, opponent_skill: float = 0.62) -> None:
        self.w, self.h, self.paddle_h = width, height, paddle_h
        self.opponent_skill = opponent_skill
        self.rng = random.Random(seed)
        self.left_score = self.right_score = 0
        self.left_y = self.right_y = (height - paddle_h) / 2
        self.serve(direction=1)

    def serve(self, direction: int) -> None:
        self.ball_x = self.w / 2
        self.ball_y = self.h / 2
        self.ball_dx = 0.9 * direction
        self.ball_dy = self.rng.choice([-0.55, -0.3, 0.3, 0.55])

    def predicted_intercept_y(self) -> float:
        """Where the ball will cross the right wall, reflecting off top/bottom.

        This is the ground truth the model is judged against. It is also what
        the escalation controller uses -- worth noting that for Pong specifically
        the closed-form answer is better than any model, which is exactly why
        this is a demo of the interface and not an argument for using an LLM here.
        """
        if self.ball_dx <= 0:
            return self.h / 2  # ball receding; hold centre
        steps = (self.w - 2 - self.ball_x) / self.ball_dx
        y = self.ball_y + self.ball_dy * steps
        span = 2 * (self.h - 1)
        y = abs(y) % span
        return span - y if y > self.h - 1 else y

    def perfect_move(self) -> str:
        delta = self.predicted_intercept_y() - (self.right_y + self.paddle_h / 2)
        if delta < -0.5:
            return "UP"
        if delta > 0.5:
            return "DOWN"
        return "STAY"

    def state(self) -> dict:
        """The `state` OpenJev decides over. Deliberately omits the intercept --
        that is the answer, and working it out is the model's job. OpenJev puts
        the state first in its prompt, so it is kept compact: every token here
        is re-prefilled on every decision."""
        return {
            "board": {"width": self.w, "height": self.h, "row_0": "top"},
            "ball": {
                "column": round(self.ball_x),
                "row": round(self.ball_y),
                "horizontal": "toward you" if self.ball_dx > 0 else "away from you",
                "vertical": "upward" if self.ball_dy < 0 else "downward",
            },
            "your_paddle": {
                "edge": "right",
                "top_row": round(self.right_y),
                "bottom_row": round(self.right_y + self.paddle_h),
            },
        }

    def apply(self, move: str, paddle: str) -> None:
        step = 1.0 if move == "DOWN" else -1.0 if move == "UP" else 0.0
        attr = f"{paddle}_y"
        setattr(self, attr, min(max(getattr(self, attr) + step, 0), self.h - self.paddle_h))

    def step_opponent(self) -> None:
        """Left paddle: tracks the ball's current row, not its intercept, and
        only reacts on a fraction of ticks. Both limits are deliberate -- a
        paddle that moves every frame toward a closed-form intercept is
        unbeatable, and a 0-0 demo shows nothing."""
        centre = self.left_y + self.paddle_h / 2
        if (self.ball_dx < 0 and abs(self.ball_y - centre) > 0.8
                and self.rng.random() < self.opponent_skill):
            self.apply("DOWN" if self.ball_y > centre else "UP", "left")

    def step_ball(self) -> str | None:
        """Advance one tick. Returns 'left'/'right' if that side just scored."""
        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy
        if self.ball_y <= 0 or self.ball_y >= self.h - 1:
            self.ball_dy *= -1
            self.ball_y = min(max(self.ball_y, 0), self.h - 1)

        if self.ball_x <= 1 and self.ball_dx < 0:
            if self.left_y - 0.5 <= self.ball_y <= self.left_y + self.paddle_h + 0.5:
                self.ball_dx *= -1
                self.ball_dy += (self.ball_y - (self.left_y + self.paddle_h / 2)) * 0.12
            else:
                self.right_score += 1
                self.serve(direction=-1)
                return "right"

        if self.ball_x >= self.w - 2 and self.ball_dx > 0:
            if self.right_y - 0.5 <= self.ball_y <= self.right_y + self.paddle_h + 0.5:
                self.ball_dx *= -1
                self.ball_dy += (self.ball_y - (self.right_y + self.paddle_h / 2)) * 0.12
            else:
                self.left_score += 1
                self.serve(direction=1)
                return "left"
        return None

    def render(self) -> str:
        grid = [[" "] * self.w for _ in range(self.h)]
        for r in range(self.h):
            if r % 2 == 0:
                grid[r][self.w // 2] = "│"
        for i in range(self.paddle_h):
            grid[min(int(self.left_y) + i, self.h - 1)][1] = "█"
            grid[min(int(self.right_y) + i, self.h - 1)][self.w - 2] = "█"
        grid[min(max(int(round(self.ball_y)), 0), self.h - 1)][
            min(max(int(round(self.ball_x)), 0), self.w - 1)
        ] = "●"
        top = "┌" + "─" * self.w + "┐"
        bottom = "└" + "─" * self.w + "┘"
        body = "\n".join("│" + "".join(row) + "│" for row in grid)
        return f"{top}\n{body}\n{bottom}"


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


def calibrate(engine, n: int, decide_every: int, seed: int, width: int) -> float:
    """Fit the engine's softmax temperature on n decisions graded by perfect play.

    A scratch game with its own seed keeps the warm-up states out of the match.
    The paddle follows perfect play so the states look like real rallies.
    """
    game = Pong(width=width, seed=seed + 1000)
    if isinstance(engine, HeuristicEngine):
        engine.game = game
    logits, targets = [], []
    frame = 0
    while len(targets) < n:
        if frame % decide_every == 0:
            decision = engine.decide_choice(game.state(), CANDIDATES, CRITERIA)
            logits.append(decision.raw_logits)
            targets.append(game.perfect_move())
            print(f"\r  calibrating {len(targets)}/{n}", end="", file=sys.stderr)
        game.apply(game.perfect_move(), "right")
        game.step_opponent()
        game.step_ball()
        frame += 1
    print(file=sys.stderr)
    return engine.calibrator.fit(logits, targets)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def bar(p: float, width: int = 18) -> str:
    filled = int(round(p * width))
    return "█" * filled + "·" * (width - filled)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--backend", choices=["auto", "openjev", "mock", "heuristic"], default="auto",
                    help="auto uses openjev if llama-cpp-python or --base-url is available")
    ap.add_argument("--base-url", default=None,
                    help="OpenAI-compatible endpoint with logprobs; omit to launch a local "
                         "llama server. For Qwen3, disable thinking server-side.")
    ap.add_argument("--model", default="qwen3-0.6b",
                    help="model name sent to the endpoint")
    ap.add_argument("--repo-id", default="Qwen/Qwen3-0.6B-GGUF")
    ap.add_argument("--filename", default="Qwen3-0.6B-Q8_0.gguf")
    ap.add_argument("--order-invariant", action="store_true",
                    help="score each candidate in an isolated pass (one request per option)")
    ap.add_argument("--calibrate", type=int, default=0, metavar="N",
                    help="fit the softmax temperature on N warm-up decisions first")
    ap.add_argument("--frames", type=int, default=0, help="0 runs until Ctrl-C")
    ap.add_argument("--fps", type=float, default=18.0)
    ap.add_argument("--decide-every", type=int, default=2,
                    help="frames between decisions; raise this for slow backends")
    ap.add_argument("--threshold", type=float, default=0.55,
                    help="guard min_confidence; below tau the decision escalates")
    ap.add_argument("--opponent-skill", type=float, default=0.62,
                    help="0..1 chance the opponent paddle reacts on a given tick")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--stats", action="store_true", help="print a summary on exit")
    args = ap.parse_args(argv)

    backend_kind = args.backend
    if backend_kind == "auto":
        try:
            if args.base_url is None:
                import llama_cpp.server  # noqa: F401
            backend_kind = "openjev"
        except ImportError:
            backend_kind = "heuristic"
            print("llama-cpp-python[server] not found; using the heuristic backend.",
                  file=sys.stderr)

    cols = shutil.get_terminal_size((80, 24)).columns
    width = min(56, max(32, cols - 6))
    game = Pong(width=width, seed=args.seed, opponent_skill=args.opponent_skill)

    if backend_kind == "openjev":
        base_url = args.base_url
        if base_url is None:
            print(f"Launching llama server for {args.repo_id}/{args.filename} ...",
                  file=sys.stderr)
            try:
                base_url = launch_llama_server(args.repo_id, args.filename, args.model)
            except RuntimeError as exc:
                print(exc, file=sys.stderr)
                return 1
        # Chat mode: without the chat template a small model's top-20 tokens
        # are mostly whitespace and the option letters fall off the list.
        # abstain_threshold=0 leaves every abstention decision to the gateway.
        client = OpenJevProClient(base_url=base_url, model=args.model, backend="openai",
                                  use_chat=True, temperature_scaling=1.0,
                                  abstain_threshold=0.0)
        engine = OpenJevEngine(client, f"openjev ({args.model})", args.order_invariant)
    elif backend_kind == "mock":
        client = OpenJevProClient(mock=True, temperature_scaling=1.0, abstain_threshold=0.0)
        engine = OpenJevEngine(client, "openjev mock", args.order_invariant)
    else:
        engine = HeuristicEngine(game, seed=args.seed)

    if args.calibrate:
        temperature = calibrate(engine, args.calibrate, args.decide_every, args.seed, width)
        print(f"  fitted temperature T = {temperature}", file=sys.stderr)
        if isinstance(engine, HeuristicEngine):
            engine.game = game
    temperature = engine.calibrator.temperature

    # The confidence gate, as OpenJev ships it. The guard's own calibrator is
    # pinned to T=1 so the engine's (possibly fitted) temperature is the only
    # scaling applied. Tier 2 is the closed-form controller.
    guard = TypeSafeJevGuardHarness(calibrator=TemperatureCalibrator(temperature=1.0),
                                    min_confidence=args.threshold, alpha=1.25)
    gateway = HybridJevGateway(
        local_engine=engine,
        cloud_engine=lambda state, candidates: (game.perfect_move(), 1.0),
        guard=guard,
        min_confidence=args.threshold,
    )
    tau = guard.compute_threshold(len(CANDIDATES))

    decisions = agreements = escalations = 0
    acting_confidence = 0.0
    latencies: list[float] = []
    probs = {c: 1 / len(CANDIDATES) for c in CANDIDATES}
    shown, confidence, routed = "STAY", 1 / len(CANDIDATES), "LOCAL"
    move = "STAY"
    frame = 0

    if not args.no_render:
        sys.stdout.write("\x1b[2J\x1b[?25l")  # clear, hide cursor
    try:
        while args.frames == 0 or frame < args.frames:
            if frame % args.decide_every == 0:
                truth = game.perfect_move()
                decision = gateway.decide_choice(game.state(), CANDIDATES, criteria=CRITERIA)
                decisions += 1
                latencies.append(engine.latency_ms)
                move = decision.choice
                probs = engine.last.probabilities
                shown = max(probs, key=probs.get)
                confidence = probs[shown]
                if decision.is_local:
                    routed = "LOCAL"
                    acting_confidence += confidence
                    if move == truth:
                        agreements += 1
                else:
                    escalations += 1
                    reason = "UNKNOWN" if shown == "UNKNOWN" else "below tau"
                    routed = f"ESCALATED ({reason}) -> closed-form"

            game.apply(move, "right")
            game.step_opponent()
            game.step_ball()

            if not args.no_render:
                graded = decisions - escalations
                acc = agreements / graded if graded else 0.0
                lat = sum(latencies) / len(latencies) if latencies else 0.0
                panel = [
                    f"  OpenJev Pong — backend: {engine.name}"
                    + ("  [order-invariant]" if args.order_invariant else ""),
                    "",
                    f"  {'AI':>4} {game.right_score:>3}     opponent {game.left_score:<3}",
                    "",
                    "  ChoiceDecision probabilities:",
                ]
                panel += [
                    f"    {c:<7} {probs.get(c, 0.0):5.1%} {bar(probs.get(c, 0.0))}"
                    + ("  <" if c == shown else "")
                    for c in CANDIDATES
                ]
                panel += [
                    "",
                    f"  confidence {confidence:5.1%}   tau {tau:.0%}   T {temperature:g}",
                    f"  route      {routed}",
                    f"  agreement  {acc:5.1%} over {graded} local decisions",
                    f"  escalated  {escalations}/{decisions}",
                    f"  latency    {lat:5.1f} ms/decision",
                    "",
                    "  Ctrl-C to stop.",
                ]
                sys.stdout.write("\x1b[H" + game.render() + "\n" + "\n".join(panel) + "\x1b[J")
                sys.stdout.flush()
                time.sleep(max(0.0, 1 / args.fps - engine.latency_ms / 1000))
            frame += 1
    except KeyboardInterrupt:
        pass
    finally:
        if not args.no_render:
            sys.stdout.write("\x1b[?25h\n")  # restore cursor
            sys.stdout.flush()

    if args.stats or args.no_render:
        lat = sum(latencies) / len(latencies) if latencies else 0.0
        graded = decisions - escalations
        print(f"backend             {engine.name}")
        print(f"order invariant     {'on' if args.order_invariant else 'off'}")
        print(f"temperature T       {temperature:g}"
              + (f" (fitted on {args.calibrate})" if args.calibrate else ""))
        print(f"tau                 {tau:.2f}")
        print(f"frames              {frame}")
        print(f"decisions           {decisions}")
        print(f"escalated           {escalations} ({escalations / decisions:.1%})"
              if decisions else "escalated           0")
        print(f"agreement           {agreements}/{graded} ({agreements / graded:.1%})"
              if graded else "agreement           n/a")
        print(f"mean confidence     {acting_confidence / graded:.1%} when acting"
              if graded else "mean confidence     n/a")
        print(f"mean latency        {lat:.1f} ms")
        print(f"final score         AI {game.right_score} – {game.left_score} opponent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
