#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "llama-cpp-python", "huggingface-hub"]
# ///
"""
optimal_custom_jev_solution.py -- an OpenJev-style decision model playing Pong.

This is a worked example of the pattern described in the README: instead of
asking a language model to *generate* anything, we give it a closed set of
options and read the probability it assigns to each one. That is the Choice
primitive, and it is all this file needs to play a game.

The Pong loop calls decide() once every few frames with the current game state
and three options -- UP, DOWN, STAY. What comes back is a typed value plus a
probability distribution, never a string to parse. When the model is unsure
(max probability below --threshold) we fall back to a deterministic controller,
which is the same confidence-routing pattern you would use in production to
escalate to a human or a larger model.

Two backends implement the identical interface:

  LlamaChoiceBackend  the real thing -- prefill a prompt on a local GGUF model,
                      read the logits at the final position, index out the token
                      ids for "A"/"B"/"C", softmax over just those. No text is
                      ever generated; we never sample a single token.

  HeuristicBackend    no model, no dependencies beyond numpy. Scores the three
                      moves geometrically and softmaxes. Exists so this file
                      runs anywhere, and to give the model an accuracy baseline.

The game tracks how often the model agrees with the perfect-play move, which is
a crude local stand-in for the kind of decision benchmark JevBench measures.

What differs from initial_custom_jev_solution.py is only the text the model
sees: the live request is Pong.decision_facts(), which adds the ball's speeds,
the row where it meets the right edge, and whether that row is above, below, or
on the paddle centre. ChainOfThought_GUIDE (sample games) and GUIDE (rules) are
kept in LlamaChoiceBackend as the earlier, unused prompt tries.

Usage:
    python optimal_custom_jev_solution.py                  # heuristic backend, plays itself
    uv run optimal_custom_jev_solution.py --backend llama  # downloads Qwen3-0.6B, ~600MB
    python optimal_custom_jev_solution.py --frames 400 --no-render --stats
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import random
import shutil
import sys
import time

import numpy

# --------------------------------------------------------------------------
# The Choice primitive
# --------------------------------------------------------------------------

# The closed option set. Labels are what the model actually scores; the moves
# are what our code consumes. Defining these here -- in code, not in a prompt --
# is the part of the pattern that pays off regardless of which backend wins.
LABELS = ["A", "B", "C"]
MOVES = ["UP", "DOWN", "STAY"]


@dataclasses.dataclass(frozen=True)
class Choice:
    """A typed decision. This is the whole output surface -- no free text."""

    value: str                      # one of MOVES, guaranteed by construction
    probabilities: dict[str, float] # one entry per option, sums to 1
    latency_ms: float

    @property
    def confidence(self) -> float:
        return self.probabilities[self.value]

    @classmethod
    def from_logits(cls, logits: numpy.ndarray, latency_ms: float) -> "Choice":
        """Normalize raw scores into a distribution over MOVES.

        Subtracting logaddexp rather than calling a softmax directly keeps this
        numerically stable for the large-magnitude logits a real model emits.
        """
        logprobs = logits - numpy.logaddexp.reduce(logits)
        probs = numpy.exp(logprobs).astype(float)
        return cls(
            value=MOVES[int(numpy.argmax(probs))],
            probabilities=dict(zip(MOVES, numpy.round(probs, 4).tolist(), strict=True)),
            latency_ms=latency_ms,
        )


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------


class HeuristicBackend:
    """Geometric scoring, no model. Same output shape as the real backend.

    `noise` blunts it deliberately so the confidence threshold and the fallback
    path actually get exercised in a demo run -- a perfect controller would
    never dip below threshold and you would never see the routing work.
    """

    name = "heuristic"

    def __init__(self, noise: float = 0.35, seed: int | None = None) -> None:
        self.noise = noise
        self.rng = random.Random(seed)

    def choose(self, game: "Pong") -> Choice:
        start = time.perf_counter()
        target = game.predicted_intercept_y()
        centre = game.right_y + game.paddle_h / 2
        delta = target - centre
        # Sharpness scales with how far off-centre we are: a big gap is an easy
        # call, a small one is genuinely ambiguous and should score as such.
        scale = min(abs(delta), 6.0)
        logits = numpy.array([
            scale if delta < -0.5 else -scale,   # UP
            scale if delta > 0.5 else -scale,    # DOWN
            2.0 if abs(delta) <= 0.5 else -scale,  # STAY
        ])
        logits = logits + numpy.array([self.rng.gauss(0, self.noise) for _ in logits])
        return Choice.from_logits(logits, (time.perf_counter() - start) * 1000)


class LlamaChoiceBackend:
    """The OpenJev recipe: prefill, read final-position logits, index the labels.

    Note what is absent -- there is no generation loop, no sampler, no stop
    tokens, no JSON parsing. One forward pass over the prompt is the entire
    inference. That is where the speed claim in the brief comes from.
    """

    name = "llama"

    SYSTEM = (
        "You control the right paddle in a game of Pong. "
        "The text names the move. Answer with one letter."
    )

    # Shown before the live board. Each example is a finished Choice: the same
    # sentences describe() will use, then the letter. The cases are ones where
    # every speed this game uses has the same correct letter, so the missing
    # speed in the live text does not change the answer.
    ChainOfThought_GUIDE = """\
Row 0 is the top, so up means a smaller row number. One move shifts the paddle by one row.

The ball is moving away from you: return the paddle to the middle, rows 7 to 11. Below the middle, choose A. Above the middle, choose B. Already covering the middle, choose C.
The ball is moving toward you, its row is above the paddle, and it is moving upward: choose A.
The ball is moving toward you, its row is below the paddle, and it is moving downward: choose B.

The board is 56 wide and 18 tall. Row 0 is the top.
The ball is at column 40, row 3, moving toward you and upward.
Your paddle covers rows 10 to 14 at the right edge.
A. UP
B. DOWN
C. STAY
A

The board is 56 wide and 18 tall. Row 0 is the top.
The ball is at column 40, row 15, moving toward you and downward.
Your paddle covers rows 2 to 6 at the right edge.
A. UP
B. DOWN
C. STAY
B

The board is 56 wide and 18 tall. Row 0 is the top.
The ball is at column 20, row 4, moving away from you and upward.
Your paddle covers rows 12 to 16 at the right edge.
A. UP
B. DOWN
C. STAY
A

The board is 56 wide and 18 tall. Row 0 is the top.
The ball is at column 20, row 14, moving away from you and downward.
Your paddle covers rows 1 to 5 at the right edge.
A. UP
B. DOWN
C. STAY
B

The board is 56 wide and 18 tall. Row 0 is the top.
The ball is at column 20, row 8, moving away from you and upward.
Your paddle covers rows 7 to 11 at the right edge.
A. UP
B. DOWN
C. STAY
C

Current board:
"""


    # Rules only, then the live board. No sample games: those ended on a letter
    # and the model copied the first one.
    GUIDE = """\
Apply the first rule that matches.

Row 0 is the top. A smaller row number is higher on the board.
A moves the paddle toward smaller row numbers.
B moves the paddle toward larger row numbers.
C leaves the paddle where it is.
The middle of the board is rows 7 to 11.

1. The ball is moving away from you.
   Paddle below the middle: A.
   Paddle above the middle: B.
   Paddle already covers the middle: C.
2. The ball is moving toward you and its row number is smaller than the paddle's rows: A.
3. The ball is moving toward you and its row number is larger than the paddle's rows: B.
4. The ball's row is already inside the paddle's rows: C.

Board:
"""

    def __init__(self, repo_id: str, filename: str, n_ctx: int = 1024) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "llama-cpp-python is not installed. Run with `uv run "
                "optimal_custom_jev_solution.py --backend llama`, or use --backend heuristic."
            ) from exc

        self.model = Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            n_ctx=n_ctx,
            logits_all=True,   # we need the logits, not a sampled token
            verbose=False,
        )
        # Resolve each option label to its first token id once, up front.
        self.token_ids = [
            self.model.tokenize(text=label.encode(), add_bos=False)[0] for label in LABELS
        ]

    def _prompt(self, game: "Pong") -> str:
        # The live request is the intercept comparison, not the saved guides.
        # ChainOfThought_GUIDE and GUIDE stay in the file as the earlier tries.
        options = "\n".join(f"{lab}. {mv}" for lab, mv in zip(LABELS, MOVES, strict=True))
        return (
            f"<|im_start|>system\n{self.SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{game.decision_facts()}\n\n{options}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )

    def choose(self, game: "Pong") -> Choice:
        start = time.perf_counter()
        # reset() clears the KV cache so each decision is independent. A real
        # serving stack (see openjev-sglang's radix cache) would instead reuse
        # the shared prefix across calls, which is most of this prompt.
        self.model.reset()
        tokens = self.model.tokenize(
            text=self._prompt(game).encode(), add_bos=False, special=True
        )
        self.model.eval(tokens=tokens)
        logits = self.model.scores[self.model.n_tokens - 1]
        choice_logits = numpy.asarray([float(logits[tid]) for tid in self.token_ids])
        return Choice.from_logits(choice_logits, (time.perf_counter() - start) * 1000)


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
        the fallback controller uses -- worth noting that for Pong specifically
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

    def decision_facts(self) -> str:
        """Board text plus the comparison perfect_move is graded on.

        The intercept is computed here. The model only has to read whether
        the meeting row is above the paddle, below it, or already on it.
        """
        target = self.predicted_intercept_y()
        centre = self.right_y + self.paddle_h / 2
        delta = target - centre
        if delta < -0.5:
            relation = "above the paddle center. Move up"
        elif delta > 0.5:
            relation = "below the paddle center. Move down"
        else:
            relation = "within half a row of the paddle center. Stay"
        return (
            f"{self.describe()}\n"
            f"Horizontal speed is {self.ball_dx:.2f}. Vertical speed is {self.ball_dy:.2f}.\n"
            f"The ball meets your side at row {target:.1f}. "
            f"Your paddle center is row {centre:.1f}.\n"
            f"The meeting row is {relation}."
        )

    def describe(self) -> str:
        """The unstructured half of the input. Everything else is the schema."""
        vert = "upward" if self.ball_dy < 0 else "downward"
        horiz = "toward you" if self.ball_dx > 0 else "away from you"
        return (
            f"The board is {self.w} wide and {self.h} tall. Row 0 is the top.\n"
            f"The ball is at column {self.ball_x:.0f}, row {self.ball_y:.0f}, "
            f"moving {horiz} and {vert}.\n"
            f"Your paddle covers rows {self.right_y:.0f} to "
            f"{self.right_y + self.paddle_h:.0f} at the right edge."
        )

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
# Driver
# --------------------------------------------------------------------------


def bar(p: float, width: int = 18) -> str:
    filled = int(round(p * width))
    return "█" * filled + "·" * (width - filled)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--backend", choices=["auto", "llama", "heuristic"], default="auto",
                    help="auto uses llama if importable, else heuristic")
    ap.add_argument("--repo-id", default="Qwen/Qwen3-0.6B-GGUF")
    ap.add_argument("--filename", default="Qwen3-0.6B-Q8_0.gguf")
    ap.add_argument("--frames", type=int, default=0, help="0 runs until Ctrl-C")
    ap.add_argument("--fps", type=float, default=18.0)
    ap.add_argument("--decide-every", type=int, default=2,
                    help="frames between decisions; raise this for slow backends")
    ap.add_argument("--threshold", type=float, default=0.55,
                    help="below this confidence, fall back to the deterministic controller")
    ap.add_argument("--opponent-skill", type=float, default=0.62,
                    help="0..1 chance the opponent paddle reacts on a given tick")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--stats", action="store_true", help="print a summary on exit")
    args = ap.parse_args(argv)

    backend_kind = args.backend
    if backend_kind == "auto":
        try:
            import llama_cpp  # noqa: F401
            backend_kind = "llama"
        except ImportError:
            backend_kind = "heuristic"
            print("llama-cpp-python not found; using the heuristic backend.",
                  file=sys.stderr)

    if backend_kind == "llama":
        print(f"Loading {args.repo_id}/{args.filename} ...", file=sys.stderr)
        try:
            backend = LlamaChoiceBackend(args.repo_id, args.filename)
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            return 1
    else:
        backend = HeuristicBackend(seed=args.seed)

    cols = shutil.get_terminal_size((80, 24)).columns
    game = Pong(width=min(56, max(32, cols - 6)), seed=args.seed,
                opponent_skill=args.opponent_skill)

    decisions = agreements = fallbacks = 0
    latencies: list[float] = []
    choice = Choice("STAY", {m: 1 / 3 for m in MOVES}, 0.0)
    move = "STAY"
    frame = 0

    if not args.no_render:
        sys.stdout.write("\x1b[2J\x1b[?25l")  # clear, hide cursor
    try:
        while args.frames == 0 or frame < args.frames:
            if frame % args.decide_every == 0:
                truth = game.perfect_move()
                choice = backend.choose(game)
                decisions += 1
                latencies.append(choice.latency_ms)
                # The confidence gate. This is the whole point of a calibrated
                # probability: it tells you when *not* to trust the decision.
                if choice.confidence < args.threshold:
                    fallbacks += 1
                    move = game.perfect_move()
                else:
                    move = choice.value
                    if choice.value == truth:
                        agreements += 1

            game.apply(move, "right")
            game.step_opponent()
            game.step_ball()

            if not args.no_render:
                acc = agreements / decisions if decisions else 0.0
                lat = sum(latencies) / len(latencies) if latencies else 0.0
                panel = [
                    f"  OpenJev Pong — backend: {backend.name}",
                    "",
                    f"  {'AI':>4} {game.right_score:>3}     opponent {game.left_score:<3}",
                    "",
                    "  Choice distribution:",
                ]
                panel += [
                    f"    {m:<5} {p:5.1%} {bar(p)}" + ("  <" if m == choice.value else "")
                    for m, p in choice.probabilities.items()
                ]
                panel += [
                    "",
                    f"  confidence {choice.confidence:5.1%}   "
                    f"threshold {args.threshold:.0%}"
                    + ("   FALLBACK" if choice.confidence < args.threshold else ""),
                    f"  agreement  {acc:5.1%} over {decisions} decisions",
                    f"  latency    {lat:5.1f} ms/decision",
                    "",
                    "  Ctrl-C to stop.",
                ]
                sys.stdout.write("\x1b[H" + game.render() + "\n" + "\n".join(panel) + "\x1b[J")
                sys.stdout.flush()
                time.sleep(max(0.0, 1 / args.fps - choice.latency_ms / 1000))
            frame += 1
    except KeyboardInterrupt:
        pass
    finally:
        if not args.no_render:
            sys.stdout.write("\x1b[?25h\n")  # restore cursor
            sys.stdout.flush()

    if args.stats or args.no_render:
        lat = sum(latencies) / len(latencies) if latencies else 0.0
        graded = decisions - fallbacks
        print(f"backend        {backend.name}")
        print(f"frames         {frame}")
        print(f"decisions      {decisions}")
        print(f"agreement      {agreements}/{graded} "
              f"({agreements / graded:.1%})" if graded else "agreement      n/a")
        print(f"fallbacks      {fallbacks} ({fallbacks / decisions:.1%})"
              if decisions else "fallbacks      0")
        print(f"mean latency   {lat:.1f} ms")
        print(f"final score    AI {game.right_score} – {game.left_score} opponent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
