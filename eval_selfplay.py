#!/usr/bin/env python3
"""Evaluate RL agent vs heuristic main.py."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rl.runner import RLAgent, run_match


def load_heuristic_agent(path: Path):
    spec = importlib.util.spec_from_file_location("heuristic_agent", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.agent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Eval RL vs heuristic")
    p.add_argument("--checkpoint", type=Path, default=_ROOT / "runs" / "ppo_v0" / "checkpoint.pt")
    p.add_argument("--heuristic", type=Path, default=_ROOT / "main.py")
    p.add_argument("--games", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episode-steps", type=int, default=500)
    p.add_argument("--rl-player", type=int, default=0, choices=(0, 1))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    rl = RLAgent.from_checkpoint(str(args.checkpoint), deterministic=True)
    heuristic = load_heuristic_agent(args.heuristic.resolve())

    wins = draws = 0
    for g in range(args.games):
        seed = args.seed + g
        if args.rl_player == 0:
            a0, a1 = rl, heuristic
        else:
            a0, a1 = heuristic, rl
        final, winner = run_match(a0, a1, seed=seed, episode_steps=args.episode_steps)
        r0, r1 = final[0].reward, final[1].reward
        if winner < 0:
            draws += 1
        elif winner == args.rl_player:
            wins += 1
        print(f"game {g} seed={seed} r0={r0} r1={r1} winner={winner}")

    losses = args.games - wins - draws
    print(f"RL wins: {wins}/{args.games} ({100.0 * wins / args.games:.1f}%)")
    print(f"draws: {draws} losses: {losses}")


if __name__ == "__main__":
    main()
