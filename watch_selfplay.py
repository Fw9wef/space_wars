"""
Самоигра Orbit Wars: один и тот же main.py за обоих игроков, HTML-реплей на диск.

Запуск (среда crypto): conda run -n crypto python watch_selfplay.py
Опции: --seed, --output replay.html, --open (открыть браузер).
"""

from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

from kaggle_environments import make

_ROOT = Path(__file__).resolve().parent
_DEFAULT_AGENT = _ROOT / "main.py"


def _obs_get(obs: object, key: str, default):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def ship_totals_from_obs(obs: object, num_agents: int) -> list[int]:
    """Сумма кораблей игрока на планетах и во флотах (как при подсчёте победителя в среде)."""
    totals = [0] * num_agents
    for p in _obs_get(obs, "planets", []) or []:
        owner = p[1]
        if 0 <= owner < num_agents:
            totals[owner] += int(p[5])
    for f in _obs_get(obs, "fleets", []) or []:
        owner = f[1]
        if 0 <= owner < num_agents:
            totals[owner] += int(f[6])
    return totals


def output_path_with_scores(base: Path, totals: list[int]) -> Path:
    tag = "_".join(f"p{i}-{t}" for i, t in enumerate(totals))
    suf = base.suffix if base.suffix else ".html"
    return base.with_name(f"{base.stem}_{tag}{suf}")


def main() -> None:
    p = argparse.ArgumentParser(description="Orbit Wars self-play + HTML replay")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output",
        type=Path,
        default=_ROOT / "replays" / "replay.html",
        help="Path to write HTML player",
    )
    p.add_argument("--open", action="store_true", help="Open replay in default browser")
    p.add_argument(
        "--episode-steps",
        type=int,
        default=None,
        metavar="N",
        help="Override episodeSteps (default: env default 500)",
    )
    p.add_argument(
        "--agent",
        type=Path,
        default=_DEFAULT_AGENT,
        help="Path to agent main.py",
    )
    args = p.parse_args()
    agent_path = args.agent.resolve()
    if not agent_path.is_file():
        raise SystemExit(f"Agent file not found: {agent_path}")

    configuration: dict = {"seed": args.seed}
    if args.episode_steps is not None:
        configuration["episodeSteps"] = args.episode_steps

    env = make("orbit_wars", configuration=configuration, debug=True)
    env.run([str(agent_path), str(agent_path)])

    final = env.steps[-1]
    num_agents = len(final)
    obs0 = final[0].observation
    totals = ship_totals_from_obs(obs0, num_agents)
    for i, s in enumerate(final):
        print(f"Player {i}: reward={s.reward}, status={s.status}, ships_total={totals[i]}")

    html = env.render(mode="html", width=800, height=600)
    if not isinstance(html, str) or not html.strip():
        raise SystemExit("env.render(mode='html') did not return HTML string")
    out = output_path_with_scores(args.output.resolve(), totals)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out}")
    if args.open:
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()
