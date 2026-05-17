"""
Самоигра Orbit Wars: один и тот же main.py за обоих игроков, HTML-реплей на диск.

Запуск (среда crypto): conda run -n crypto python watch_selfplay.py
Опции: --seed, --output replay.html, --open (открыть браузер).
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from kaggle_environments import make
_ROOT = Path(__file__).resolve().parent
_DEFAULT_AGENT = _ROOT / "main.py"

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@dataclass
class TimingStats:
    agent_s: float = 0.0
    env_s: float = 0.0
    intercept_s: float = 0.0
    intercept_calls: int = 0
    agent_calls: int = 0
    env_steps: int = 0
    episode_s: float = 0.0
    render_s: float = 0.0


def install_intercept_timer(stats: TimingStats) -> None:
    """Patch intercept_angle_for_target before the agent module is loaded."""
    import continuous_intercept
    import orbit_dynamics

    for mod in (orbit_dynamics, continuous_intercept):
        original = mod.intercept_angle_for_target

        def timed(*args, _orig=original, **kwargs):
            t0 = perf_counter()
            try:
                return _orig(*args, **kwargs)
            finally:
                stats.intercept_s += perf_counter() - t0
                stats.intercept_calls += 1

        mod.intercept_angle_for_target = timed


def run_episode_timed(env, agent_paths: list[Path], stats: TimingStats) -> None:
    """Same as env.run, but accumulates agent / env step times."""
    agent_specs = [str(p) for p in agent_paths]
    runner = env._Environment__agent_runner(agent_specs)  # noqa: SLF001

    if env.state is None or len(env.steps) == 1 or env.done:
        env.reset(len(agent_specs))
    if len(env.state) != len(agent_specs):
        raise ValueError(f"expected {len(env.state)} agents, got {len(agent_specs)}")

    deadline = perf_counter() + float(env.configuration.runTimeout)
    t_episode = perf_counter()
    while not env.done and perf_counter() < deadline:
        t0 = perf_counter()
        actions, logs = runner.act()
        stats.agent_s += perf_counter() - t0
        stats.agent_calls += len(agent_specs)

        t0 = perf_counter()
        env.step(actions, logs)
        stats.env_s += perf_counter() - t0
        stats.env_steps += 1

    stats.episode_s = perf_counter() - t_episode


def print_timing_stats(stats: TimingStats) -> None:
    ep = stats.episode_s
    agent = stats.agent_s
    env = stats.env_s
    theta = stats.intercept_s
    agent_rest = max(0.0, agent - theta)
    overhead = max(0.0, ep - agent - env)

    def pct(part: float, whole: float) -> float:
        return 100.0 * part / whole if whole > 0 else 0.0

    print()
    print("=== Время (сек) ===")
    print(f"  эпизод (ходы):     {ep:10.3f}")
    print(f"  среда (step):      {env:10.3f}  ({stats.env_steps} шагов)")
    print(f"  агент (всего):     {agent:10.3f}  ({stats.agent_calls} вызовов)")
    print(f"    └ поиск θ:       {theta:10.3f}  ({stats.intercept_calls} вызовов intercept)")
    print(f"    └ остальное:     {agent_rest:10.3f}")
    if overhead > 0.001:
        print(f"  прочее в цикле:    {overhead:10.3f}")
    if stats.render_s > 0:
        print(f"  HTML render:       {stats.render_s:10.3f}")

    print()
    print("=== Доля от времени эпизода ===")
    print(f"  среда:             {pct(env, ep):6.1f}%")
    print(f"  агент (всего):     {pct(agent, ep):6.1f}%")
    print(f"    поиск θ:         {pct(theta, ep):6.1f}%  ({pct(theta, agent):.1f}% от агента)")
    print(f"    остальное:       {pct(agent_rest, ep):6.1f}%")
    if stats.intercept_calls:
        print(f"  среднее на intercept: {theta / stats.intercept_calls * 1e3:.2f} ms")
    if stats.agent_calls:
        print(f"  среднее на вызов agent: {agent / stats.agent_calls * 1e3:.2f} ms")
    if stats.env_steps:
        print(f"  среднее на step среды:  {env / stats.env_steps * 1e3:.2f} ms")


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

    stats = TimingStats()
    install_intercept_timer(stats)

    env = make("orbit_wars", configuration=configuration, debug=True)
    run_episode_timed(env, [agent_path, agent_path], stats)

    final = env.steps[-1]
    num_agents = len(final)
    obs0 = final[0].observation
    totals = ship_totals_from_obs(obs0, num_agents)
    for i, s in enumerate(final):
        print(f"Player {i}: reward={s.reward}, status={s.status}, ships_total={totals[i]}")

    t0 = perf_counter()
    html = env.render(mode="html", width=800, height=600)
    stats.render_s = perf_counter() - t0
    if not isinstance(html, str) or not html.strip():
        raise SystemExit("env.render(mode='html') did not return HTML string")
    out = output_path_with_scores(args.output.resolve(), totals)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out}")
    print_timing_stats(stats)
    if args.open:
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()
