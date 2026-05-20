"""TensorBoard logging for PPO self-play training."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rl.runner import EpisodeStats


class TrainingLogger:
    """Writes training scalars to TensorBoard under a fixed log directory."""

    def __init__(self, log_dir: Path, *, enabled: bool = True) -> None:
        self.log_dir = Path(log_dir)
        self.enabled = enabled
        self._writer = None
        if enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            from torch.utils.tensorboard import SummaryWriter

            self._writer = SummaryWriter(log_dir=str(self.log_dir))

    def log_update(
        self,
        step: int,
        *,
        update: int,
        ppo_stats: dict[str, float],
        rollout_sec: float,
        batch_size: int,
        ep_stats: EpisodeStats | None = None,
        progress: float | None = None,
    ) -> None:
        if not self.enabled or self._writer is None:
            return
        w = self._writer
        w.add_scalar("train/update", update, step)
        w.add_scalar("train/batch_size", batch_size, step)
        w.add_scalar("time/rollout_sec", rollout_sec, step)
        if progress is not None:
            w.add_scalar("train/progress", progress, step)
        for key, value in ppo_stats.items():
            w.add_scalar(f"ppo/{key}", value, step)
        if ep_stats is not None:
            w.add_scalar("episode/steps", ep_stats.steps, step)
            w.add_scalar("episode/num_agents", ep_stats.num_agents, step)
            for i, r in enumerate(ep_stats.rewards):
                w.add_scalar(f"episode/reward_p{i}", r, step)
            for i, t in enumerate(ep_stats.terminals):
                w.add_scalar(f"episode/terminal_p{i}", t, step)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __enter__(self) -> TrainingLogger:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
