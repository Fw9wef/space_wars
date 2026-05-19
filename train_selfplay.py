#!/usr/bin/env python3
"""Train PPO with self-play on Orbit Wars."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rl.buffer import RolloutBatch, RolloutBuffer
from rl.graph_encoding import GraphFeatureConfig
from rl.graph_policy import GraphActorCritic
from rl.ppo import PPOConfig, PPOTrainer
from rl.rewards import RewardConfig
from rl.runner import EpisodeStats, RolloutCollector, format_episode_stats


def _merge_batches(batches: list[RolloutBatch]) -> RolloutBatch:
    return RolloutBatch(
        nodes=np.concatenate([b.nodes for b in batches]),
        edges=np.concatenate([b.edges for b in batches]),
        global_features=np.concatenate([b.global_features for b in batches]),
        actions=np.concatenate([b.actions for b in batches]),
        logprobs=np.concatenate([b.logprobs for b in batches]),
        values=np.concatenate([b.values for b in batches]),
        rewards=np.concatenate([b.rewards for b in batches]),
        dones=np.concatenate([b.dones for b in batches]),
        node_valid=np.concatenate([b.node_valid for b in batches]),
        source_mask=np.concatenate([b.source_mask for b in batches]),
        target_mask=np.concatenate([b.target_mask for b in batches]),
        advantages=np.concatenate([b.advantages for b in batches]),
        returns=np.concatenate([b.returns for b in batches]),
    )


def _collect_rollout(
    policy: GraphActorCritic,
    *,
    seed: int,
    n_steps: int,
    progress: float,
    episode_steps: int,
    four_player_fraction: float,
    reward_config: RewardConfig,
    graph_config: GraphFeatureConfig,
    gamma: float,
    gae_lambda: float,
    log_interval: int,
    label: str,
) -> tuple[RolloutBatch, EpisodeStats | None]:
    collector = RolloutCollector(
        policy,
        seed=seed,
        episode_steps=episode_steps,
        reward_config=reward_config,
        training_progress=progress,
        four_player_fraction=four_player_fraction,
        graph_config=graph_config,
        debug=False,
    )
    buf = RolloutBuffer(n_steps, config=graph_config)

    def on_progress(env_steps: int, collected: int) -> None:
        print(
            f"  [{label}] env_steps={env_steps} transitions={collected}/{n_steps}",
            flush=True,
        )

    _, ep_stats, last_v = collector.collect(
        buf,
        n_steps,
        log_interval=log_interval,
        progress_callback=on_progress if log_interval > 0 else None,
    )
    batch = buf.compute_gae(last_v, gamma, gae_lambda)
    return batch, ep_stats


def save_checkpoint(
    path: Path,
    policy: GraphActorCritic,
    graph_config: GraphFeatureConfig,
    update: int,
    total_steps: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy": policy.state_dict(),
            "graph_config": {
                "history_steps": graph_config.history_steps,
                "future_steps": graph_config.future_steps,
                "include_edges": graph_config.include_edges,
            },
            "update": update,
            "total_steps": total_steps,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPO self-play training for Orbit Wars")
    p.add_argument("--total-timesteps", type=int, default=500_000)
    p.add_argument(
        "--n-steps",
        type=int,
        default=4096,
        help="Transitions per rollout (sum over all players each env step)",
    )
    p.add_argument(
        "--four-player-fraction",
        type=float,
        default=0.5,
        help="Fraction of episodes with 4-player FFA (rest are 1v1)",
    )
    p.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Sequential rollouts per update (different seeds, same process; no subprocess)",
    )
    p.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Print rollout progress every N env steps (0=disable)",
    )
    p.add_argument("--save-every", type=int, default=10, help="Checkpoint every N updates")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episode-steps", type=int, default=500)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--history-steps", type=int, default=5)
    p.add_argument("--future-steps", type=int, default=5)
    p.add_argument(
        "--graph-edges",
        action="store_true",
        help="Fill full N×N edge travel tensor (very slow; many intercept rollouts per step)",
    )
    p.add_argument(
        "--save-dir",
        type=Path,
        default=_ROOT / "runs" / "ppo_graph_v0",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    graph_config = GraphFeatureConfig(
        history_steps=args.history_steps,
        future_steps=args.future_steps,
        include_edges=args.graph_edges,
    )
    policy = GraphActorCritic(config=graph_config).to(device)
    ppo_cfg = PPOConfig(lr=args.lr, ent_coef=args.ent_coef)
    trainer = PPOTrainer(policy, ppo_cfg)
    reward_cfg = RewardConfig()

    total_steps = 0
    update = 0
    per_rollout = max(64, args.n_steps // max(1, args.n_envs))

    print(
        f"Training: device={device} rollouts/update={args.n_envs} "
        f"steps/rollout={per_rollout} episode_steps={args.episode_steps} "
        f"four_player_fraction={args.four_player_fraction}",
        flush=True,
    )

    while total_steps < args.total_timesteps:
        progress = total_steps / max(1, args.total_timesteps)
        t_rollout = time.perf_counter()
        batches: list[RolloutBatch] = []
        ep_stats = None

        print(f"rollout update={update + 1} progress={progress:.3f}", flush=True)
        for w in range(args.n_envs):
            batch, ep_stats_w = _collect_rollout(
                policy,
                seed=args.seed + update * 1000 + w,
                n_steps=per_rollout,
                progress=progress,
                episode_steps=args.episode_steps,
                four_player_fraction=args.four_player_fraction,
                reward_config=reward_cfg,
                graph_config=graph_config,
                gamma=ppo_cfg.gamma,
                gae_lambda=ppo_cfg.gae_lambda,
                log_interval=args.log_interval,
                label=f"{w + 1}/{args.n_envs}",
            )
            batches.append(batch)
            if w == args.n_envs - 1:
                ep_stats = ep_stats_w

        batch = batches[0] if len(batches) == 1 else _merge_batches(batches)
        rollout_s = time.perf_counter() - t_rollout
        print(f"  rollout done in {rollout_s:.1f}s ({batch.size} transitions)", flush=True)

        stats = trainer.update(batch)
        total_steps += batch.size
        update += 1

        ent = stats["entropy"]
        msg = (
            f"update={update} steps={total_steps} "
            f"pl={stats['policy_loss']:.4f} vl={stats['value_loss']:.4f} "
            f"ent={ent:.4f} kl={stats['approx_kl']:.5f}"
        )
        if ep_stats is not None:
            msg += f" {format_episode_stats(ep_stats)}"
        print(msg, flush=True)

        if update % args.save_every == 0:
            ckpt = args.save_dir / "checkpoint.pt"
            save_checkpoint(ckpt, policy, graph_config, update, total_steps)
            print(f"saved {ckpt}", flush=True)

    save_checkpoint(
        args.save_dir / "checkpoint_final.pt", policy, graph_config, update, total_steps
    )
    print("done", flush=True)


if __name__ == "__main__":
    main()
