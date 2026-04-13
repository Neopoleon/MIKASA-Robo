"""evaluate imitation learning policies on MIKASA-Robo environments.

modes:
  eval:          python mikasa_eval.py --env-id ShellGameTouch-v0 --checkpoint path/to/ckpt --num-envs 16
                 normal eval with pd_joint_delta_pos controller.
  abs-joint-pos: python mikasa_eval.py --env-id RememberColor3-v0 --checkpoint path/to/ckpt \
                     --abs-joint-pos --num-envs 16
                 policy predicts absolute target qpos, executed via pd_joint_pos PD controller.
  direct-qpos:   python mikasa_eval.py --env-id RememberShape3-v0 --checkpoint path/to/ckpt \
                     --direct-qpos --num-envs 16
                 policy predicts absolute joint positions, robot is teleported (no physics).
  replay:        python mikasa_eval.py --env-id RememberColor3-v0 --checkpoint path/to/ckpt \
                     --replay-data path/to/episode_data.zarr --replay-episodes 0 1 5
                 feeds expert obs to policy, overlays expert (green) vs policy (red) actions.

any mode can be combined with --train-seeds-only to eval on exact training seeds.
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
import zarr
from tqdm import tqdm

from eval_utils import (
    TASK_REGISTRY,
    ACTION_HORIZON,
    load_policy,
    make_eval_env,
    predict_action,
    teleport_and_step,
    EpisodeBuffers,
    handle_episode_completions,
    _divergence_stats,
    _annotate_replay_obs_frames,
    _save_video_and_frames,
)


def _maybe_report_progress(policy, step: int, total: int) -> None:
    """Report progress to tmp file if using remote policy (for orchestrator to read)."""
    if hasattr(policy, "report_progress"):
        policy.report_progress(step, total)


# eval loop

def run_eval(env, inner_env, policy, normalizer, num_envs, num_eval_steps, device, seed,
             output_dir, env_id, no_proprio=False, action_horizon=ACTION_HORIZON, task_dir=None):
    """predict → execute ACTION_HORIZON steps → repeat. save episodes on completion.
    output: <task_dir>/episode_NNNN_{success|failure}/ + summary.json
    """
    if task_dir is None:
        task_dir = os.path.join(output_dir, env_id)
    os.makedirs(task_dir, exist_ok=True)

    policy.reset()
    obs, _ = env.reset(seed=seed)
    buffers = EpisodeBuffers(num_envs)
    metrics = defaultdict(list)
    num_episodes = 0
    completed = np.zeros(num_envs, dtype=bool)
    step = 0
    pbar = tqdm(total=num_eval_steps, desc="eval")

    while step < num_eval_steps:
        action_chunk = predict_action(obs, policy, normalizer, num_envs, device, no_proprio=no_proprio, action_horizon=action_horizon)

        for t in range(action_horizon):
            if step >= num_eval_steps:
                break

            render_frame = inner_env.render()  # (B,H,W,3) with wrapper overlays
            buffers.append_obs(obs, render_frame=render_frame)
            action = action_chunk[:, t, :]
            buffers.append_action(action)

            obs, reward, _term, _trunc, infos = env.step(action)
            buffers.append_reward(reward)
            step += 1
            pbar.update(1)
            _maybe_report_progress(policy, step, num_eval_steps)

            if "final_info" in infos:
                num_episodes = handle_episode_completions(
                    infos, buffers, num_envs, num_episodes, task_dir, metrics, env=env,
                )
                completed |= infos["_final_info"].cpu().numpy()
                if completed.all():
                    # all num_envs done — safe to reset policy memory.
                    # ManiSkill auto-resets done envs, obs is already the
                    # fresh reset obs for the next round.
                    policy.reset()
                    completed[:] = False
                    break  # discard remaining chunk actions

    pbar.close()

    results = {"num_episodes": num_episodes}
    for k, v in metrics.items():
        results[k] = torch.stack(v).float().mean().item()

    with open(os.path.join(task_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {num_episodes} episodes to {task_dir}/")

    return results


# direct-qpos mode — policy predicts absolute joint positions, robot is teleported there.
# bare minimum test: if the policy can't predict the next state, action-based control won't work either.

def run_direct_qpos(env, inner_env, policy, normalizer, num_envs, num_eval_steps, device, seed,
                    output_dir, env_id, no_proprio=False, action_horizon=ACTION_HORIZON, task_dir=None):
    """policy predicts absolute qpos → teleport robot → get obs → repeat.
    same as run_eval but replaces env.step(action) with teleport_to_qpos(predicted_state).
    """
    if task_dir is None:
        task_dir = os.path.join(output_dir, env_id)
    os.makedirs(task_dir, exist_ok=True)

    policy.reset()
    obs, _ = env.reset(seed=seed)
    buffers = EpisodeBuffers(num_envs)
    metrics = defaultdict(list)
    num_episodes = 0
    completed = np.zeros(num_envs, dtype=bool)
    step = 0
    pbar = tqdm(total=num_eval_steps, desc="direct-qpos eval")

    while step < num_eval_steps:
        action_chunk = predict_action(obs, policy, normalizer, num_envs, device, no_proprio=no_proprio, action_horizon=action_horizon)

        for t in range(action_horizon):
            if step >= num_eval_steps:
                break

            render_frame = inner_env.render()
            buffers.append_obs(obs, render_frame=render_frame)
            predicted_qpos = action_chunk[:, t, :]  # (B, 8) — absolute joint positions
            buffers.append_action(predicted_qpos)

            # teleport robot and advance env (bypasses PD controller entirely)
            obs, reward, _term, _trunc, infos = teleport_and_step(env, predicted_qpos)
            buffers.append_reward(reward)
            step += 1
            pbar.update(1)
            _maybe_report_progress(policy, step, num_eval_steps)

            if "final_info" in infos:
                num_episodes = handle_episode_completions(
                    infos, buffers, num_envs, num_episodes, task_dir, metrics, env=env,
                )
                completed |= infos["_final_info"].cpu().numpy()
                if completed.all():
                    policy.reset()
                    completed[:] = False
                    break

    pbar.close()

    results = {"num_episodes": num_episodes}
    for k, v in metrics.items():
        results[k] = torch.stack(v).float().mean().item()

    with open(os.path.join(task_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {num_episodes} episodes to {task_dir}/")

    return results


# train-seeds-only mode — loop through exact training seeds in parallel batches.
# ensures each eval episode uses the exact same scene as training.

def run_with_train_seeds(env, inner_env, policy, normalizer, device, train_seeds, task_cfg,
                         num_envs, output_dir, env_id, no_proprio=False, direct_qpos=False,
                         action_horizon=ACTION_HORIZON, task_dir=None):
    """run eval/direct-qpos on training seeds in batches of num_envs.
    each episode gets a unique seed from train_seeds. resets all envs per batch.
    """
    if task_dir is None:
        task_dir = os.path.join(output_dir, env_id)
    os.makedirs(task_dir, exist_ok=True)
    episode_steps = task_cfg.episode_steps
    mode = "direct-qpos" if direct_qpos else "eval"

    # pad seed list so it divides evenly into batches of num_envs
    remainder = len(train_seeds) % num_envs
    if remainder:
        train_seeds = train_seeds + train_seeds[:num_envs - remainder]
    num_batches = len(train_seeds) // num_envs

    print(f"\ntrain-seeds-only {mode}: {env_id} | {len(train_seeds)} seeds | "
          f"{episode_steps} steps/ep | {num_envs} envs | {num_batches} batches")

    all_metrics = defaultdict(list)
    num_episodes = 0

    # process seeds in batches of num_envs
    for batch_start in tqdm(range(0, len(train_seeds), num_envs), desc=f"train-seeds {mode}"):
        batch_seeds = train_seeds[batch_start:batch_start + num_envs]
        policy.reset()
        obs, _ = env.reset(seed=batch_seeds)
        buffers = EpisodeBuffers(num_envs)

        for step in range(episode_steps):
            action_chunk = predict_action(obs, policy, normalizer, num_envs, device, no_proprio=no_proprio, action_horizon=action_horizon)
            action = action_chunk[:, 0, :]

            render_frame = inner_env.render()
            buffers.append_obs(obs, render_frame=render_frame)
            buffers.append_action(action)

            if direct_qpos:
                obs, reward, _term, _trunc, infos = teleport_and_step(env, action)
            else:
                obs, reward, _term, _trunc, infos = env.step(action)
            buffers.append_reward(reward)
            total_steps = len(train_seeds) * episode_steps
            _maybe_report_progress(policy, batch_start + step, total_steps)

            if "final_info" in infos:
                num_episodes = handle_episode_completions(
                    infos, buffers, num_envs, num_episodes, task_dir, all_metrics, env=env,
                )

    results = {"num_episodes": num_episodes}
    for k, v in all_metrics.items():
        results[k] = torch.stack(v).float().mean().item()

    with open(os.path.join(task_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {num_episodes} episodes to {task_dir}/")

    return results


# replay comparison — feed expert obs from zarr to policy, compare predicted vs expert actions.
# no env stepping — uses the exact observations the expert saw.

def _zarr_obs_to_policy_input(ep, t, device, no_proprio=False):
    """construct policy input from zarr episode at timestep t. no env needed.
    zarr keys: third_person_camera (T,256,256,3), robot0_wrist_camera (T,256,256,3), robot0_8d (T,8).
    """
    tp = torch.from_numpy(ep["third_person_camera"][t]).float().div_(255.0).permute(2, 0, 1).unsqueeze(0).unsqueeze(1)
    wc = torch.from_numpy(ep["robot0_wrist_camera"][t]).float().div_(255.0).permute(2, 0, 1).unsqueeze(0).unsqueeze(1)
    result = {
        "third_person_camera": tp.to(device),
        "robot0_wrist_camera": wc.to(device),
        "episode_idx": torch.zeros(1, device=device, dtype=torch.long),
    }
    if not no_proprio:
        proprio = torch.from_numpy(ep["robot0_8d"][t]).float().unsqueeze(0).unsqueeze(1)
        result["robot0_8d"] = proprio.to(device)
    return result


def run_replay(policy, normalizer, device, replay_data_path,
               replay_episodes, output_dir, env_id, fps=10, no_proprio=False):
    """replay expert episodes offline: feed expert obs to policy, compare predicted vs expert actions.
    no env needed — loads obs directly from zarr (the exact frames the expert saw).
    zarr structure: episode_N/{action0_8d, third_person_camera, robot0_wrist_camera, robot0_8d}.
    output per episode: obs video/frames with expert (green) vs policy (red) action overlays + metrics.
    """
    task_dir = os.path.join(output_dir, env_id, "replay")
    os.makedirs(task_dir, exist_ok=True)

    zroot = zarr.open_group(replay_data_path, mode="r")
    all_ep_keys = sorted(zroot.keys(), key=lambda k: int(k.split("_")[1]))

    if replay_episodes is not None:
        ep_keys = [f"episode_{i}" for i in replay_episodes if f"episode_{i}" in zroot]
    else:
        ep_keys = all_ep_keys

    print(f"replay: {len(ep_keys)} episodes from {replay_data_path}")

    for ep_key in tqdm(ep_keys, desc="replay episodes"):
        ep = zroot[ep_key]
        expert_actions = ep["action0_8d"][:]       # (T, 8)
        T = len(expert_actions)

        policy.reset()
        policy_actions = []

        for t in range(T):
            with torch.no_grad():
                policy_input = _zarr_obs_to_policy_input(ep, t, device, no_proprio=no_proprio)
                policy_input = normalizer.normalize(policy_input)
                action_dict = policy.predict_action(policy_input)
                action_dict = normalizer.unnormalize(action_dict)
                pred_action = action_dict["action0_8d"][0, 0, :].cpu().numpy()
            policy_actions.append(pred_action)

        # save episode
        ep_idx = int(ep_key.split("_")[1])
        episode_dir = os.path.join(task_dir, f"episode_{ep_idx:04d}")
        os.makedirs(episode_dir, exist_ok=True)

        expert_arr = expert_actions                              # (T, 8)
        policy_arr = np.stack(policy_actions)                    # (T, 8)
        per_step_mse = np.mean((expert_arr - policy_arr) ** 2, axis=1)  # (T,)

        # obs side-by-side video: concat third_person + wrist horizontally
        tp_frames = ep["third_person_camera"][:]   # (T, 256, 256, 3)
        wc_frames = ep["robot0_wrist_camera"][:]   # (T, 256, 256, 3)
        obs_frames = np.concatenate([tp_frames, wc_frames], axis=2)  # (T, 256, 512, 3)

        # no reward available offline — pass zeros for overlay
        dummy_rewards = np.zeros(T, dtype=np.float32)
        obs_annotated = _annotate_replay_obs_frames(
            obs_frames.copy(), expert_arr, policy_arr, dummy_rewards,
        )
        _save_video_and_frames(episode_dir, "obs", obs_annotated, fps)

        # per-step divergence stats
        per_step_cos, per_step_rel = [], []
        for t in range(T):
            cs, re = _divergence_stats(expert_arr[t], policy_arr[t])
            per_step_cos.append(cs)
            per_step_rel.append(re)

        # metrics
        metrics = {
            "episode_idx": ep_idx,
            "num_steps": T,
            "mean_cosine_sim_%": float(np.mean(per_step_cos)),
            "mean_relative_err_%": float(np.mean(per_step_rel)),
            "mean_mse": float(per_step_mse.mean()),
            "max_mse": float(per_step_mse.max()),
            "max_mse_step": int(per_step_mse.argmax()),
            "per_step_cosine_sim_%": per_step_cos,
            "per_step_relative_err_%": per_step_rel,
            "per_step_mse": per_step_mse.tolist(),
        }
        with open(os.path.join(episode_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    print(f"saved {len(ep_keys)} replay episodes to {task_dir}/")


# cli

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate IL policy on MIKASA-Robo")
    parser.add_argument("--env-id", type=str, default=None, choices=list(TASK_REGISTRY.keys()),
                        help="task env id (auto-detected from server config when using --remote-server)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="path to .ckpt file (not needed when using --remote-server)")
    parser.add_argument("--remote-server", type=str, default=None,
                        help="tcp://host:port of RemotePolicyServer. Overrides --checkpoint.")
    parser.add_argument("--progress-file", type=str, default=None,
                        help="path to write eval progress JSON (for orchestrator). auto-created if omitted.")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--num-eval-steps", type=int, default=None,
                        help="total env.step() calls (overrides --num-eval-episodes)")
    parser.add_argument("--num-eval-episodes", type=int, default=None,
                        help="target number of episodes (rounds up to fill parallel envs)")
    parser.add_argument("--capture-video", action="store_true")
    parser.add_argument("--output-dir", type=str, default="eval_results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reward-mode", type=str, default="normalized_dense")
    # replay comparison mode
    parser.add_argument("--replay-data", type=str, default=None,
                        help="path to expert zarr (episode_data.zarr). enables replay mode: "
                             "feed expert obs to policy offline, compare predictions per step.")
    parser.add_argument("--replay-episodes", type=int, nargs="+", default=None,
                        help="which episode indices to replay (default: all)")
    parser.add_argument("--no-proprio", action="store_true",
                        help="vision-only policy: omit robot0_8d from policy input")
    parser.add_argument("--abs-joint-pos", action="store_true",
                        help="policy predicts absolute target qpos; executed via pd_joint_pos "
                             "PD controller (real physics, unlike --direct-qpos)")
    parser.add_argument("--direct-qpos", action="store_true",
                        help="policy predicts absolute joint positions; robot is teleported "
                             "to predicted qpos each step (perfect controller, no physics)")
    parser.add_argument("--train-seeds-only", type=str, default=None,
                        help="path to training seeds JSON file (e.g. seeds/train_seeds.json). "
                             "eval on exact training seeds in batches of --num-envs.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.remote_server:
        from remote_policy_client import RemotePolicyClient
        policy = RemotePolicyClient(args.remote_server, progress_file=args.progress_file)
        normalizer = None
        if args.env_id is None:
            args.env_id = policy.env_id
        action_horizon = policy.action_horizon
        args.no_proprio = policy.no_proprio
    else:
        if args.checkpoint is None or args.env_id is None:
            raise ValueError("--checkpoint and --env-id are required when not using --remote-server")
        policy, normalizer = load_policy(args.checkpoint, device)
        action_horizon = ACTION_HORIZON

    task_cfg = TASK_REGISTRY[args.env_id]

    if args.abs_joint_pos and args.direct_qpos:
        raise ValueError("--abs-joint-pos and --direct-qpos are mutually exclusive")

    # load training seeds if requested
    train_seeds = None
    if args.train_seeds_only:
        with open(args.train_seeds_only) as f:
            train_seeds = json.load(f)
        print(f"\ntrain-seeds-only mode: loaded {len(train_seeds)} seeds from {args.train_seeds_only}")

    # replay mode: no env needed
    if args.replay_data:
        print(f"\nreplay mode: {args.env_id} | expert data: {args.replay_data}")
        run_replay(
            policy, normalizer, device,
            replay_data_path=args.replay_data,
            replay_episodes=args.replay_episodes,
            output_dir=args.output_dir, env_id=args.env_id,
            no_proprio=args.no_proprio,
        )
        return

    # determine control mode
    if args.abs_joint_pos:
        control_mode = "pd_joint_pos"
        mode_label = "abs-joint-pos"
    elif args.direct_qpos:
        control_mode = "pd_joint_delta_pos"  # doesn't matter — teleport bypasses controller
        mode_label = "direct-qpos"
    else:
        control_mode = "pd_joint_delta_pos"
        mode_label = "eval"

    if args.num_eval_steps is not None:
        num_eval_steps = args.num_eval_steps
    elif args.num_eval_episodes is not None:
        import math
        num_batches = math.ceil(args.num_eval_episodes / args.num_envs)
        num_eval_steps = num_batches * task_cfg.episode_steps
    else:
        num_eval_steps = task_cfg.episode_steps
    env, inner_env = make_eval_env(
        env_id=args.env_id, task_cfg=task_cfg, num_envs=args.num_envs,
        num_eval_steps=num_eval_steps, capture_video=args.capture_video,
        output_dir=args.output_dir, seed=args.seed, reward_mode=args.reward_mode,
        control_mode=control_mode,
    )

    if train_seeds is not None:
        results = run_with_train_seeds(
            env, inner_env, policy, normalizer, device, train_seeds, task_cfg,
            num_envs=args.num_envs, output_dir=args.output_dir, env_id=args.env_id,
            no_proprio=args.no_proprio, direct_qpos=args.direct_qpos,
            action_horizon=action_horizon,
        )
    elif args.direct_qpos:
        print(f"\n{mode_label}: {args.env_id} | {num_eval_steps} steps | {args.num_envs} envs")
        results = run_direct_qpos(
            env, inner_env, policy, normalizer, args.num_envs, num_eval_steps, device, args.seed,
            output_dir=args.output_dir, env_id=args.env_id, no_proprio=args.no_proprio,
            action_horizon=action_horizon,
        )
    else:
        # normal eval and abs-joint-pos use the same loop — only control_mode differs
        print(f"\n{mode_label}: {args.env_id} | {num_eval_steps} steps | {args.num_envs} envs")
        results = run_eval(
            env, inner_env, policy, normalizer, args.num_envs, num_eval_steps, device, args.seed,
            output_dir=args.output_dir, env_id=args.env_id, no_proprio=args.no_proprio,
            action_horizon=action_horizon,
        )

    env.close()

    print(f"\n{'='*50}")
    n = int(results.pop("num_episodes"))
    print(f"{mode_label}: {n} episodes in {num_eval_steps} steps "
          f"({num_eval_steps * args.num_envs} transitions across {args.num_envs} envs)")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
    print(f"{'='*50}")

    if args.remote_server:
        results["num_episodes"] = n
        policy.report_results(results)
        policy.cleanup()


if __name__ == "__main__":
    main()
