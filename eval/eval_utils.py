"""shared utilities for MIKASA-Robo evaluation and replay.

obs:    2 rgb cameras (third-person + wrist, 256x256) + 8d proprio (7 arm qpos + 1 gripper).
action: 8d joint-space — deltas (pd_joint_delta_pos), absolute qpos (pd_joint_pos), or teleport (direct-qpos).
policy: diffusion transformer (SigLIP2 + memory transformer), predicts action chunks.
"""

import json
import os
import sys
from dataclasses import dataclass, field

import cv2
import gymnasium as gym
import imageio
import numpy as np
import torch
import yaml

sys.path.insert(0, "/home/jeff/imitation-learning-policies")
sys.path.insert(0, "/home/jeff/robot-utils/src")

from omegaconf import OmegaConf
import hydra

import mani_skill.envs  # noqa: F401 — registers ManiSkill envs
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from mikasa_robo_suite.memory_envs import *  # noqa: F403 — registers MIKASA envs
from mikasa_robo_suite.utils.wrappers import *  # noqa: F403
from baselines.ppo.ppo_memtasks import FlattenRGBDObservationWrapper

from imitation_learning.common.dataclasses import construct_data_meta_dict
from imitation_learning.datasets.normalizer import FixedNormalizer
from imitation_learning.policies.base_policy import BasePolicy
from robot_utils.config_utils import register_resolvers

import warnings
warnings.filterwarnings(
    "ignore", message=".*env\\.\\w+ to get variables from other wrappers is deprecated.*"
)

register_resolvers()


# task registry — add new tasks here.
# wrappers applied: _base_wrappers → extra_wrappers → FlattenRGBD → FlattenAction → VecEnv.

WrapperSpec = tuple[type, dict]


@dataclass
class TaskConfig:
    episode_steps: int
    extra_wrappers: list[WrapperSpec] = field(default_factory=list)


def _base_wrappers() -> list[WrapperSpec]:
    return [
        (StateOnlyTensorToDictWrapper, {}),            # raw tensor → dict with 'prompt'/'oracle_info' (required by FlattenRGBD)
        (InitialZeroActionWrapper, {"n_initial_steps": 0}),
        (RenderStepInfoWrapper, {}),
        (RenderRewardInfoWrapper, {}),
    ]


TASK_REGISTRY: dict[str, TaskConfig] = {
    # shell game (90 steps)
    "ShellGameTouch-v0": TaskConfig(episode_steps=90, extra_wrappers=[(ShellGameRenderCupInfoWrapper, {})]),
    "ShellGamePush-v0":  TaskConfig(episode_steps=90, extra_wrappers=[(ShellGameRenderCupInfoWrapper, {})]),
    "ShellGamePick-v0":  TaskConfig(episode_steps=90, extra_wrappers=[(ShellGameRenderCupInfoWrapper, {})]),
    # intercept (90 steps)
    "InterceptSlow-v0":     TaskConfig(episode_steps=90),
    "InterceptMedium-v0":   TaskConfig(episode_steps=90),
    "InterceptFast-v0":     TaskConfig(episode_steps=90),
    "InterceptGrabSlow-v0":   TaskConfig(episode_steps=90),
    "InterceptGrabMedium-v0": TaskConfig(episode_steps=90),
    "InterceptGrabFast-v0":   TaskConfig(episode_steps=90),
    # rotate (90 steps)
    "RotateStrictPos-v0":      TaskConfig(episode_steps=90, extra_wrappers=[(RotateRenderAngleInfoWrapper, {})]),
    "RotateStrictPosNeg-v0":   TaskConfig(episode_steps=90, extra_wrappers=[(RotateRenderAngleInfoWrapper, {})]),
    "RotateLenientPos-v0":     TaskConfig(episode_steps=90, extra_wrappers=[(RotateRenderAngleInfoWrapper, {})]),
    "RotateLenientPosNeg-v0":  TaskConfig(episode_steps=90, extra_wrappers=[(RotateRenderAngleInfoWrapper, {})]),
    # take it back (180 steps)
    "TakeItBack-v0": TaskConfig(episode_steps=180),
    # remember color (60 steps)
    "RememberColor3-v0": TaskConfig(episode_steps=60, extra_wrappers=[(RememberColorInfoWrapper, {})]),
    "RememberColor5-v0": TaskConfig(episode_steps=60, extra_wrappers=[(RememberColorInfoWrapper, {})]),
    "RememberColor6-v0": TaskConfig(episode_steps=60, extra_wrappers=[(RememberColorInfoWrapper, {})]),
    "RememberColor9-v0": TaskConfig(episode_steps=60, extra_wrappers=[(RememberColorInfoWrapper, {})]),
    # remember shape (60 steps)
    "RememberShape3-v0": TaskConfig(episode_steps=60, extra_wrappers=[(RememberShapeInfoWrapper, {})]),
    "RememberShape5-v0": TaskConfig(episode_steps=60, extra_wrappers=[(RememberShapeInfoWrapper, {})]),
    "RememberShape9-v0": TaskConfig(episode_steps=60, extra_wrappers=[(RememberShapeInfoWrapper, {})]),
    # remember shape + color (60 steps)
    "RememberShapeAndColor3x2-v0": TaskConfig(episode_steps=60, extra_wrappers=[(RememberShapeAndColorInfoWrapper, {})]),
    "RememberShapeAndColor3x3-v0": TaskConfig(episode_steps=60, extra_wrappers=[(RememberShapeAndColorInfoWrapper, {})]),
    "RememberShapeAndColor5x3-v0": TaskConfig(episode_steps=60, extra_wrappers=[(RememberShapeAndColorInfoWrapper, {})]),
    # color sequence tasks (120 steps)
    "BunchOfColors3-v0": TaskConfig(episode_steps=120),
    "BunchOfColors5-v0": TaskConfig(episode_steps=120),
    "BunchOfColors7-v0": TaskConfig(episode_steps=120),
    "SeqOfColors3-v0":   TaskConfig(episode_steps=120),
    "SeqOfColors5-v0":   TaskConfig(episode_steps=120),
    "SeqOfColors7-v0":   TaskConfig(episode_steps=120),
    "ChainOfColors3-v0": TaskConfig(episode_steps=120),
    "ChainOfColors5-v0": TaskConfig(episode_steps=120),
    "ChainOfColors7-v0": TaskConfig(episode_steps=120),
}


# policy loading

def load_policy(checkpoint_path: str, device: torch.device) -> tuple[BasePolicy, FixedNormalizer]:
    """load policy + normalizer from checkpoint.
    ckpt keys: model_state_dict, normalizer_state_dict, cfg_str_unresolved (hydra yaml).
    """
    print(f"loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    cfg = OmegaConf.create(ckpt["cfg_str_unresolved"])
    # Save config to a yaml file in the same path as the checkpoint
    with open(checkpoint_path.replace(".ckpt", ".yaml"), "w") as f:
        yaml.dump(OmegaConf.to_container(cfg), f)
    print(f"Saved config to: {checkpoint_path.replace('.ckpt', '.yaml')}")
    policy = hydra.utils.instantiate(cfg["workspace"]["model"])
    assert isinstance(policy, BasePolicy)
    policy.load_state_dict(ckpt["model_state_dict"])
    policy.to(device)
    policy.eval()
    policy.reset()

    data_meta = construct_data_meta_dict(cfg["workspace"]["train_dataset"]["output_data_meta"])
    normalizer = FixedNormalizer(data_meta)
    normalizer.to(device)
    normalizer.load_state_dict(ckpt["normalizer_state_dict"])

    return policy, normalizer


# environment

def make_eval_env(
    env_id: str,
    task_cfg: TaskConfig,
    num_envs: int,
    num_eval_steps: int,
    capture_video: bool,
    output_dir: str,
    seed: int,
    reward_mode: str,
    control_mode: str = "pd_joint_delta_pos",
) -> tuple[ManiSkillVectorEnv, gym.Env]:
    """create gpu-vectorized env with wrapper stack.
    returns (vec_env, inner_env) — inner_env.render() has wrapper overlays,
    vec_env.render() bypasses them.
    control_mode: 'pd_joint_delta_pos' (default) or 'pd_joint_pos' (absolute target qpos).
    """
    env_kwargs = dict(
        obs_mode="rgb",
        control_mode=control_mode,
        render_mode="all" if capture_video else "rgb_array",
        sim_backend="gpu",
        reward_mode=reward_mode,
    )
    env = gym.make(env_id, num_envs=num_envs, reconfiguration_freq=1, **env_kwargs)

    for wrapper_cls, wrapper_kwargs in _base_wrappers() + task_cfg.extra_wrappers:
        env = wrapper_cls(env, **wrapper_kwargs)

    # → {rgb: (B,H,W,6) uint8, joints: (B,25) float}
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=False, joints=True)

    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)

    inner_env = env  # keep ref for render() with overlays

    vec_env = ManiSkillVectorEnv(env, num_envs, ignore_terminations=True, record_metrics=True)
    return vec_env, inner_env


# obs → policy input

def obs_to_policy_input(
    obs: dict[str, torch.Tensor],
    num_envs: int,
    device: torch.device,
    no_proprio: bool = False,
) -> dict[str, torch.Tensor]:
    """env obs → policy input dict.
    rgb (B,H,W,6): [:3]=third_person, [3:]=wrist → (B,1,3,H,W) float [0,1] each.
    joints (B,25): [7:14]=arm_qpos, [14:15]=gripper_qpos → robot0_8d (B,1,8).
    episode_idx (B,): static arange for MemoryTransformer per-env history tracking.
    if no_proprio: omit robot0_8d (vision-only policy).
    """
    rgb = obs["rgb"]
    joints = obs["joints"]

    # # Downscale 256→128 then upscale 128→256 to match training resolution
    # if rgb.shape[1] == 256 and rgb.shape[2] == 256:
    #     import kornia
    #     print("\033[96mResizing obs rgb from 256x256 → 128x128 → 256x256\033[0m")
    #     # kornia expects (B, C, H, W) float
    #     rgb_f = rgb.float().permute(0, 3, 1, 2)  # (B, 6, 256, 256)
    #     down = kornia.geometry.transform.resize(rgb_f, (128, 128), antialias=True)
    #     up = kornia.geometry.transform.resize(down, (256, 256), antialias=True)
    #     rgb = up.permute(0, 2, 3, 1).to(rgb.dtype)  # (B, 256, 256, 6)
    #     print("\033[96mResized obs rgb\033[0m")

    third_person = rgb[..., :3].float().div_(255.0).permute(0, 3, 1, 2).unsqueeze(1)
    wrist_cam    = rgb[..., 3:].float().div_(255.0).permute(0, 3, 1, 2).unsqueeze(1)

    result = {
        "third_person_camera": third_person.to(device),
        "robot0_wrist_camera": wrist_cam.to(device),
        "episode_idx":         torch.arange(num_envs, device=device),
    }
    if not no_proprio:
        proprio = torch.cat([joints[:, 7:14], joints[:, 14:15]], dim=-1).unsqueeze(1)
        result["robot0_8d"] = proprio.to(device)
    return result


# direct qpos teleportation (perfect controller mode)

def teleport_and_step(env, predicted_8d: torch.Tensor):
    """teleport robot to predicted absolute joint positions and advance the env by one step.
    NO physics simulation — perfect controller assumption means we just set qpos directly.

    we monkey-patch scene.step to a no-op during env.step(None) so that:
    - gravity doesn't pull objects hidden at z=1000 during memory phases
    - the full wrapper chain (FlattenRGBD, task wrappers, etc.) still processes obs normally
    - evaluate() still runs to handle shape visibility / reward / termination

    predicted_8d: (B, 8) — 7 arm qpos + 1 gripper qpos.
    panda has 9 DOF (7 arm + 2 gripper fingers), so we mirror the single gripper value.

    returns: (obs, reward, terminated, truncated, info) — same as env.step().
    """
    base_env = env.unwrapped
    robot = base_env.agent.robot

    # build full 9-dof qpos from predicted 8d
    qpos = robot.get_qpos()                            # (B, 9)
    qpos[:, :7] = predicted_8d[:, :7]                   # arm joints
    qpos[:, 7:9] = predicted_8d[:, 7:8].expand(-1, 2)   # mirror gripper to both fingers

    # teleport: set qpos + zero velocity, sync GPU
    robot.set_qpos(qpos)
    robot.set_qvel(torch.zeros_like(qpos))
    base_env.scene._gpu_apply_all()
    base_env.scene.px.gpu_update_articulation_kinematics()
    base_env.scene._gpu_fetch_all()

    # fix shape visibility: evaluate() captures original_poses from current cube/shape
    # positions at the TOP of evaluate(). during the hiding phase shapes are at z=1000,
    # so original_poses gets z=1000 too. when the reappear phase triggers, evaluate()
    # restores from original_poses — which is still z=1000. with real physics the shapes
    # would fall back, but we disabled physics. fix: before evaluate() runs, reset all
    # shapes/cubes to their initial_poses (table-level z) so original_poses is correct.
    initial_poses = getattr(base_env, 'initial_poses', None)
    if initial_poses is not None:
        # remember_color uses self.cubes, remember_shape uses self.shapes
        for actor_dict in [getattr(base_env, 'cubes', {}), getattr(base_env, 'shapes', {})]:
            for key in initial_poses:
                if key in actor_dict:
                    pose = actor_dict[key].pose.raw_pose.clone()
                    pose[:, :3] = initial_poses[key]
                    actor_dict[key].pose = pose
        base_env.scene._gpu_apply_all()
        base_env.scene._gpu_fetch_all()

    # disable physics sim but keep GPU sync: replace scene.step with apply+fetch
    # so that pose changes from evaluate() are visible to get_obs() camera rendering,
    # but no forces/gravity pull objects from their correct positions.
    scene = base_env.scene
    original_step = scene.step

    def _sync_only():
        scene._gpu_apply_all()
        scene.px.gpu_update_articulation_kinematics()
        scene._gpu_fetch_all()

    scene.step = _sync_only
    try:
        result = env.step(None)
    finally:
        scene.step = original_step

    return result


# inference

ACTION_HORIZON = 2  # actions per prediction, matches training chunk size


@torch.no_grad()
def predict_action(
    obs: dict[str, torch.Tensor],
    policy,
    normalizer,
    num_envs: int,
    device: torch.device,
    no_proprio: bool = False,
    action_horizon: int = ACTION_HORIZON,
) -> torch.Tensor:
    """obs → normalize → diffusion denoise → unnormalize → action chunk (B, action_horizon, 8).
    auto-detects output key: action0_8d (action-space) or robot0_8d (state-space / direct-qpos).
    """
    # lazy import: remote client uses its own action_horizon and no_proprio from server config
    from remote_policy_client import RemotePolicyClient
    if isinstance(policy, RemotePolicyClient):
        return policy.predict_action(obs, num_envs, device, no_proprio)

    batch = obs_to_policy_input(obs, num_envs, device, no_proprio=no_proprio)
    batch = normalizer.normalize(batch)
    action_dict = policy.predict_action(batch)
    action_dict = normalizer.unnormalize(action_dict)
    output_key = "action0_8d" if "action0_8d" in action_dict else "robot0_8d"
    return action_dict[output_key][:, :action_horizon, :]


# frame annotation helpers

def _put_text(frame, text, pos, font_scale=0.5, thickness=1, color=(255, 255, 255)):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def _annotate_frames(frames, actions, rewards, font_scale=0.35, thickness=1):
    """overlay step, reward, and action on each frame. for obs side-by-side (256x512)."""
    for i in range(len(frames)):
        y = 15
        _put_text(frames[i], f"step: {i}", (5, y), font_scale, thickness)
        y += 18
        _put_text(frames[i], f"rew: {rewards[i]:.3f}", (5, y), font_scale, thickness)
        y += 18
        act = actions[i]
        _put_text(frames[i], f"act: [{', '.join(f'{a:.2f}' for a in act)}]", (5, y), font_scale, thickness)
    return frames


def _annotate_action_on_frames(frames, actions, font_scale=0.5, thickness=1):
    """overlay action text on render frames (512x512). other overlays already applied by wrappers."""
    for i in range(len(frames)):
        act = actions[i]
        # split 8d action into two lines to fit within 512px width
        _put_text(frames[i], f"act[0:4]: [{', '.join(f'{a:.2f}' for a in act[:4])}]", (10, 150), font_scale, thickness)
        _put_text(frames[i], f"act[4:8]: [{', '.join(f'{a:.2f}' for a in act[4:])}]", (10, 175), font_scale, thickness)
    return frames


def _divergence_stats(ea, pa):
    """compute intuitive divergence metrics between expert and policy action vectors.
    returns (cosine_sim %, relative_err %) — cosine_sim: direction agreement (100%=identical),
    relative_err: mean |diff| / mean |expert| * 100 (0%=perfect match).
    """
    dot = float(np.dot(ea, pa))
    norm_e, norm_p = float(np.linalg.norm(ea)), float(np.linalg.norm(pa))
    cos_sim = dot / max(norm_e * norm_p, 1e-8) * 100.0
    rel_err = float(np.mean(np.abs(ea - pa))) / max(float(np.mean(np.abs(ea))), 1e-8) * 100.0
    return cos_sim, rel_err


def _annotate_replay_obs_frames(frames, expert_actions, policy_actions, rewards):
    """overlay expert (green) vs policy (red) actions + divergence on obs side-by-side frames (256x512)."""
    fs, th = 0.32, 1
    for i in range(len(frames)):
        y = 14
        _put_text(frames[i], f"step:{i}  rew:{rewards[i]:.3f}", (4, y), fs, th)
        ea, pa = expert_actions[i], policy_actions[i]
        cos_sim, rel_err = _divergence_stats(ea, pa)
        y += 16
        _put_text(frames[i], f"expert: [{', '.join(f'{a:.2f}' for a in ea)}]", (4, y), fs, th, color=(0, 255, 0))
        y += 16
        _put_text(frames[i], f"policy: [{', '.join(f'{a:.2f}' for a in pa)}]", (4, y), fs, th, color=(0, 0, 255))
        y += 16
        _put_text(frames[i], f"cos:{cos_sim:.0f}%  relErr:{rel_err:.0f}%", (4, y), fs, th, color=(255, 255, 0))
    return frames


def _annotate_replay_render_frames(frames, expert_actions, policy_actions):
    """overlay expert (green) vs policy (red) actions + divergence on render frames (512x512).
    step/reward/task overlays already applied by wrappers."""
    fs, th = 0.45, 1
    for i in range(len(frames)):
        ea, pa = expert_actions[i], policy_actions[i]
        cos_sim, rel_err = _divergence_stats(ea, pa)
        y = 145
        _put_text(frames[i], f"expert[0:4]: [{', '.join(f'{a:.2f}' for a in ea[:4])}]", (8, y), fs, th, color=(0, 255, 0))
        y += 22
        _put_text(frames[i], f"expert[4:8]: [{', '.join(f'{a:.2f}' for a in ea[4:])}]", (8, y), fs, th, color=(0, 255, 0))
        y += 22
        _put_text(frames[i], f"policy[0:4]: [{', '.join(f'{a:.2f}' for a in pa[:4])}]", (8, y), fs, th, color=(0, 0, 255))
        y += 22
        _put_text(frames[i], f"policy[4:8]: [{', '.join(f'{a:.2f}' for a in pa[4:])}]", (8, y), fs, th, color=(0, 0, 255))
        y += 22
        _put_text(frames[i], f"cos:{cos_sim:.0f}%  relErr:{rel_err:.0f}%", (8, y), fs, th, color=(255, 255, 0))
    return frames


def _save_video_and_frames(episode_dir, name, frames, fps):
    """save video and individual pngs under episode_dir/{name}."""
    imageio.mimsave(os.path.join(episode_dir, f"video_{name}.mp4"), frames, fps=fps)
    # frames_dir = os.path.join(episode_dir, f"frames_{name}")
    # os.makedirs(frames_dir, exist_ok=True)
    # for i, frame in enumerate(frames):
    #     imageio.imwrite(os.path.join(frames_dir, f"frame_{i:03d}.png"), frame)


# episode recording

def save_episode(
    episode_dir: str,
    rgb_frames: list[np.ndarray],
    joints_frames: list[np.ndarray],
    actions: list[np.ndarray],
    rewards: list[float],
    success: bool,
    ep_metrics: dict,
    render_frames: list[np.ndarray] | None = None,
    fps: int = 30,
):
    """save episode.npz (clean obs) + video/pngs (with overlays if available) + metrics json."""
    os.makedirs(episode_dir, exist_ok=True)
    T = len(rgb_frames)

    rgb = np.stack(rgb_frames)
    joints = np.stack(joints_frames)
    action = np.stack(actions)
    reward = np.array(rewards, dtype=np.float32)
    success_arr = np.full(T, int(success), dtype=np.int64)
    done = np.zeros(T, dtype=np.int64)
    done[-1] = 1

    np.savez_compressed(
        os.path.join(episode_dir, "episode.npz"),
        rgb=rgb, joints=joints, action=action,
        reward=reward, success=success_arr, done=done,
    )

    # raw obs: side-by-side two-camera view (T,H,2W,3)
    obs_frames = np.concatenate([rgb[..., :3], rgb[..., 3:]], axis=2)

    # overlay text on both obs and render frames
    obs_annotated = _annotate_frames(obs_frames.copy(), action, reward, font_scale=0.35, thickness=1)
    _save_video_and_frames(episode_dir, "obs", obs_annotated, fps)

    if render_frames:
        render_arr = np.stack(render_frames)
        # render already has step/reward/task overlays from wrappers; add action overlay
        render_annotated = _annotate_action_on_frames(render_arr.copy(), action, font_scale=0.7, thickness=2)
        _save_video_and_frames(episode_dir, "render", render_annotated, fps)

    with open(os.path.join(episode_dir, "metrics.json"), "w") as f:
        json.dump(ep_metrics, f, indent=2)


# per-env episode buffers

class EpisodeBuffers:
    """per-env rolling buffers. append() every step, flush() on episode completion."""

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.rgb = [[] for _ in range(num_envs)]
        self.joints = [[] for _ in range(num_envs)]
        self.actions = [[] for _ in range(num_envs)]
        self.rewards = [[] for _ in range(num_envs)]
        self.render = [[] for _ in range(num_envs)]  # overlaid frames from env.render()

    def append_obs(self, obs: dict[str, torch.Tensor], render_frame=None):
        rgb_cpu = obs["rgb"].cpu().numpy()
        joints_cpu = obs["joints"].cpu().numpy()
        for i in range(self.num_envs):
            self.rgb[i].append(rgb_cpu[i])
            self.joints[i].append(joints_cpu[i])
        if render_frame is not None:
            if isinstance(render_frame, torch.Tensor):
                render_frame = render_frame.cpu().numpy()
            for i in range(self.num_envs):
                self.render[i].append(render_frame[i])

    def append_action(self, action: torch.Tensor):
        action_cpu = action.cpu().numpy()
        for i in range(self.num_envs):
            self.actions[i].append(action_cpu[i])

    def append_reward(self, reward: torch.Tensor):
        reward_cpu = reward.cpu().numpy()
        for i in range(self.num_envs):
            self.rewards[i].append(float(reward_cpu[i]))

    def flush(self, env_idx: int) -> tuple[list, list, list, list, list]:
        data = (
            list(self.rgb[env_idx]),
            list(self.joints[env_idx]),
            list(self.actions[env_idx]),
            list(self.rewards[env_idx]),
            list(self.render[env_idx]),
        )
        self.rgb[env_idx].clear()
        self.joints[env_idx].clear()
        self.actions[env_idx].clear()
        self.rewards[env_idx].clear()
        self.render[env_idx].clear()
        return data


# episode completion handling

def handle_episode_completions(
    infos: dict,
    buffers: EpisodeBuffers,
    num_envs: int,
    num_episodes: int,
    task_dir: str,
    metrics: dict[str, list],
    env=None,
) -> int:
    """save completed episodes to disk. returns updated episode count.
    ManiSkillVectorEnv sets infos["_final_info"] (B,) bool mask and
    infos["final_info"]["episode"] dict of (B,) metric tensors on episode end.
    """
    if "final_info" not in infos:
        return num_episodes

    mask = infos["_final_info"]
    ep_data = infos["final_info"]["episode"]

    for k, v in ep_data.items():
        metrics[k].append(v)

    for i in range(num_envs):
        if not mask[i]:
            continue

        ep_metrics = {k: float(v[i]) for k, v in ep_data.items()}
        success = ep_metrics.get("success_once", 0.0) > 0.5
        tag = "success" if success else "failure"
        ep_metrics["episode_idx"] = num_episodes
        if env is not None:
            seeds = getattr(env.unwrapped, '_episode_seed', None)
            if seeds is not None:
                ep_metrics["env_seed"] = int(seeds[i])

        rgb_frames, joints_frames, actions, rewards, render_frames = buffers.flush(i)
        save_episode(
            episode_dir=os.path.join(task_dir, f"episode_{num_episodes:04d}_{tag}"),
            rgb_frames=rgb_frames,
            joints_frames=joints_frames,
            actions=actions,
            rewards=rewards,
            success=success,
            ep_metrics=ep_metrics,
            render_frames=render_frames,
        )
        num_episodes += 1

    return num_episodes
