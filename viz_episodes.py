"""Visualize zarr episodes as side-by-side MP4s (third_person_camera | wrist_camera)."""

import argparse
import os
import subprocess
import numpy as np
import zarr
from tqdm import tqdm


def render_episode(zarr_group, episode_key, output_path, fps=15):
    ep = zarr_group[episode_key]
    third = ep["third_person_camera"][:]  # (T, 256, 256, 3)
    wrist = ep["robot0_wrist_camera"][:]  # (T, 256, 256, 3)

    frames = np.concatenate([third, wrist], axis=2)  # (T, H, W*2, 3)
    T, H, W, _ = frames.shape

    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{W}x{H}", "-pix_fmt", "rgb24", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
        "-movflags", "+faststart", "-an", output_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    proc.stdin.write(frames.tobytes())
    proc.stdin.close()
    proc.wait()


def main():
    parser = argparse.ArgumentParser(description="Render zarr episodes to side-by-side MP4s")
    parser.add_argument(
        "--zarr_path",
        type=str,
        default="/storage/ssd1/jeff/mikasa_zarr_processed/RememberColor9-v0/episode_data.zarr",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Override output dir. Default: save next to the zarr.")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument(
        "--episodes",
        type=str,
        default=None,
        help="Comma-separated episode indices (e.g. '0,1,5') or -1 for all. Default: all.",
    )
    args = parser.parse_args()

    z = zarr.open_group(args.zarr_path, mode="r")

    episode_keys = sorted(z.keys(), key=lambda k: int(k.split("_")[1]))

    if args.episodes is not None and args.episodes != "-1":
        selected = {int(x) for x in args.episodes.split(",")}
        episode_keys = [k for k in episode_keys if int(k.split("_")[1]) in selected]

    for key in tqdm(episode_keys, desc="Rendering episodes"):
        ep_dir = os.path.join(args.zarr_path, key)
        out_path = os.path.join(ep_dir, "viz.mp4")
        render_episode(z, key, out_path, fps=args.fps)

    print(f"Done. {len(episode_keys)} videos saved under {args.zarr_path}/<episode>/viz.mp4")


if __name__ == "__main__":
    main()