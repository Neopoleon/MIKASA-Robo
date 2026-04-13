"""Convert delta actions to absolute target joint positions for all zarr episodes.

Reads from input dir, writes converted zarr to a separate output dir.
"""
import argparse
import numpy as np
import zarr
from pathlib import Path
from tqdm import tqdm


def convert_task(src_zarr_path: str, dst_zarr_path: str):
    src = zarr.open(src_zarr_path, mode="r")
    dst = zarr.open(dst_zarr_path, mode="w", zarr_format=2)
    dst.attrs.update(dict(src.attrs))

    ep_keys = sorted([k for k in src.keys() if k.startswith("episode_")],
                     key=lambda x: int(x.split("_")[1]))
    for ep_key in tqdm(ep_keys, desc=Path(src_zarr_path).parent.name):
        src_ep = src[ep_key]
        dst_ep = dst.create_group(ep_key)
        dst_ep.attrs.update(dict(src_ep.attrs))

        robot = src_ep["robot0_8d"][:]
        delta = src_ep["action0_8d"][:]
        abs_action = robot + np.clip(delta, -1, 1) * 0.1

        # Copy all arrays, replacing action with converted version
        for key in src_ep.keys():
            if key == "action0_8d":
                dst_ep.create_array(key, data=abs_action, chunks=src_ep[key].chunks)
            else:
                data = src_ep[key][:]
                dst_ep.create_array(key, data=data, chunks=src_ep[key].chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Root dir with task folders (e.g. /storage/ssd1/jeff/mikasa_zarr_unprocessed)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output root dir (e.g. /storage/ssd1/jeff/mikasa_zarr_processed)")
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    task_dirs = sorted(Path(args.input_dir).iterdir())
    for task_dir in task_dirs:
        zarr_path = task_dir / "episode_data.zarr"
        if zarr_path.exists():
            dst_path = out_root / task_dir.name / "episode_data.zarr"
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Converting {task_dir.name}...")
            convert_task(str(zarr_path), str(dst_path))
    print("Done.")