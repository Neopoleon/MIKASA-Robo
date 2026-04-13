# Will export all tracked (non-gitignored) files into a MIKASA-Robo-release folder

import os
import subprocess
import shutil
import click


@click.command()
@click.argument("output_root_dir", type=str, default=".")
def export_release_files(output_root_dir: str):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Get all tracked files via git (respects .gitignore)
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    tracked_files = [f for f in result.stdout.strip().split("\n") if f]

    # Files to exclude from export (relative paths)
    exclude_files = {
        "scripts/export_release_files.py",
    }

    # Directories to exclude from export
    exclude_dirs = set()

    # Post-processing: remove lines containing these substrings
    # Maps relative file path -> list of substrings to filter out
    line_removal_filters = {}

    # Full-line replacements applied to exported files
    # Maps relative file path -> list of (old, new) tuples
    line_replacements = {}

    if output_root_dir == ".":
        output_dir = os.path.join(repo_root, "mikasa-robo-env-release")
    else:
        output_dir = os.path.join(output_root_dir, "mikasa-robo-env-release")

    # Clean and recreate output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    copied = 0
    for rel_path in tracked_files:
        if rel_path in exclude_files:
            continue
        if any(rel_path.startswith(d + "/") for d in exclude_dirs):
            continue

        src = os.path.join(repo_root, rel_path)
        if not os.path.isfile(src):
            print(f"WARNING: Tracked file not found: {rel_path}")
            continue

        dst = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1

    # Apply post-processing to remove personal information
    for rel_path, filters in line_removal_filters.items():
        dst = os.path.join(output_dir, rel_path)
        if not os.path.exists(dst):
            continue
        with open(dst, "r") as f:
            lines = f.readlines()
        filtered = [l for l in lines if not any(s in l for s in filters)]
        with open(dst, "w") as f:
            f.writelines(filtered)
        print(f"Filtered personal info from {rel_path}")

    for rel_path, replacements in line_replacements.items():
        dst = os.path.join(output_dir, rel_path)
        if not os.path.exists(dst):
            continue
        with open(dst, "r") as f:
            content = f.read()
        for old, new in replacements:
            content = content.replace(old, new)
        with open(dst, "w") as f:
            f.write(content)
        print(f"Applied replacements to {rel_path}")

    print(f"\nExported {copied} files to {output_dir}")


if __name__ == "__main__":
    export_release_files()