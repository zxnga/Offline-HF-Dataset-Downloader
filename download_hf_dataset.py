import argparse
import json
import os
import tarfile
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_tar_gz(source_dir: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(source_dir, arcname=source_dir.name)


def download_prepared_dataset(cfg: dict) -> None:
    from datasets import load_dataset

    dataset_id = cfg["dataset_id"]
    config_name = cfg.get("config_name")
    split = cfg.get("split")
    revision = cfg.get("revision")
    output_dir = Path(cfg["output_dir"])
    token = os.environ.get(cfg.get("token_env", "HF_TOKEN"))
    trust_remote_code = bool(cfg.get("trust_remote_code", False))

    output_dir.mkdir(parents=True, exist_ok=True)

    kwargs = {
        "path": dataset_id,
        "revision": revision,
        "token": token,
        "trust_remote_code": trust_remote_code,
    }

    if config_name is not None:
        kwargs["name"] = config_name

    if split is not None:
        kwargs["split"] = split

    # Remove empty values so public datasets work without a token.
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    print(f"Downloading dataset with datasets.load_dataset(): {dataset_id}")
    ds = load_dataset(**kwargs)

    print(f"Saving prepared dataset to: {output_dir}")
    ds.save_to_disk(str(output_dir))

    manifest = {
        "dataset_id": dataset_id,
        "config_name": config_name,
        "split": split,
        "revision": revision,
        "mode": "prepared",
        "load_offline_with": "datasets.load_from_disk",
    }

    with open(output_dir / "offline_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Prepared dataset saved successfully.")


def download_raw_dataset_repo(cfg: dict) -> None:
    from huggingface_hub import snapshot_download

    dataset_id = cfg["dataset_id"]
    revision = cfg.get("revision")
    output_dir = Path(cfg["output_dir"])
    token = os.environ.get(cfg.get("token_env", "HF_TOKEN"))

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading raw dataset repo files: {dataset_id}")
    snapshot_download(
        repo_id=dataset_id,
        repo_type="dataset",
        revision=revision,
        local_dir=str(output_dir),
        token=token,
    )

    manifest = {
        "dataset_id": dataset_id,
        "revision": revision,
        "mode": "raw",
        "load_offline_with": "datasets.load_dataset using local files",
    }

    with open(output_dir / "offline_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Raw dataset files saved successfully.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="dataset_config.yaml",
        help="Path to dataset YAML config.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    mode = cfg.get("mode", "prepared")
    output_dir = Path(cfg["output_dir"])
    archive_path = Path(cfg["archive_path"])

    if mode == "prepared":
        download_prepared_dataset(cfg)
    elif mode == "raw":
        download_raw_dataset_repo(cfg)
    else:
        raise ValueError("mode must be either 'prepared' or 'raw'")

    print(f"Creating archive: {archive_path}")
    make_tar_gz(output_dir, archive_path)

    print("\nDone.")
    print(f"Transfer this file to the offline machine:\n  {archive_path}")


if __name__ == "__main__":
    main()