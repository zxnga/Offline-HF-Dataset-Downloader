import argparse
import json
import os
import re
import tarfile
from pathlib import Path


DEFAULT_MODE = "prepared"
VALID_MODES = {"prepared", "raw"}
DEFAULT_FALLBACK_TO_RAW = False


def load_config(path: str) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_tar_gz(source_dir: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_arcname = None

    try:
        archive_relative = archive_path.resolve().relative_to(source_dir.resolve())
        archive_arcname = (Path(source_dir.name) / archive_relative).as_posix()
    except ValueError:
        pass

    def exclude_archive(tarinfo: tarfile.TarInfo):
        if archive_arcname is not None and tarinfo.name == archive_arcname:
            return None
        return tarinfo

    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(source_dir, arcname=source_dir.name, filter=exclude_archive)


def dataset_folder_name(dataset_id: str) -> str:
    name = dataset_id.rstrip("/").split("/")[-1]
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")

    if not name:
        raise ValueError(f"Could not derive a folder name from dataset_id: {dataset_id!r}")

    return name


def resolve_dataset_mode(cfg: dict, label: str) -> None:
    mode = cfg.get("mode", DEFAULT_MODE)
    if mode not in VALID_MODES:
        valid_modes = ", ".join(f"{mode!r}" for mode in sorted(VALID_MODES))
        raise ValueError(f"{label} mode must be one of {valid_modes}; got {mode!r}")

    cfg["mode"] = mode


def resolve_fallback_to_raw(cfg: dict, label: str) -> None:
    fallback_to_raw = cfg.get("fallback_to_raw", DEFAULT_FALLBACK_TO_RAW)
    if not isinstance(fallback_to_raw, bool):
        raise ValueError(f"{label} fallback_to_raw must be true or false")

    cfg["fallback_to_raw"] = fallback_to_raw


def resolve_dataset_paths(cfg: dict, label: str) -> dict:
    cfg = cfg.copy()
    resolve_dataset_mode(cfg, label)
    resolve_fallback_to_raw(cfg, label)

    dataset_id = cfg["dataset_id"]
    folder_name = dataset_folder_name(dataset_id)
    base_output_dir = Path(cfg.get("output_dir", "./offline_datasets"))

    # Preserve existing configs that already point directly at the dataset folder.
    if base_output_dir.name == folder_name:
        output_dir = base_output_dir
    else:
        output_dir = base_output_dir / folder_name

    archive_path = cfg.get("archive_path")
    if archive_path is None:
        archive_path = output_dir / f"{folder_name}.tar.gz"
    else:
        archive_path = Path(archive_path)

    cfg["output_dir"] = str(output_dir)
    cfg["archive_path"] = str(archive_path)

    return cfg


def iter_dataset_configs(config: dict) -> list[dict]:
    if "datasets" not in config:
        if "dataset_id" not in config:
            raise ValueError("config is missing required field: dataset_id")

        return [resolve_dataset_paths(config, "config")]

    datasets = config["datasets"]
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("datasets must be a non-empty list")

    defaults = {key: value for key, value in config.items() if key != "datasets"}
    resolved_configs = []

    for index, dataset_cfg in enumerate(datasets, start=1):
        if not isinstance(dataset_cfg, dict):
            raise ValueError(f"datasets[{index}] must be a mapping")

        merged_cfg = defaults.copy()
        merged_cfg.update(dataset_cfg)

        if "dataset_id" not in merged_cfg:
            raise ValueError(f"datasets[{index}] is missing required field: dataset_id")

        resolved_configs.append(resolve_dataset_paths(merged_cfg, f"datasets[{index}]"))

    output_dirs = [cfg["output_dir"] for cfg in resolved_configs]
    if len(output_dirs) != len(set(output_dirs)):
        raise ValueError("Multiple datasets resolve to the same output_dir")

    archive_paths = [cfg["archive_path"] for cfg in resolved_configs]
    if len(archive_paths) != len(set(archive_paths)):
        raise ValueError("Multiple datasets resolve to the same archive_path")

    return resolved_configs


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
    fallback_from_mode = cfg.get("fallback_from_mode")

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

    if fallback_from_mode is not None:
        manifest["fallback_from_mode"] = fallback_from_mode

    with open(output_dir / "offline_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Raw dataset files saved successfully.")


def download_dataset(cfg: dict) -> None:
    mode = cfg["mode"]

    if mode == "raw":
        download_raw_dataset_repo(cfg)
        return

    try:
        download_prepared_dataset(cfg)
    except Exception as prepared_error:
        if not cfg["fallback_to_raw"]:
            raise

        print(
            "Prepared mode failed; fallback_to_raw is enabled, "
            "so trying raw mode instead."
        )
        print(f"Prepared error: {type(prepared_error).__name__}: {prepared_error}")

        fallback_cfg = cfg.copy()
        fallback_cfg["mode"] = "raw"
        fallback_cfg["fallback_from_mode"] = "prepared"

        try:
            download_raw_dataset_repo(fallback_cfg)
        except Exception as raw_error:
            raise RuntimeError(
                "Prepared mode failed, and raw fallback also failed."
            ) from raw_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="dataset_config.yaml",
        help="Path to dataset YAML config.",
    )
    args = parser.parse_args()

    configs = iter_dataset_configs(load_config(args.config))
    archive_paths = []

    for cfg in configs:
        mode = cfg["mode"]
        output_dir = Path(cfg["output_dir"])
        archive_path = Path(cfg["archive_path"])

        print(f"\nProcessing dataset: {cfg['dataset_id']} (mode: {mode})")

        download_dataset(cfg)

        print(f"Creating archive: {archive_path}")
        make_tar_gz(output_dir, archive_path)
        archive_paths.append(archive_path)

    print("\nDone.")
    print("Transfer these files to the offline machine:")
    for archive_path in archive_paths:
        print(f"  {archive_path}")


if __name__ == "__main__":
    main()
