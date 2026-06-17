import json
import os
from pathlib import Path

from archive import make_tar_gz, remove_output_dir
from utils import format_error


def download_prepared_dataset(cfg: dict) -> dict:
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

    dataset_name = dataset_display_name(cfg)

    print(f"Downloading dataset with datasets.load_dataset(): {dataset_name}")
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
    return manifest


def download_raw_dataset_repo(cfg: dict) -> dict:
    from huggingface_hub import snapshot_download

    dataset_id = cfg["dataset_id"]
    config_name = cfg.get("config_name")
    revision = cfg.get("revision")
    output_dir = Path(cfg["output_dir"])
    token = os.environ.get(cfg.get("token_env", "HF_TOKEN"))
    fallback_from_mode = cfg.get("fallback_from_mode")
    fallback_reason = cfg.get("fallback_reason")
    config_discovery_error = cfg.get("config_discovery_error")
    prepared_error = cfg.get("prepared_error")

    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = dataset_display_name(cfg)

    print(f"Downloading raw dataset repo files: {dataset_name}")
    snapshot_download(
        repo_id=dataset_id,
        repo_type="dataset",
        revision=revision,
        local_dir=str(output_dir),
        token=token,
    )

    manifest = {
        "dataset_id": dataset_id,
        "config_name": config_name,
        "revision": revision,
        "mode": "raw",
        "load_offline_with": "datasets.load_dataset using local files",
    }

    if fallback_from_mode is not None:
        manifest["fallback_from_mode"] = fallback_from_mode

    if fallback_reason is not None:
        manifest["fallback_reason"] = fallback_reason

    if config_discovery_error is not None:
        manifest["config_discovery_error"] = config_discovery_error

    if prepared_error is not None:
        manifest["prepared_error"] = prepared_error

    with open(output_dir / "offline_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Raw dataset files saved successfully.")
    return manifest


def dataset_display_name(cfg: dict) -> str:
    dataset_name = cfg["dataset_id"]

    if cfg.get("config_name") is not None:
        dataset_name = f"{dataset_name} ({cfg['config_name']})"

    return dataset_name


def download_dataset(cfg: dict) -> dict:
    mode = cfg["mode"]

    if mode == "raw":
        return download_raw_dataset_repo(cfg)

    try:
        return download_prepared_dataset(cfg)
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
        fallback_cfg["fallback_reason"] = "prepared_download_failed"
        fallback_cfg["prepared_error"] = format_error(prepared_error)

        try:
            return download_raw_dataset_repo(fallback_cfg)
        except Exception as raw_error:
            raise RuntimeError(
                "Prepared mode failed "
                f"({type(prepared_error).__name__}: {prepared_error}); "
                "raw fallback also failed "
                f"({type(raw_error).__name__}: {raw_error})"
            ) from raw_error


def download_dataset_job(cfg: dict) -> dict:
    mode = cfg["mode"]
    output_dir = Path(cfg["output_dir"])
    dataset_name = dataset_display_name(cfg)

    print(f"\nProcessing dataset: {dataset_name} (mode: {mode})")

    download_manifest = download_dataset(cfg)

    return {
        "status": "downloaded",
        "final_mode": download_manifest["mode"],
        "output_dir": str(output_dir),
        "download_manifest": download_manifest,
    }


def archive_dataset(cfg: dict) -> dict:
    output_dir = Path(cfg["output_dir"])
    archive_path = Path(cfg["archive_path"])
    cleanup_error_message = None

    print(f"Creating archive: {archive_path}")
    make_tar_gz(output_dir, archive_path)

    if cfg["keep_only_archive"]:
        try:
            remove_output_dir(output_dir)
        except Exception as cleanup_error:
            cleanup_error_message = format_error(cleanup_error)
            print(
                "Warning: archive was created, but removing the dataset folder failed: "
                f"{cleanup_error_message}"
            )

    return {
        "archive_path": str(archive_path),
        "output_dir": str(output_dir),
        "cleanup_error": cleanup_error_message,
    }


def process_dataset(cfg: dict) -> dict:
    result = download_dataset_job(cfg)
    result.update(archive_dataset(cfg))
    result["status"] = "success"
    return result
