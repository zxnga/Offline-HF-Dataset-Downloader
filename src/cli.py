import argparse
from pathlib import Path

from cache import hf_cache_context, resolve_keep_hf_cache
from config import iter_dataset_configs
from download import dataset_display_name, process_dataset
from manifest import (
    build_global_manifest,
    resolve_global_manifest_path,
    update_global_manifest_status,
    write_global_manifest,
)
from utils import format_error, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="dataset_config.yaml",
        help="Path to dataset YAML config.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    keep_hf_cache = resolve_keep_hf_cache(config)
    global_manifest_path = resolve_global_manifest_path(config)

    with hf_cache_context(keep_hf_cache):
        configs = iter_dataset_configs(config)
        global_manifest, job_records = build_global_manifest(
            args.config,
            config,
            keep_hf_cache,
            global_manifest_path,
            configs,
        )
        write_global_manifest(global_manifest, global_manifest_path)
        print(f"Writing global manifest to: {global_manifest_path}")

        archive_paths = []
        skipped_datasets = []

        for cfg in configs:
            job_record = job_records[cfg["_manifest_job_id"]]
            if cfg.get("_skip_before_download"):
                dataset_name = dataset_display_name(cfg)
                error_message = cfg.get("_skip_error")
                print(
                    f"\nSkipping dataset before download: {dataset_name} "
                    f"({cfg.get('_skip_stage')})"
                )
                print(f"Error: {error_message}")
                skipped_datasets.append((dataset_name, error_message))
                update_global_manifest_status(global_manifest)
                write_global_manifest(global_manifest, global_manifest_path)
                continue

            try:
                result = process_dataset(cfg)
                archive_paths.append(Path(result["archive_path"]))

                job_record["status"] = "success"
                job_record["stage"] = "completed"
                job_record["final_mode"] = result["final_mode"]
                job_record["archive_path"] = result["archive_path"]
                job_record["output_dir"] = result["output_dir"]
                job_record["download_manifest"] = result["download_manifest"]
                job_record["cleanup_error"] = result["cleanup_error"]

                for key in (
                    "fallback_from_mode",
                    "fallback_reason",
                    "prepared_error",
                    "config_discovery_error",
                    "load_offline_with",
                ):
                    if key in result["download_manifest"]:
                        job_record[key] = result["download_manifest"][key]
            except Exception as error:
                error_message = format_error(error)
                job_record["status"] = "failed"
                job_record["stage"] = "download_or_archive"
                job_record["error"] = error_message
                job_record["continued_after_error"] = cfg["continue_on_error"]
                update_global_manifest_status(global_manifest)
                write_global_manifest(global_manifest, global_manifest_path)

                if not cfg["continue_on_error"]:
                    raise

                dataset_name = dataset_display_name(cfg)
                print(
                    f"\nSkipping dataset after error: {dataset_name} "
                    "(continue_on_error is enabled)"
                )
                print(f"Error: {error_message}")
                skipped_datasets.append((dataset_name, error_message))

            update_global_manifest_status(global_manifest)
            write_global_manifest(global_manifest, global_manifest_path)

    print("\nDone.")
    if archive_paths:
        print("Transfer these files to the offline machine:")
        for archive_path in archive_paths:
            print(f"  {archive_path}")
    else:
        print("No archives were created.")

    if skipped_datasets:
        print("\nSkipped these datasets:")
        for dataset_name, error_message in skipped_datasets:
            print(f"  {dataset_name}: {error_message}")

    print(f"\nGlobal manifest: {global_manifest_path}")
