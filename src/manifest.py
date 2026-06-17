from datetime import datetime, timezone
import json
from pathlib import Path

from constants import (
    DEFAULT_GLOBAL_MANIFEST_FILENAME,
    DEFAULT_KEEP_ONLY_ARCHIVE,
)
from utils import dataset_folder_name, get_bool_setting


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_global_manifest_path(config: dict) -> Path:
    manifest_path = config.get("global_manifest_path")

    if manifest_path is None:
        output_dir = Path(config.get("output_dir", "./offline_datasets"))
        keep_only_archive = get_bool_setting(
            config, "keep_only_archive", DEFAULT_KEEP_ONLY_ARCHIVE, "config"
        )

        if keep_only_archive and "datasets" not in config and "dataset_id" in config:
            folder_name = dataset_folder_name(config["dataset_id"])
            if output_dir.name == folder_name:
                return output_dir.parent / DEFAULT_GLOBAL_MANIFEST_FILENAME

        return output_dir / DEFAULT_GLOBAL_MANIFEST_FILENAME

    return Path(manifest_path)


def make_dataset_manifest_record(cfg: dict) -> dict:
    record = {
        "label": cfg.get("_manifest_group_label"),
        "dataset_index": cfg.get("_manifest_group_index"),
        "dataset_id": cfg["dataset_id"],
        "requested_mode": cfg.get("_manifest_requested_mode", cfg.get("mode")),
        "config_selection": cfg.get("_manifest_config_selection"),
        "config_names_source": cfg.get("_manifest_config_names_source"),
        "config_names": cfg.get("_manifest_config_names"),
        "requested_config_name": cfg.get("_manifest_requested_config_name"),
        "requested_config_names": cfg.get("_manifest_requested_config_names"),
        "requested_all_config_names": cfg.get("_manifest_requested_all_config_names"),
        "split": cfg.get("split"),
        "revision": cfg.get("revision"),
        "fallback_to_raw": cfg.get("fallback_to_raw"),
        "continue_on_error": cfg.get("continue_on_error"),
        "keep_only_archive": cfg.get("keep_only_archive"),
        "archive_timing": cfg.get("archive_timing"),
        "status": "pending",
        "jobs": [],
    }

    note = cfg.get("_manifest_config_selection_note")
    if note is not None:
        record["config_selection_note"] = note

    return record


def make_job_manifest_record(cfg: dict, job_id: str) -> dict:
    job = {
        "job_id": job_id,
        "label": cfg.get("_manifest_job_label"),
        "dataset_id": cfg["dataset_id"],
        "config_name": cfg.get("config_name"),
        "requested_mode": cfg.get("_manifest_requested_mode", cfg.get("mode")),
        "planned_mode": cfg.get("mode"),
        "final_mode": None,
        "status": "pending",
        "stage": None,
        "output_dir": cfg.get("output_dir"),
        "archive_path": cfg.get("archive_path"),
        "archive_timing": cfg.get("archive_timing"),
        "fallback_to_raw": cfg.get("fallback_to_raw"),
        "fallback_from_mode": cfg.get("fallback_from_mode"),
        "fallback_reason": cfg.get("fallback_reason"),
        "config_discovery_error": cfg.get("config_discovery_error"),
        "continued_after_error": False,
        "error": None,
    }

    if cfg.get("_skip_before_download"):
        job["status"] = "skipped"
        job["stage"] = cfg.get("_skip_stage")
        job["error"] = cfg.get("_skip_error")

    return job


def build_global_manifest(
    config_path: str,
    config: dict,
    keep_hf_cache: bool,
    manifest_path: Path,
    configs: list[dict],
) -> tuple[dict, dict]:
    manifest = {
        "manifest_version": 1,
        "created_at_utc": utc_now_iso(),
        "updated_at_utc": None,
        "config_path": str(Path(config_path).resolve()),
        "global_manifest_path": str(manifest_path),
        "keep_hf_cache": keep_hf_cache,
        "status": "pending",
        "summary": {},
        "datasets": [],
    }
    dataset_records = {}
    job_records = {}

    for cfg in configs:
        group_label = cfg.get("_manifest_group_label", cfg["dataset_id"])

        if group_label not in dataset_records:
            dataset_record = make_dataset_manifest_record(cfg)
            dataset_records[group_label] = dataset_record
            manifest["datasets"].append(dataset_record)

        dataset_record = dataset_records[group_label]
        job_id = f"{group_label}#{cfg.get('_manifest_job_index', len(dataset_record['jobs']) + 1)}"
        cfg["_manifest_job_id"] = job_id
        job_record = make_job_manifest_record(cfg, job_id)
        dataset_record["jobs"].append(job_record)
        job_records[job_id] = job_record

    update_global_manifest_status(manifest)
    return manifest, job_records


def update_global_manifest_status(manifest: dict) -> None:
    all_jobs = []

    for dataset in manifest["datasets"]:
        jobs = dataset["jobs"]
        all_jobs.extend(jobs)
        statuses = [job["status"] for job in jobs]

        if not statuses:
            dataset["status"] = "pending"
        elif any(status in ("pending", "downloaded") for status in statuses):
            dataset["status"] = "pending"
        elif all(status == "success" for status in statuses):
            dataset["status"] = "success"
        elif all(status == "skipped" for status in statuses):
            dataset["status"] = "skipped"
        elif any(status == "success" for status in statuses):
            dataset["status"] = "partial"
        else:
            dataset["status"] = "failed"

    summary = {
        "datasets_total": len(manifest["datasets"]),
        "datasets_success": sum(
            dataset["status"] == "success" for dataset in manifest["datasets"]
        ),
        "datasets_partial": sum(
            dataset["status"] == "partial" for dataset in manifest["datasets"]
        ),
        "datasets_failed": sum(
            dataset["status"] == "failed" for dataset in manifest["datasets"]
        ),
        "datasets_skipped": sum(
            dataset["status"] == "skipped" for dataset in manifest["datasets"]
        ),
        "jobs_total": len(all_jobs),
        "jobs_success": sum(job["status"] == "success" for job in all_jobs),
        "jobs_downloaded": sum(job["status"] == "downloaded" for job in all_jobs),
        "jobs_failed": sum(job["status"] == "failed" for job in all_jobs),
        "jobs_skipped": sum(job["status"] == "skipped" for job in all_jobs),
        "jobs_pending": sum(job["status"] == "pending" for job in all_jobs),
    }
    manifest["summary"] = summary

    if summary["jobs_pending"] or summary["jobs_downloaded"]:
        manifest["status"] = "pending"
    elif summary["jobs_failed"] or summary["jobs_skipped"] or summary["datasets_partial"]:
        manifest["status"] = "completed_with_errors"
    else:
        manifest["status"] = "success"

    manifest["updated_at_utc"] = utc_now_iso()


def write_global_manifest(manifest: dict, manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
