import os
from pathlib import Path

from archive import path_is_relative_to
from constants import (
    DEFAULT_ALL_CONFIG_NAMES,
    DEFAULT_ARCHIVE_TIMING,
    DEFAULT_CONTINUE_ON_ERROR,
    DEFAULT_FALLBACK_TO_RAW,
    DEFAULT_KEEP_ONLY_ARCHIVE,
    DEFAULT_MODE,
    VALID_ARCHIVE_TIMINGS,
    VALID_MODES,
)
from utils import (
    config_folder_name,
    copy_string_list,
    dataset_folder_name,
    get_bool_setting,
)


def set_config_selection_metadata(
    cfg: dict,
    selection: str,
    source: str,
    config_names=None,
    note=None,
) -> None:
    cfg["_manifest_config_selection"] = selection
    cfg["_manifest_config_names_source"] = source
    cfg["_manifest_config_names"] = copy_string_list(config_names)

    if note is not None:
        cfg["_manifest_config_selection_note"] = note


def resolve_dataset_mode(cfg: dict, label: str) -> None:
    mode = cfg.get("mode", DEFAULT_MODE)
    if mode not in VALID_MODES:
        valid_modes = ", ".join(f"{mode!r}" for mode in sorted(VALID_MODES))
        raise ValueError(f"{label} mode must be one of {valid_modes}; got {mode!r}")

    cfg["mode"] = mode


def resolve_fallback_to_raw(cfg: dict, label: str) -> None:
    cfg["fallback_to_raw"] = get_bool_setting(
        cfg, "fallback_to_raw", DEFAULT_FALLBACK_TO_RAW, label
    )


def resolve_keep_only_archive(cfg: dict, label: str) -> None:
    cfg["keep_only_archive"] = get_bool_setting(
        cfg, "keep_only_archive", DEFAULT_KEEP_ONLY_ARCHIVE, label
    )


def resolve_continue_on_error(cfg: dict, label: str) -> None:
    cfg["continue_on_error"] = get_bool_setting(
        cfg, "continue_on_error", DEFAULT_CONTINUE_ON_ERROR, label
    )


def resolve_archive_timing(cfg: dict, label: str) -> None:
    archive_timing = cfg.get("archive_timing", DEFAULT_ARCHIVE_TIMING)
    if archive_timing not in VALID_ARCHIVE_TIMINGS:
        valid_timings = ", ".join(
            f"{archive_timing!r}" for archive_timing in sorted(VALID_ARCHIVE_TIMINGS)
        )
        raise ValueError(
            f"{label} archive_timing must be one of {valid_timings}; "
            f"got {archive_timing!r}"
        )

    cfg["archive_timing"] = archive_timing


def discover_config_names(cfg: dict, label: str) -> list[str]:
    from datasets import get_dataset_config_names

    dataset_id = cfg["dataset_id"]
    revision = cfg.get("revision")
    token = os.environ.get(cfg.get("token_env", "HF_TOKEN"))

    kwargs = {
        "path": dataset_id,
        "revision": revision,
        "token": token,
    }
    kwargs = {key: value for key, value in kwargs.items() if value is not None}

    print(f"Discovering dataset subsets/configs: {dataset_id}")
    try:
        config_names = get_dataset_config_names(**kwargs)
    except Exception as error:
        raise RuntimeError(
            f"{label} could not discover config names for {dataset_id!r}. "
            "If this dataset uses an older dataset script, set all_config_names: false "
            "and either set config_name/config_names manually or use raw mode. "
            f"Original error: {type(error).__name__}: {error}"
        ) from None

    if not config_names:
        raise ValueError(f"{label} no config names were found for {dataset_id!r}")

    preview = ", ".join(config_names[:10])
    if len(config_names) > 10:
        preview = f"{preview}, ..."

    print(f"Found {len(config_names)} config(s): {preview}")
    return list(config_names)


def expand_config_names(cfg: dict, label: str) -> list[dict]:
    cfg = cfg.copy()
    all_config_names = get_bool_setting(
        cfg, "all_config_names", DEFAULT_ALL_CONFIG_NAMES, label
    )
    fallback_to_raw = get_bool_setting(
        cfg, "fallback_to_raw", DEFAULT_FALLBACK_TO_RAW, label
    )
    continue_on_error = get_bool_setting(
        cfg, "continue_on_error", DEFAULT_CONTINUE_ON_ERROR, label
    )

    cfg["all_config_names"] = all_config_names
    cfg["fallback_to_raw"] = fallback_to_raw
    cfg["continue_on_error"] = continue_on_error
    mode = cfg.get("mode", DEFAULT_MODE)

    has_config_name = cfg.get("config_name") is not None
    has_config_names = "config_names" in cfg and cfg["config_names"] is not None
    requested_config_names = copy_string_list(cfg.get("config_names"))

    cfg["_manifest_requested_mode"] = mode
    cfg["_manifest_requested_config_name"] = cfg.get("config_name")
    cfg["_manifest_requested_config_names"] = requested_config_names
    cfg["_manifest_requested_all_config_names"] = all_config_names

    if mode == "raw":
        if has_config_name or has_config_names or all_config_names:
            raise ValueError(
                f"{label} cannot set config_name, config_names, or "
                "all_config_names with raw mode"
            )

        set_config_selection_metadata(
            cfg,
            "raw_repo",
            "raw_repo",
            None,
            "Raw mode downloads the full dataset repository; config selectors are not expanded.",
        )
        cfg["all_config_names"] = False
        cfg.pop("config_names", None)
        return [cfg]

    selected_options = sum([has_config_name, has_config_names, all_config_names])
    if selected_options > 1:
        raise ValueError(
            f"{label} can only set one of config_name, config_names, or all_config_names"
        )

    if has_config_name:
        set_config_selection_metadata(cfg, "config_name", "specified", [cfg["config_name"]])
        return [cfg]

    if not has_config_names and not all_config_names:
        set_config_selection_metadata(cfg, "default", "not_specified", None)
        return [cfg]

    if all_config_names:
        try:
            config_names = discover_config_names(cfg, label)
        except Exception as error:
            if fallback_to_raw:
                print(
                    f"{label} could not discover configs; fallback_to_raw is enabled, "
                    "so downloading the raw dataset repo instead."
                )
                print(f"Discovery error: {type(error).__name__}: {error}")

                fallback_cfg = cfg.copy()
                fallback_cfg["mode"] = "raw"
                fallback_cfg["all_config_names"] = False
                fallback_cfg["fallback_from_mode"] = "prepared"
                fallback_cfg["fallback_reason"] = "config_discovery_failed"
                fallback_cfg["config_discovery_error"] = f"{type(error).__name__}: {error}"
                set_config_selection_metadata(
                    fallback_cfg,
                    "all_config_names",
                    "discovery_failed",
                    None,
                )
                fallback_cfg.pop("config_name", None)
                fallback_cfg.pop("config_names", None)
                return [fallback_cfg]

            if continue_on_error:
                print(
                    f"{label} could not discover configs; continue_on_error is enabled, "
                    "so skipping this dataset."
                )
                print(f"Discovery error: {type(error).__name__}: {error}")
                skipped_cfg = cfg.copy()
                skipped_cfg["all_config_names"] = False
                skipped_cfg["_skip_before_download"] = True
                skipped_cfg["_skip_stage"] = "config_discovery"
                skipped_cfg["_skip_error"] = f"{type(error).__name__}: {error}"
                set_config_selection_metadata(
                    skipped_cfg,
                    "all_config_names",
                    "discovery_failed",
                    None,
                )
                skipped_cfg.pop("config_name", None)
                skipped_cfg.pop("config_names", None)
                return [skipped_cfg]

            raise
    else:
        config_names = cfg["config_names"]

    if not isinstance(config_names, list) or not config_names:
        raise ValueError(f"{label} config_names must be a non-empty list")

    expanded_configs = []
    seen_config_names = set()

    for index, config_name in enumerate(config_names, start=1):
        if not isinstance(config_name, str) or not config_name:
            raise ValueError(f"{label} config_names[{index}] must be a non-empty string")

        if config_name in seen_config_names:
            raise ValueError(f"{label} config_names contains duplicate value: {config_name!r}")

        seen_config_names.add(config_name)

        expanded_cfg = cfg.copy()
        expanded_cfg.pop("config_names", None)
        expanded_cfg["config_name"] = config_name
        expanded_cfg["_folder_suffix"] = config_folder_name(config_name)
        set_config_selection_metadata(
            expanded_cfg,
            "all_config_names" if all_config_names else "config_names",
            "discovered" if all_config_names else "specified",
            config_names,
        )
        expanded_configs.append(expanded_cfg)

    return expanded_configs


def resolve_dataset_paths(cfg: dict, label: str) -> dict:
    cfg = cfg.copy()
    resolve_dataset_mode(cfg, label)
    resolve_fallback_to_raw(cfg, label)
    resolve_keep_only_archive(cfg, label)
    resolve_continue_on_error(cfg, label)
    resolve_archive_timing(cfg, label)

    dataset_id = cfg["dataset_id"]
    folder_name = dataset_folder_name(dataset_id)
    folder_suffix = cfg.pop("_folder_suffix", None)

    if folder_suffix is not None:
        folder_name = f"{folder_name}__{folder_suffix}"

    base_output_dir = Path(cfg.get("output_dir", "./offline_datasets"))

    # Preserve existing configs that already point directly at the dataset folder.
    output_dir_is_dataset_folder = base_output_dir.name == folder_name
    if output_dir_is_dataset_folder:
        output_dir = base_output_dir
    else:
        output_dir = base_output_dir / folder_name

    archive_path = cfg.get("archive_path")
    if archive_path is None:
        if cfg["keep_only_archive"]:
            archive_dir = output_dir.parent if output_dir_is_dataset_folder else base_output_dir
            archive_path = archive_dir / f"{folder_name}.tar.gz"
        else:
            archive_path = output_dir / f"{folder_name}.tar.gz"
    else:
        archive_path = Path(archive_path)

    if cfg["keep_only_archive"] and path_is_relative_to(archive_path, output_dir):
        raise ValueError(
            f"{label} archive_path must be outside output_dir when keep_only_archive is true"
        )

    cfg["output_dir"] = str(output_dir)
    cfg["archive_path"] = str(archive_path)

    return cfg


def iter_dataset_configs(config: dict) -> list[dict]:
    if "datasets" not in config:
        if "dataset_id" not in config:
            raise ValueError("config is missing required field: dataset_id")

        config = config.copy()
        config["_manifest_group_label"] = "config"
        config["_manifest_group_index"] = 1
        expanded_configs = expand_config_names(config, "config")
        has_expanded_configs = (
            ("config_names" in config and config["config_names"] is not None)
            or bool(config.get("all_config_names", DEFAULT_ALL_CONFIG_NAMES))
        )

        resolved_configs = []
        for index, dataset_cfg in enumerate(expanded_configs, start=1):
            label = f"config config_names[{index}]" if has_expanded_configs else "config"
            dataset_cfg["_manifest_job_label"] = label
            dataset_cfg["_manifest_job_index"] = index
            resolved_configs.append(resolve_dataset_paths(dataset_cfg, label))

        return resolved_configs

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
        merged_cfg["_manifest_group_label"] = f"datasets[{index}]"
        merged_cfg["_manifest_group_index"] = index

        if "dataset_id" not in merged_cfg:
            raise ValueError(f"datasets[{index}] is missing required field: dataset_id")

        has_dataset_config_selector = (
            ("config_name" in dataset_cfg and dataset_cfg["config_name"] is not None)
            or ("config_names" in dataset_cfg and dataset_cfg["config_names"] is not None)
        )
        if has_dataset_config_selector and "all_config_names" not in dataset_cfg:
            merged_cfg["all_config_names"] = False

        if dataset_cfg.get("mode") == "raw":
            if "config_name" not in dataset_cfg:
                merged_cfg.pop("config_name", None)
            if "config_names" not in dataset_cfg:
                merged_cfg.pop("config_names", None)
            if "all_config_names" not in dataset_cfg:
                merged_cfg["all_config_names"] = False

        expanded_configs = expand_config_names(merged_cfg, f"datasets[{index}]")
        has_expanded_configs = (
            ("config_names" in merged_cfg and merged_cfg["config_names"] is not None)
            or bool(merged_cfg.get("all_config_names", DEFAULT_ALL_CONFIG_NAMES))
        )

        for config_index, expanded_cfg in enumerate(expanded_configs, start=1):
            label = (
                f"datasets[{index}] config_names[{config_index}]"
                if has_expanded_configs
                else f"datasets[{index}]"
            )
            expanded_cfg["_manifest_job_label"] = label
            expanded_cfg["_manifest_job_index"] = config_index
            resolved_configs.append(resolve_dataset_paths(expanded_cfg, label))

    output_dirs = [cfg["output_dir"] for cfg in resolved_configs]
    if len(output_dirs) != len(set(output_dirs)):
        raise ValueError("Multiple datasets resolve to the same output_dir")

    archive_paths = [cfg["archive_path"] for cfg in resolved_configs]
    if len(archive_paths) != len(set(archive_paths)):
        raise ValueError("Multiple datasets resolve to the same archive_path")

    return resolved_configs
