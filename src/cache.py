from contextlib import contextmanager
import os
import shutil
import tempfile
from pathlib import Path

from constants import (
    DEFAULT_CLEANUP_HF_CACHE_BETWEEN_DATASETS,
    DEFAULT_KEEP_HF_CACHE,
    HF_CACHE_ENV_VARS,
)
from utils import get_bool_setting


def resolve_keep_hf_cache(config: dict) -> bool:
    return get_bool_setting(config, "keep_hf_cache", DEFAULT_KEEP_HF_CACHE, "config")


def resolve_cleanup_hf_cache_between_datasets(config: dict, keep_hf_cache: bool) -> bool:
    cleanup = get_bool_setting(
        config,
        "cleanup_hf_cache_between_datasets",
        DEFAULT_CLEANUP_HF_CACHE_BETWEEN_DATASETS,
        "config",
    )

    if cleanup and keep_hf_cache:
        raise ValueError(
            "config cleanup_hf_cache_between_datasets requires keep_hf_cache: false"
        )

    return cleanup


def resolve_hf_cache_dir(config: dict):
    hf_cache_dir = config.get("hf_cache_dir")

    if hf_cache_dir is None:
        return None

    if not isinstance(hf_cache_dir, str) or not hf_cache_dir.strip():
        raise ValueError("config hf_cache_dir must be null or a non-empty path string")

    return Path(hf_cache_dir)


def set_hf_cache_env(temp_cache_root: Path) -> dict:
    original_env = {name: os.environ.get(name) for name in HF_CACHE_ENV_VARS}

    for env_name, folder_name in HF_CACHE_ENV_VARS.items():
        cache_path = temp_cache_root / folder_name
        cache_path.mkdir(parents=True, exist_ok=True)
        os.environ[env_name] = str(cache_path)

    return original_env


def restore_hf_cache_env(original_env: dict) -> None:
    for env_name, original_value in original_env.items():
        if original_value is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = original_value


class HfCacheManager:
    def __init__(self, cache_root=None, cleanup_between_datasets: bool = False):
        self.cache_root = Path(cache_root) if cache_root is not None else None
        self.cleanup_between_datasets = cleanup_between_datasets

    def cleanup_after_dataset(self) -> None:
        if not self.cleanup_between_datasets or self.cache_root is None:
            return

        try:
            cleanup_hf_cache_root(self.cache_root)
        except Exception as error:
            print(
                "Warning: failed to clean temporary Hugging Face cache between datasets: "
                f"{type(error).__name__}: {error}"
            )


def cleanup_hf_cache_root(cache_root: Path) -> None:
    resolved_cache_root = cache_root.resolve()

    if not cache_root.exists():
        return

    if resolved_cache_root.parent == resolved_cache_root:
        raise ValueError(f"Refusing to clean filesystem root: {cache_root}")

    if not resolved_cache_root.name.startswith("offline_hf_cache_"):
        raise ValueError(f"Refusing to clean unexpected cache root: {cache_root}")

    print(f"Cleaning temporary Hugging Face cache: {cache_root}")

    for folder_name in HF_CACHE_ENV_VARS.values():
        cache_path = cache_root / folder_name
        cache_path.mkdir(parents=True, exist_ok=True)

        for child in cache_path.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()


@contextmanager
def hf_cache_context(
    keep_hf_cache: bool,
    hf_cache_dir=None,
    cleanup_between_datasets: bool = False,
):
    if keep_hf_cache and hf_cache_dir is None:
        yield HfCacheManager()
        return

    if keep_hf_cache:
        cache_root = Path(hf_cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        original_env = set_hf_cache_env(cache_root)

        print(f"Using persistent Hugging Face cache: {cache_root}")

        try:
            yield HfCacheManager()
        finally:
            restore_hf_cache_env(original_env)

        return

    temporary_cache_parent = None
    if hf_cache_dir is not None:
        temporary_cache_parent = Path(hf_cache_dir)
        temporary_cache_parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="offline_hf_cache_",
        dir=temporary_cache_parent,
    ) as temp_cache_dir:
        temp_cache_root = Path(temp_cache_dir)
        original_env = set_hf_cache_env(temp_cache_root)

        print(f"Using temporary Hugging Face cache: {temp_cache_root}")
        cache_manager = HfCacheManager(
            temp_cache_root,
            cleanup_between_datasets=cleanup_between_datasets,
        )

        try:
            yield cache_manager
        finally:
            restore_hf_cache_env(original_env)

            print("Temporary Hugging Face cache removed.")
