from contextlib import contextmanager
import os
import tempfile
from pathlib import Path

from constants import DEFAULT_KEEP_HF_CACHE, HF_CACHE_ENV_VARS
from utils import get_bool_setting


def resolve_keep_hf_cache(config: dict) -> bool:
    return get_bool_setting(config, "keep_hf_cache", DEFAULT_KEEP_HF_CACHE, "config")


@contextmanager
def hf_cache_context(keep_hf_cache: bool):
    if keep_hf_cache:
        yield
        return

    original_env = {name: os.environ.get(name) for name in HF_CACHE_ENV_VARS}

    with tempfile.TemporaryDirectory(prefix="offline_hf_cache_") as temp_cache_dir:
        temp_cache_root = Path(temp_cache_dir)

        for env_name, folder_name in HF_CACHE_ENV_VARS.items():
            cache_path = temp_cache_root / folder_name
            cache_path.mkdir(parents=True, exist_ok=True)
            os.environ[env_name] = str(cache_path)

        print(f"Using temporary Hugging Face cache: {temp_cache_root}")

        try:
            yield
        finally:
            for env_name, original_value in original_env.items():
                if original_value is None:
                    os.environ.pop(env_name, None)
                else:
                    os.environ[env_name] = original_value

            print("Temporary Hugging Face cache removed.")
