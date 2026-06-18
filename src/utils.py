import re
from urllib.parse import quote


def load_config(path: str) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_error(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


def huggingface_dataset_url(dataset_id: str) -> str:
    encoded_dataset_id = quote(dataset_id.strip("/"), safe="/")
    return f"https://huggingface.co/datasets/{encoded_dataset_id}"


def dataset_folder_name(dataset_id: str) -> str:
    name = dataset_id.rstrip("/").split("/")[-1]
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")

    if not name:
        raise ValueError(f"Could not derive a folder name from dataset_id: {dataset_id!r}")

    return name


def config_folder_name(config_name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", config_name).strip("._")

    if not name:
        raise ValueError(f"Could not derive a folder name from config_name: {config_name!r}")

    return name


def get_bool_setting(cfg: dict, key: str, default: bool, label: str) -> bool:
    value = cfg.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{label} {key} must be true or false")

    return value


def copy_string_list(value):
    if isinstance(value, list):
        return list(value)

    return value
