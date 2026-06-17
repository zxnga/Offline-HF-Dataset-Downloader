DEFAULT_MODE = "prepared"
VALID_MODES = {"prepared", "raw"}
DEFAULT_FALLBACK_TO_RAW = False
DEFAULT_KEEP_ONLY_ARCHIVE = False
DEFAULT_ALL_CONFIG_NAMES = False
DEFAULT_KEEP_HF_CACHE = True
DEFAULT_CONTINUE_ON_ERROR = False
DEFAULT_GLOBAL_MANIFEST_FILENAME = "download_manifest.json"

HF_CACHE_ENV_VARS = {
    "HF_HOME": "home",
    "HF_HUB_CACHE": "hub",
    "HF_DATASETS_CACHE": "datasets",
    "HF_MODULES_CACHE": "modules",
    "HF_ASSETS_CACHE": "assets",
    "HF_XET_CACHE": "xet",
}
