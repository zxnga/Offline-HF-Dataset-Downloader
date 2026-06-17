# Offline Hugging Face Dataset Downloader
Downloads one or more Hugging Face datasets on an internet-connected machine, saves them locally, and packages each dataset into a `.tar.gz` archive for transfer to an offline training machine.

This is useful when your training machine does not have internet access.

## Files

```text
.
|-- dataset_config.yaml
|-- download_hf_dataset.py
|-- src/
|   |-- archive.py
|   |-- cache.py
|   |-- cli.py
|   |-- config.py
|   |-- download.py
|   `-- manifest.py
`-- README.md
```

## Overview

The workflow is:

1. Edit `dataset_config.yaml` with one or more Hugging Face dataset IDs.
2. Run `download_hf_dataset.py` on a machine with internet access.
3. Transfer the generated `.tar.gz` archive files to the offline training machine.
4. Extract the archives.
5. Load each dataset locally with `load_from_disk()`.

## Install dependencies

On the internet-connected machine:

```bash
pip install -U datasets huggingface_hub pyyaml
```

## Configure datasets

Edit `dataset_config.yaml`:

```yaml
mode: "prepared"
fallback_to_raw: false
continue_on_error: false
keep_only_archive: false
archive_timing: "after_each"
keep_hf_cache: true
all_config_names: false
output_dir: "./offline_datasets"
archive_path: null
global_manifest_path: null
token_env: "HF_TOKEN"
trust_remote_code: false

datasets:
  - dataset_id: "HuggingFaceH4/ultrachat_200k"
    config_name: null
    split: null
    revision: "main"

  - dataset_id: "stanfordnlp/imdb"
    mode: "raw"
    fallback_to_raw: false
    config_name: null
    split: "train"
    revision: "main"
```

With this config, the script creates one folder per dataset under `output_dir`:

```text
offline_datasets/
|-- ultrachat_200k/
|   `-- ultrachat_200k.tar.gz
`-- imdb/
    `-- imdb.tar.gz
```

The dataset folder name is derived from the last part of `dataset_id`. For example, `HuggingFaceH4/ultrachat_200k` becomes `ultrachat_200k`.

### Important fields

| Field | Description |
|---|---|
| `datasets` | List of datasets to download in one run |
| `dataset_id` | Hugging Face dataset repo ID, for example `"HuggingFaceH4/ultrachat_200k"` |
| `config_name` | Optional dataset subset/config name |
| `config_names` | Optional list of subset/config names to download as separate prepared datasets |
| `all_config_names` | Set to `true` to discover and download every subset/config as separate prepared datasets |
| `split` | Optional split such as `"train"`, `"validation"`, or `"test"` |
| `revision` | Dataset branch, tag, or commit hash |
| `mode` | Use `"prepared"` for offline training or `"raw"` for original repo files |
| `fallback_to_raw` | Set to `true` to try raw mode if prepared mode fails |
| `continue_on_error` | Set to `true` to skip datasets that still fail and continue the run |
| `keep_only_archive` | Set to `true` to delete the downloaded dataset folder after the archive is created |
| `archive_timing` | Use `"after_each"` to archive after every download, or `"end"` to download everything first and archive afterward |
| `keep_hf_cache` | Set to `false` to use a temporary Hugging Face cache and delete it after the run |
| `output_dir` | Base folder where per-dataset folders are created |
| `archive_path` | Optional archive file path. Set to `null` for the default per-dataset archive |
| `global_manifest_path` | Optional run manifest path. Set to `null` to write `download_manifest.json` under `output_dir` |
| `token_env` | Environment variable containing your Hugging Face token |
| `trust_remote_code` | Set to `true` only if the dataset requires custom dataset code |

Top-level fields are defaults shared by every dataset. Dataset entries can override them when needed, including `mode`, `fallback_to_raw`, `continue_on_error`, `keep_only_archive`, `archive_timing`, and `all_config_names`. `keep_hf_cache` is a run-wide setting and should be set at the top level.

Use `config_name` for one subset/config, `config_names` for a specific list, or `all_config_names: true` to discover and download every subset/config. Do not set more than one of these on the same dataset entry.

If `all_config_names: true` is set at the top level, a dataset entry with `config_name` or `config_names` uses that explicit value instead. Raw-mode datasets do not use config expansion because raw mode downloads the full repository snapshot.

## Archive behavior

When `archive_path` is `null`, each archive is written inside its dataset folder using the same name as the folder:

```text
./offline_datasets/ultrachat_200k/ultrachat_200k.tar.gz
```

If `keep_only_archive` is `true`, the default archive path is moved outside the dataset folder so the folder can be removed safely:

```text
./offline_datasets/ultrachat_200k.tar.gz
```

You can still set `archive_path` for a single-dataset config or override it per dataset. Avoid using the same archive path for multiple datasets, because the script rejects duplicate archive outputs.

When `keep_only_archive` is `true`, any custom `archive_path` must be outside that dataset's `output_dir`.

`archive_timing` controls when archives are created:

```yaml
archive_timing: "after_each"
```

This is the default. It downloads one dataset, creates its archive, and then removes the dataset folder if `keep_only_archive: true`. This usually uses less peak disk space.

```yaml
archive_timing: "end"
```

This downloads every dataset first and creates archives only after the download phase is complete. Use this when the internet connection window matters more than disk usage. It can require much more temporary disk space, because downloaded folders remain on disk until the archive phase. When `keep_only_archive: true`, folders are deleted after their archive is created during that final archive phase.

## Global run manifest

Each run writes a global manifest in addition to the per-dataset `offline_manifest.json` files. By default it is:

```text
./offline_datasets/download_manifest.json
```

Set a custom path with:

```yaml
global_manifest_path: "./offline_datasets/my_run_manifest.json"
```

The global manifest records each dataset, the config names that were specified or discovered, every expanded download job, the final mode used (`prepared` or `raw`), archive paths, raw fallback metadata, skipped datasets, and failures.

## Hugging Face cache behavior

By default, Hugging Face may cache downloads under your user cache directory. To avoid leaving new persistent Hugging Face cache files from this script, set:

```yaml
keep_hf_cache: false
```

The script then creates a temporary cache for the run, points `HF_HOME`, `HF_HUB_CACHE`, `HF_DATASETS_CACHE`, `HF_MODULES_CACHE`, `HF_ASSETS_CACHE`, and `HF_XET_CACHE` at it, and deletes that temporary cache before exiting.

This does not delete cache files that already existed before the run.

If you use `keep_hf_cache: false` for private or gated datasets, set `HF_TOKEN` in your environment instead of relying on a token saved by `huggingface-cli login`.

## Single-dataset config

The script still accepts the older single-dataset shape:

```yaml
dataset_id: "HuggingFaceH4/ultrachat_200k"
config_name: null
split: null
revision: "main"
mode: "prepared"
keep_only_archive: false
archive_timing: "after_each"
keep_hf_cache: true
output_dir: "./offline_datasets"
archive_path: null
global_manifest_path: null
token_env: "HF_TOKEN"
trust_remote_code: false
```

This creates:

```text
offline_datasets/ultrachat_200k/
offline_datasets/ultrachat_200k/ultrachat_200k.tar.gz
```

## Recommended mode

For offline training, use:

```yaml
mode: "prepared"
```

This downloads the dataset using the `datasets` library and saves a local copy using `save_to_disk()`.

The offline machine can then load it with:

```python
from datasets import load_from_disk

ds = load_from_disk("./offline_datasets/ultrachat_200k")
```

## Download datasets

Run this on the internet-connected machine:

```bash
python download_hf_dataset.py --config dataset_config.yaml
```

Transfer the generated `.tar.gz` files to the offline training machine.

## Download a private or gated dataset

If the dataset requires authentication, create a Hugging Face access token and export it before running the script:

For Linux/macOS:
```bash
export HF_TOKEN="hf_your_token_here"
```

For Windows:
```bash
$env:HF_TOKEN="hf_your_token_here"
```

Then run:

```bash
python download_hf_dataset.py --config dataset_config.yaml
```

Do not put your token directly into the YAML file.

## Extract on the offline machine

On the offline training machine:

```bash
tar -xzf ultrachat_200k.tar.gz
```

This should create:

```text
ultrachat_200k/
```

## Load the dataset offline

In your training code:

```python
from datasets import load_from_disk

ds = load_from_disk("./ultrachat_200k")

print(ds)
```

If you downloaded all splits, `ds` will usually be a `DatasetDict`. If you downloaded a single split, `ds` may be a single `Dataset`.

## Force offline mode

On the offline machine, you can set these environment variables to make sure no internet access is attempted:

```bash
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

Then run your training job as normal.

## Example: download only the train split

```yaml
mode: "prepared"
output_dir: "./offline_datasets"
archive_path: null
token_env: "HF_TOKEN"
trust_remote_code: false

datasets:
  - dataset_id: "HuggingFaceH4/ultrachat_200k"
    config_name: null
    split: "train"
    revision: "main"
```

## Example: keep only archives

To keep only the `.tar.gz` files after each dataset is archived:

```yaml
mode: "prepared"
keep_only_archive: true
archive_timing: "after_each"
keep_hf_cache: false
output_dir: "./offline_datasets"
archive_path: null

datasets:
  - dataset_id: "HuggingFaceH4/ultrachat_200k"
    split: null
    revision: "main"
```

This leaves:

```text
offline_datasets/
`-- ultrachat_200k.tar.gz
```

## Example: download multiple subsets

Some Hugging Face datasets expose many subsets/configs. To download several of them in prepared mode, use `config_names`:

```yaml
mode: "prepared"
output_dir: "./offline_datasets"
archive_path: null

datasets:
  - dataset_id: "flax-sentence-embeddings/stackexchange_titlebody_best_voted_answer_jsonl"
    config_names:
      - "3dprinting"
      - "android"
      - "askubuntu"
    split: "train"
    revision: "main"
```

This creates one prepared dataset folder and archive per subset:

```text
offline_datasets/
|-- stackexchange_titlebody_best_voted_answer_jsonl__3dprinting/
|   `-- stackexchange_titlebody_best_voted_answer_jsonl__3dprinting.tar.gz
|-- stackexchange_titlebody_best_voted_answer_jsonl__android/
|   `-- stackexchange_titlebody_best_voted_answer_jsonl__android.tar.gz
`-- stackexchange_titlebody_best_voted_answer_jsonl__askubuntu/
    `-- stackexchange_titlebody_best_voted_answer_jsonl__askubuntu.tar.gz
```

To download every subset/config without listing them manually, use `all_config_names`:

```yaml
mode: "prepared"
keep_only_archive: true
keep_hf_cache: false
output_dir: "./offline_datasets"
archive_path: null

datasets:
  - dataset_id: "flax-sentence-embeddings/stackexchange_titlebody_best_voted_answer_jsonl"
    all_config_names: true
    split: "train"
    revision: "main"
```

If automatic config discovery fails and `fallback_to_raw: true` is enabled, the script downloads that dataset in raw mode instead of stopping the whole run.

If automatic config discovery fails and `fallback_to_raw: false`, setting `continue_on_error: true` skips that dataset and continues with the others.

## Example: pin a dataset version

For reproducibility, use a commit hash instead of `"main"`:

```yaml
revision: "abc123..."
```

This helps ensure that future downloads use the exact same dataset version.

## Raw mode

The script also supports:

```yaml
mode: "raw"
```

This downloads the raw files from the Hugging Face dataset repository.

Use raw mode only if you specifically need the original files, such as `.jsonl`, `.csv`, or `.parquet`.
`config_names` and `all_config_names` are only supported in prepared mode because raw mode downloads the full dataset repository.

For training, `prepared` mode is usually better because it can be loaded directly with:

```python
from datasets import load_from_disk

ds = load_from_disk("./ultrachat_200k")
```

## Fallback to raw mode

If you want prepared mode first, but still want an archive of the raw repo files when `datasets.load_dataset()` fails, enable:

```yaml
mode: "prepared"
fallback_to_raw: true
continue_on_error: true
```

You can also enable it for only one dataset:

```yaml
datasets:
  - dataset_id: "owner/dataset_name"
    mode: "prepared"
    fallback_to_raw: true
    continue_on_error: true
```

If prepared mode fails, the script tries raw mode first when `fallback_to_raw: true`. If raw mode also fails and `continue_on_error: true`, that dataset is skipped and the next one starts.

## Troubleshooting

### `DatasetNotFoundError`

Check that `dataset_id` is correct.

Example:

```yaml
dataset_id: "owner/dataset_name"
```

### Authentication error

For private or gated datasets, make sure you have access to the dataset and that your token is set:

```bash
export HF_TOKEN="hf_your_token_here"
```

### Dataset requires custom code

Some older or custom datasets may require:

```yaml
trust_remote_code: true
```

Only enable this for datasets you trust.

### Offline machine tries to access the internet

Set:

```bash
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

## Full offline training example

```python
from datasets import load_from_disk

dataset = load_from_disk("./offline_dataset")

if "train" in dataset:
    train_dataset = dataset["train"]
else:
    train_dataset = dataset

print(train_dataset[0])
print(len(train_dataset))
```
