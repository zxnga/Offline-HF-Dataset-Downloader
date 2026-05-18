# Offline Hugging Face Dataset Downloader
Downloads one or more Hugging Face datasets on an internet-connected machine, saves them locally, and packages each dataset into a `.tar.gz` archive for transfer to an offline training machine.

This is useful when your training machine does not have internet access.

## Files

```text
.
|-- dataset_config.yaml
|-- download_hf_dataset.py
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
output_dir: "./offline_datasets"
archive_path: null
token_env: "HF_TOKEN"
trust_remote_code: false

datasets:
  - dataset_id: "HuggingFaceH4/ultrachat_200k"
    config_name: null
    split: null
    revision: "main"

  - dataset_id: "stanfordnlp/imdb"
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
| `split` | Optional split such as `"train"`, `"validation"`, or `"test"` |
| `revision` | Dataset branch, tag, or commit hash |
| `mode` | Use `"prepared"` for offline training |
| `output_dir` | Base folder where per-dataset folders are created |
| `archive_path` | Optional archive file path. Set to `null` for the default per-dataset archive |
| `token_env` | Environment variable containing your Hugging Face token |
| `trust_remote_code` | Set to `true` only if the dataset requires custom dataset code |

Top-level fields are defaults shared by every dataset. Dataset entries can override them when needed.

## Archive behavior

When `archive_path` is `null`, each archive is written inside its dataset folder using the same name as the folder:

```text
./offline_datasets/ultrachat_200k/ultrachat_200k.tar.gz
```

You can still set `archive_path` for a single-dataset config or override it per dataset. Avoid using the same archive path for multiple datasets, because the script rejects duplicate archive outputs.

## Single-dataset config

The script still accepts the older single-dataset shape:

```yaml
dataset_id: "HuggingFaceH4/ultrachat_200k"
config_name: null
split: null
revision: "main"
mode: "prepared"
output_dir: "./offline_datasets"
archive_path: null
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

```bash
export HF_TOKEN="hf_your_token_here"
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

For training, `prepared` mode is usually better because it can be loaded directly with:

```python
from datasets import load_from_disk

ds = load_from_disk("./ultrachat_200k")
```

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
