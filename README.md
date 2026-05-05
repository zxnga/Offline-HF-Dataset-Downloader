# Offline Hugging Face Dataset Downloader

Downloads a Hugging Face dataset on an internet-connected machine, packages it into a `.tar.gz` archive suited for transfer to an offline machine for training.

This is useful when your training machine does not have internet access.

## Files

```text
.
├── dataset_config.yaml
├── download_hf_dataset.py
└── README.md
```

## Overview

The workflow is:

1. Edit `dataset_config.yaml` with the Hugging Face dataset ID.
2. Run `download_hf_dataset.py` on a machine with internet access.
3. Transfer the generated `.tar.gz` file to the offline training machine.
4. Extract the archive.
5. Load the dataset locally with `load_from_disk()`.

## Install dependencies

On the internet-connected machine:

```bash
pip install -U datasets huggingface_hub pyyaml
```

## Configure the dataset

Edit `dataset_config.yaml`:

```yaml
dataset_id: "HuggingFaceH4/ultrachat_200k"

config_name: null
split: null
revision: "main"

mode: "prepared"

output_dir: "./offline_dataset"
archive_path: "./offline_dataset.tar.gz"

token_env: "HF_TOKEN"
trust_remote_code: false
```

### Important fields

| Field | Description |
|---|---|
| `dataset_id` | Hugging Face dataset repo ID, for example `"HuggingFaceH4/ultrachat_200k"` |
| `config_name` | Optional dataset subset/config name |
| `split` | Optional split such as `"train"`, `"validation"`, or `"test"` |
| `revision` | Dataset branch, tag, or commit hash |
| `mode` | Use `"prepared"` for offline training |
| `output_dir` | Local folder where the dataset will be saved |
| `archive_path` | Final archive file to transfer |
| `token_env` | Environment variable containing your Hugging Face token |
| `trust_remote_code` | Set to `true` only if the dataset requires custom dataset code |

## Recommended mode

For offline training, use:

```yaml
mode: "prepared"
```

This downloads the dataset using the `datasets` library and saves a local copy using `save_to_disk()`.

The offline machine can then load it with:

```python
from datasets import load_from_disk

ds = load_from_disk("./offline_dataset")
```

## Download a public dataset

Run this on the internet-connected machine:

```bash
python download_hf_dataset.py --config dataset_config.yaml
```

This creates an archive such as:

```text
offline_dataset.tar.gz
```

Transfer this file to the offline training machine.

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
tar -xzf offline_dataset.tar.gz
```

This should create:

```text
offline_dataset/
```

## Load the dataset offline

In your training code:

```python
from datasets import load_from_disk

ds = load_from_disk("./offline_dataset")

print(ds)
```

If you downloaded all splits, `ds` will usually be a `DatasetDict`, for example:

```text
DatasetDict({
    train: Dataset(...)
    test: Dataset(...)
})
```

If you downloaded a single split, `ds` may be a single `Dataset`.

## Force offline mode

On the offline machine, you can set these environment variables to make sure no internet access is attempted:

```bash
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

Then run your training job as normal.

## Example: download only the train split

Update `dataset_config.yaml`:

```yaml
dataset_id: "HuggingFaceH4/ultrachat_200k"
config_name: null
split: "train"
revision: "main"
mode: "prepared"
output_dir: "./offline_dataset"
archive_path: "./offline_dataset.tar.gz"
token_env: "HF_TOKEN"
trust_remote_code: false
```

Then run:

```bash
python download_hf_dataset.py --config dataset_config.yaml
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

ds = load_from_disk("./offline_dataset")
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

Also make sure your training code uses:

```python
load_from_disk("./offline_dataset")
```

instead of:

```python
load_dataset("owner/dataset_name")
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
