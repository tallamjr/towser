# 🐈 `towser`

## CVPR25: Space-Time Instance Segmentation Challenge

> **Task Overview**
> You are tasked with developing a method for spatio-temporal instance
> segmentation that produces mask-accurate predictions of mice with consistent
> temporal IDs. We offer two tracks:
>
> 1. Event-Only Track: Methods must use only event data as input
> 2. Events-and-Frames Track: Methods can use both event and frame data

<img src="https://tub-rip.github.io/eventvision2025/images/sis_challenge_visu.png" alt="" width="500">

- https://github.com/tub-rip/MouseSIS
- https://arxiv.org/abs/2409.03358
- https://tub-rip.github.io/eventvision2025/
- https://www.codabench.org/competitions/5600/

**Why "Towser"?**

> A tortoiseshell cat named Towser holds the Guinness World Record for most mice
> caught at 28,899. She was so famous that a statue was erected of her in the
> city where she lived, and her paw prints are marked on every bottle of
> Fairlie’s Light Highland Liqueur [^1]

[^1]: [The Great Cat](https://www.thegreatcat.org/towser/)

<!-- mtoc-start -->

* [Getting Started](#getting-started)
  * [1. Setup Dev Environment](#1-setup-dev-environment)
  * [2. Download and Configure Dataset](#2-download-and-configure-dataset)
  * [3. Download Models](#3-download-models)
  * [4. Run Baseline Test](#4-run-baseline-test)

<!-- mtoc-end -->

## Getting Started

The instructions on the main Github repo are great but I have recently moved to
exclusively using `uv` for my Python development. Here are the steps to set up a
development environment using `uv` and how to programmatically download and
arrange the dataset in the expected format.

### 1. Setup Dev Environment

Assuming you have `uv` installed (if not, a simple `curl -LsSf https://astral.sh/uv/install.sh | sh` will do the trick), then create new Python environment with:

```bash
uv venv --python 3.10 # Note: event-vision-library in `requirements.txt` requires Python >3.7 && <3.11
source .venv/bin/activate
uv pip install pip
```

Install requirements with:

```bash
uv pip install -r requirements
```

### 2. Download and Configure Dataset

Use `gdown` to download the dataset and configure the folder structure to be as
expected:

```bash
uv pip install gdown
mkdir data
gdown --folder https://drive.google.com/drive/folders/1amY4kuaZFWdpgHg4RfTrw9Qb-tKrM-8h -O data/original
```

From the main repository, the folder structure should be as follows:

```console
data/original
│
├── top/
│   ├── train
│   │   ├── seq_02.hdf5
│   │   ├── seq_05.hdf5
│   │   ├── ...
│   │   └── seq_33.hdf5
|   ├── val
│   │   ├── seq_03.hdf5
│   │   ├── seq_04.hdf5
│   │   ├── ...
│   │   └── seq_25.hdf5
│   └── test
│       ├── seq_01.hdf5
│       ├── seq_07.hdf5
│       ├── ...
│       └── seq_32.hdf5
├── dataset_info.csv
├── val_annotations.json
└── train_annotations.json
```

But from the download you would see `train` and `val` sequences are not properly
arranged. This should put the remaining files where everting is expected to be:

```bash
mkdir -p data/original/train
mkdir -p data/original/val
mv data/original/seq_03.h5 data/original/seq_04.h5 data/original/seq_12.h5 data/original/seq_25.h5 data/original/val/
mv data/original/seq_* train/
```

### 3. Download Models

```bash
gdown --folder https://drive.google.com/drive/folders/1-P1HN4FZEy3ETn5rrQiMoDQx3378HLQW -O models
```

### 4. Run Baseline Test

Now the data and folder structure is set up, we can pre-process some data to
play with following the step in the main repo, i.e.

> This preprocessing step is required only when evaluating the ModelMixSort
> method from the paper. It relies on e2vid images reconstructed at the
> grayscale image timesteps.

```bash
python scripts/preprocess_events_to_e2vid_images.py --data_root data/original
```

While in theory the above is possible on the CPU, it is very slow so it's better
to ensure to run on a system with hardware acceleration, be that CUDA on NVIDA
GPUs or Metal for macOS.

```bash
python scripts/preprocess_events_to_e2vid_images.py --data_root data/original

Loading model /Users/tallam/github/tallamjr/forks/msis/.venv/lib/python3.10/site-packages/evlib/processing/reconstruction/../../../../../artifacts/E2VID_lightweight.pth.tar...
Using TransposedConvLayer (fast, with checkerboard artefacts)
2025-02-07 14:52:31,985 [INFO] Device: mps
2025-02-07 14:52:32,014 [INFO] Device: mps
== Image reconstruction ==
Image size: 720x1280
== Event preprocessing ==
Will normalize event tensors.
Processing seq10.h5
100%|██████████████████████████████████████████████████████| 398/398 [01:37<00:00,  4.08it/s]
Loading model /Users/tallam/github/tallamjr/forks/msis/.venv/lib/python3.10/site-packages/evl
...
```

... TBC ...
