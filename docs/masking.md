# Pot Background Removal and Mask Generation

This tool starts from a folder produced by `video_frame_extraction.py`. It keeps
the source photographs unchanged and creates pot-only masks, COLMAP masks,
transparent PNGs, review overlays, and a quality-control report.

The matte terracotta pot and glossy dragon jar use silhouette tracking rather
than blue-screen removal. No color correction is applied. Blue reflections on
the glossy jar remain part of the jar.

## 1. Environment setup

Create the light-weight environment first.

```powershell
micromamba create -f environment-masking.yml -y
micromamba activate pot-masking
```

Install a platform-appropriate PyTorch build with Python 3.12. PyTorch 2.5.1
or newer and its matching TorchVision version are required by SAM 2. Follow the
[official PyTorch installer](https://pytorch.org/get-started/locally/) so the
NVIDIA machine receives a CUDA build while the M3 Air receives the macOS build.

Clone and install Meta's official SAM 2 repository. On Windows, WSL 2 with
Ubuntu is the recommended processing environment.

```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
```

On macOS, disable the CUDA extension during installation.

```bash
SAM2_BUILD_CUDA=0 pip install -e .
```

Download only the SAM 2.1 Hiera Tiny checkpoint from Meta and place it here.

```text
models/sam2.1_hiera_tiny.pt
```

Official checkpoint URL

```text
https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
```

Check the installation. Add `--full` to load the model and run a small
inference test.

```powershell
python scripts/masking/mask_dataset.py doctor
python scripts/masking/mask_dataset.py doctor --full
```

The program automatically prefers CUDA, then Apple MPS, then CPU. Use
`--device cuda`, `--device mps`, or `--device cpu` to force one backend.

## 2. Annotate a pot

Run annotation in a desktop environment with GUI support. Draw a tight box
around the pot only. Do not include the white turntable. Left-click adds an
optional positive pot point. Right-click adds a negative background point. A
negative point on the white turntable is recommended.

```powershell
python scripts/masking/mask_dataset.py annotate `
  data/interim/pot_unglazed_1_n3_frames `
  data/processed/pot_unglazed_1_masks `
  --material matte
```

```powershell
python scripts/masking/mask_dataset.py annotate `
  data/interim/pot2-glazed-n3_frames `
  data/processed/pot2-glazed-masks `
  --material glossy
```

For a machine without a GUI, supply original-image pixel coordinates.

```powershell
python scripts/masking/mask_dataset.py annotate INPUT_FRAMES OUTPUT `
  --material matte `
  --box X0 Y0 X1 Y1 `
  --positive X Y `
  --negative X Y
```

The first annotation must be on frame zero. Later correction prompts may use an
index or exact filename.

```powershell
python scripts/masking/mask_dataset.py annotate INPUT_FRAMES OUTPUT `
  --material glossy `
  --frame frame_000600.jpg
```

## 3. Generate masks and QC files

```powershell
python scripts/masking/mask_dataset.py process INPUT_FRAMES OUTPUT --device auto
```

The default run uses 120-frame chunks with an eight-frame overlap and erodes
the COLMAP masks by three pixels. On the 4 GB GTX 1650, video frames and model
state are automatically offloaded to CPU memory.

Rerunning after a correction requires explicit permission to replace derived
outputs. `prompts.json` and all source images are preserved.

```powershell
python scripts/masking/mask_dataset.py process INPUT_FRAMES OUTPUT `
  --device auto `
  --overwrite
```

The COLMAP input remains the original RGB frame directory. Pass
`OUTPUT/masks_colmap` as COLMAP's mask path. Do not give COLMAP the files in the
`rgba` directory.

## 4. Review and correct masks

Print a list of flagged frames without opening a window.

```powershell
python scripts/masking/mask_dataset.py review INPUT_FRAMES OUTPUT --summary
```

Open the interactive reviewer.

```powershell
python scripts/masking/mask_dataset.py review INPUT_FRAMES OUTPUT
```

Reviewer keys

- `N` or Space moves forward
- `P` moves backward
- `A` accepts the displayed frame
- `C` records a replacement box and optional clicks for that frame
- `Q` saves and exits

After adding corrections, rerun `process` with `--overwrite` and review the new
QC contact sheet.

## Output meaning

- `masks_object` contains full pot-only binary masks
- `masks_colmap` contains three-pixel-eroded masks named for COLMAP
- `rgba` contains unmodified source RGB plus binary alpha
- `overlays` contains smaller visual-review images
- `mask_manifest.csv` contains measurements and QC warnings for every frame
- `run_metadata.json` records the checkpoint hash, environment, hardware, and settings
- `qc_contact_sheet.jpg` contains all flagged frames plus every twentieth frame
