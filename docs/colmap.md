# Masked COLMAP Sparse Reconstruction

This stage uses every-sixth RGB frames and their SAM 2 masks to estimate camera
poses and a sparse point cloud. All reconstruction products remain under the
project's `data/processed` directory.

## Inputs

```text
data/frames_output/pot1-unglazed_every6_frames
data/processed/pot1-unglazed_every6/masks_colmap
```

COLMAP requires a mask to keep the source image's complete filename and append
`.png`. For example, `frame_000006.jpg` uses `frame_000006.jpg.png`. Black mask
pixels suppress feature extraction; non-black pixels retain features.

## Configuration

Edit `configs/colmap_pot1_unglazed_every6.yml`. The default configuration uses:

- `C:/Tools/COLMAP-4.1.1/bin/colmap.exe`
- the CUDA feature extractor and matcher
- one shared `SIMPLE_RADIAL` camera
- standard GPU SIFT features
- sequential guided matching with an overlap of 15
- incremental sparse mapping

The configuration invokes `bin/colmap.exe` directly. The distributed Windows
`COLMAP.bat` wrapper expands arguments through `cmd.exe` and cannot safely pass
this repository's `(Computer Vision)` path to COLMAP.

### Feature-memory profiles

Select a profile with `feature_extraction.memory_profile`:

| Profile | Maximum image size | First octave | Workers | Use case |
| --- | ---: | ---: | ---: | --- |
| `quality_gpu` | Original | -1 | 1 | Maximum detail when GPU memory is ample |
| `balanced_gpu` | 3200 px | 0 | 1 | Default for the 4 GB GTX 1650 |
| `low_memory_gpu` | 2400 px | 0 | 1 | Fallback if the balanced profile fails |

The balanced profile reduces temporary image-pyramid memory while retaining the
8192 strongest features. Affine-shape estimation and domain-size pooling remain
disabled because enabling either forces COLMAP's covariant CPU extractor, even
when CUDA is requested. The CPU implementation can allocate substantial memory
per worker for the 2160×3840 inputs.

For a different dataset, copy the YAML file and change only the input, output,
and reconstruction parameters. Pass the new file with `--config`.

## Environment

Update and activate the project environment after pulling this change.

```powershell
micromamba env update -n pot-masking -f environment-masking.yml
micromamba activate pot-masking
```

## Validate without running COLMAP

Run from the repository root:

```powershell
python scripts/reconstruction/run_colmap.py --dry-run
```

The dry run checks the COLMAP installation, RGB images, exact mask mapping,
project path boundary, and existing outputs. It then prints the three COLMAP
commands without creating files.

## Matching modes and reconstruction commands

The YAML file contains the completed exhaustive settings as comments and keeps
the successful sequential settings active. YAML ignores lines beginning with
`#`, so the old values remain available for reference.

### Exhaustive matching

Use these YAML values for an exhaustive run:

```yaml
output:
  workspace: data/processed/pot1-unglazed_every6/colmap_sparse_masked_exhaustion
  database: database.db
  sparse: sparse

matching:
  method: exhaustive
  guided_matching: true
  sequential_overlap: 10
```

Run from the repository root:

```powershell
python scripts/reconstruction/run_colmap.py --dry-run
python scripts/reconstruction/run_colmap.py
```

The completed exhaustive result is stored at:

```text
data/processed/pot1-unglazed_every6/colmap_sparse_masked_exhaustion
```

Open its largest submodel, model 5, in the GUI:

```powershell
$env:QT_PLUGIN_PATH="C:\Tools\COLMAP-4.1.1\plugins"; & "C:\Tools\COLMAP-4.1.1\bin\colmap.exe" gui --database_path "data\processed\pot1-unglazed_every6\colmap_sparse_masked_exhaustion\database.db" --image_path "data\frames_output\pot1-unglazed_every6_frames" --import_path "data\processed\pot1-unglazed_every6\colmap_sparse_masked_exhaustion\sparse\5"
```

### Sequential matching

These are the active and recommended YAML values for the ordered video frames:

```yaml
output:
  workspace: data/processed/pot1-unglazed_every6/colmap_sparse_masked_sequential
  database: database.db
  sparse: sparse

matching:
  method: sequential
  guided_matching: true
  sequential_overlap: 15
```

Run from the repository root:

```powershell
python scripts/reconstruction/run_colmap.py --dry-run
python scripts/reconstruction/run_colmap.py
```

The completed sequential result is stored at:

```text
data/processed/pot1-unglazed_every6/colmap_sparse_masked_sequential
```

Open its successful model 0 in the GUI:

```powershell
$env:QT_PLUGIN_PATH="C:\Tools\COLMAP-4.1.1\plugins"; & "C:\Tools\COLMAP-4.1.1\bin\colmap.exe" gui --database_path "data\processed\pot1-unglazed_every6\colmap_sparse_masked_sequential\database.db" --image_path "data\frames_output\pot1-unglazed_every6_frames" --import_path "data\processed\pot1-unglazed_every6\colmap_sparse_masked_sequential\sparse\0"
```

Each workspace contains `database.db`, one or more sparse model directories,
and a PLY export for each model. The runner also prints COLMAP's model analysis.

The runner deliberately refuses to overwrite an existing database or non-empty
sparse output. Both paths above contain completed reconstructions. To rerun a
mode, first choose a new workspace suffix such as `_rerun`; do not overwrite the
completed results.

Official references:

- <https://colmap.github.io/install.html>
- <https://colmap.github.io/faq.html#mask-image-regions>
- <https://colmap.github.io/cli.html>
