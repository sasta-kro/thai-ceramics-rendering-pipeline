
```text
project/resources/video-sample-2.mp4
```

From the repository root, run:

```bash
conda activate csx4213-CV
```

Extract every frame:

```bash
python project/scripts/extract_video_frames.py \
  project/resources/video-sample-2.mp4 \
  project/scripts/outputs/video-sample-2_frames
```

Create the mega contact sheet:

```bash
python project/scripts/build_frame_contact_sheet.py \
  project/scripts/outputs/video-sample-2_frames \
  project/scripts/outputs/video-sample-2_contact_sheet.jpg
```

If need to rerun and replace existing output:

```bash
python project/scripts/extract_video_frames.py \
  project/resources/video-sample-2.mp4 \
  project/scripts/outputs/video-sample-2_frames \
  --overwrite
```

For every third frame instead of all frames:

```bash
python project/scripts/extract_video_frames.py \
  project/resources/video-sample-2.mp4 \
  project/scripts/outputs/video-sample-2_frames \
  --every-n 3 \
  --overwrite
```