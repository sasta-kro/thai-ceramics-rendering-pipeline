# Video Frame Extraction and Contact Sheets

Run these commands from the repository root with the project environment
activated.

```powershell
micromamba activate pot-masking
```

Extract every frame from a video.

```powershell
python scripts/capture/video_frame_extraction.py data/raw/videos/video-sample-2.mp4 data/interim/video-sample-2_frames
```

Extract every third frame.

```powershell
python scripts/capture/video_frame_extraction.py data/raw/videos/video-sample-2.mp4 data/interim/video-sample-2_frames --every-n 3 --overwrite
```

Create a labeled contact sheet from the extracted frames.

```powershell
python scripts/capture/build_frame_contact_sheet.py data/interim/video-sample-2_frames data/results/video-sample-2_contact-sheet.jpg
```

The extraction command writes `frames_manifest.csv` beside the frames. The
contact-sheet command reads that manifest when available and includes the
source frame number and timestamp in each tile.
