# Video Frame Extraction and Contact Sheets

Run these commands from the repository root with the project environment
activated.

```powershell
micromamba activate pot-masking
```

Extract every frame from both pot videos.

```powershell
python scripts/capture/video_frame_extraction.py data/raw/videos/pot1-unglazed.mp4
python scripts/capture/video_frame_extraction.py data/raw/videos/pot2-glazed.mp4
```

Alternatively, extract every third frame from both videos. Use `--overwrite`
when replacing frames created by an earlier extraction run.

```powershell
python scripts/capture/video_frame_extraction.py data/raw/videos/pot1-unglazed.mp4 --every-n 3 --overwrite
python scripts/capture/video_frame_extraction.py data/raw/videos/pot2-glazed.mp4 --every-n 3 --overwrite
```

Create a labeled contact sheet for each pot.

```powershell
python scripts/capture/build_frame_contact_sheet.py data/frames_output/pot1-unglazed_frames data/processed/pot1-unglazed/frame_contact_sheet.jpg
python scripts/capture/build_frame_contact_sheet.py data/frames_output/pot2-glazed_frames data/processed/pot2-glazed/frame_contact_sheet.jpg
```

When no output directory is supplied, extraction writes the images and
`frames_manifest.csv` to
`data/frames_output/<video-name>_frames`. You can still pass an explicit output
directory after the input video to override this default. The contact-sheet
command reads the manifest when available and includes the source frame number
and timestamp in each tile.
