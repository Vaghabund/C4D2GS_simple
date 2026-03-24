# c4d2gs_lite Script Workflow

This repository includes a fast Cinema 4D script workflow to generate synthetic COLMAP data for Gaussian Splatting pipelines.

## What It Exports

Running the script produces:

- Synthetic COLMAP data files: `cameras.txt`, `images.txt`, `points3D.txt`
- Rendered image sequence from the animated render camera
- Optional `camera_poses.json` for NeRF-style tooling

## Quick Usage

1. Open Cinema 4D and select your target object.
2. Open [c4d2gs_lite.py](c4d2gs_lite.py) in Cinema 4D Script Manager.
3. Set your output path and basic capture settings near the top of the script.
4. Run the script to build the rig and export synthetic COLMAP data. Make sure your target object is selected in the object-manager; the generated cameras are always aimed at the selected object’s center axis.
5. Render the animation to output your frame sequence.
6. Import the synthetic COLMAP data and rendered images into your reconstruction/training app.

## Output Layout

Typical output folder structure:

- `cameras.txt`
- `images.txt`
- `points3D.txt`
- `camera_poses.json` (optional)
- `images/gs_0000.png`, `images/gs_0001.png`, ...

## Note About the Full Plugin

If you want a cleaner UI, richer controls, and a more guided production workflow, use the full C4D2GS plugin in [http://Vaghabund.gumroad.com/l/c4d2gs](c4d2gs). It is the easiest way to generate synthetic COLMAP data at scale with fewer manual steps.
