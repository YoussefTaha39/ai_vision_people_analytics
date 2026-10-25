# AI Vision & People Analytics — Dual-Screen Desktop Edition

A Windows desktop application with **two independent native windows**. No Streamlit and no browser dashboard.

- **Window 1 — Camera & Visitor Registration:** camera feed, person detection, tracking IDs, and add/delete visitor photos.
- **Window 2 — Statistics & Settings:** live metrics, occupancy chart, event table, GPU/AI settings, and Excel/CSV exports.
- When Windows detects two monitors, each app window is placed on a separate monitor. With one monitor, the windows open side-by-side.
- The yellow vertical center line from the earlier version has been removed entirely.

## Main features
1. YOLO11n person detection on laptop webcam with bounding boxes and current-frame count.
2. ByteTrack track IDs and short-lived box persistence to reduce flicker.
3. Optional local face identification for pre-registered visitors using OpenCV Zoo YuNet/SFace ONNX models.
4. Local SQLite database for visitors, recognition events, track events, settings, and occupancy samples.
5. Built-in native occupancy line chart, recent event table, Excel workbook export (Events, Occupancy, Visitors), and CSV export.
6. Settings stored locally and applied immediately; if the camera is running, applying settings restarts the camera worker.
7. GPU auto-detection through PyTorch; CPU fallback if CUDA is unavailable.

## Recommended setup (Windows 10/11, Python 3.12 x64)
1. Extract the ZIP into a normal writable folder (not inside the ZIP preview).
2. Double-click `setup_windows.bat`.
3. Choose:
   - `1` for NVIDIA GPU (CUDA 12.8 PyTorch wheel; if it fails, use the official PyTorch selector below).
   - `2` for CPU.
   - `3` only if PyTorch is already installed correctly in this `.venv`.
4. After setup finishes, double-click `run_windows.bat`.

Official PyTorch installer: https://pytorch.org/get-started/locally/
The exact supported CUDA wheel can change over time; use the official selector if the included CUDA 12.8 command fails.

## First launch
- The first launch may download `yolo11n.pt` automatically.
- In the Camera & Visitor Registration window, click `Download / rebuild face templates` (internet required).
- Add each visitor with a clear photo containing exactly one visible face, a name, and the appropriate consent/authorization checkbox.
- In Statistics & Settings, enable `Identify registered visitors`, select a similarity threshold, and click `Save & apply settings`.
- If face models were installed after visitors were registered, the rebuild operation creates templates for existing visitor photos too.

## Two-monitor layout
- If two displays are connected and enabled in Windows Display Settings, the Camera window opens on the primary display and Statistics & Settings opens on the second display.
- If only one display is detected, the windows open tiled side-by-side. Move/maximize them as desired.
- Close either window to exit the application and stop the camera.

## Important behavior / limitations
- **No yellow center line is drawn.** This build intentionally does not claim doorway entry/exit counts; such counts require a configured and tested doorway/gate line.
- The occupancy graph shows people detected in the webcam field of view, sampled about every two seconds. It is not a count of everybody in a room outside the camera view.
- Track IDs can change after long occlusion or when people cross paths. A short grace period reduces flicker but cannot guarantee perfect tracking.
- A displayed name means the face matcher exceeded its configured similarity threshold; it is not proof of identity. Incorrect matches are possible. Test with your visitor photos and lighting before operational use.
- Face matching is disabled by default. Only register faces when the participants have given appropriate consent/authorization. Photos, embeddings, and analytics are stored locally. The app does not save the camera video.
- For production use, review third-party model licenses and applicable privacy/data-protection requirements.

## Data locations
- `data/vision_analytics.sqlite3` — visitors/settings/events/occupancy.
- `visitor_photos/` — local copies of photos.
- `models/` — downloaded face ONNX models.
- YOLO weights are managed by Ultralytics in its standard model cache.
- Keep `data`, `visitor_photos`, and `models` when updating code if you want to preserve the local records.

## Troubleshooting
- If no camera opens, close Teams/Zoom/browser camera tabs and try Camera index 1 in Statistics & Settings.
- If the image is slow, use inference size 416 or 512.
- If `CPU fallback` appears on the status bar, run `gpu_check.py`. If `torch.cuda.is_available()` is `False`, install a compatible GPU wheel via the official PyTorch selector.
- If the face matcher says no template is available, download/rebuild face templates and check the Face template column.
- If a photo is skipped during rebuild, use a clearer single-person portrait and ensure the face is large enough.
