# RAWViewer

A lightweight and smooth photo culling tool designed for photographers.

## Overview

RAWViewer was created to solve the pain point of batch deleting rejected photos during large photo shoots like weddings, events, and sports photography.

### The Problem

In scenarios such as weddings, events, or sports photography, photographers typically capture thousands of photos with many rejected shots. Traditional deletion methods are inefficient:
- Deleting one by one in file explorer risks accidental deletion
- Cannot quickly preview photos marked for deletion
- No way to undo deletions

### The Solution

RAWViewer's core workflow:
1. **Quick Marking**: Mark photos as Reject or Pick using keyboard shortcuts
2. **Batch Preview**: Preview all photos marked for deletion before deleting
3. **One-Click Cleanup**: Batch delete all rejected photos to recycle bin

## Features

### Photo Marking
- `D` key: Mark as "Reject" (to be deleted)
- `` ` `` key: Mark as "Pick" (preferred)
- `U` key: Clear marking

Marking status is displayed as colored corner labels on photos and saved to `.raw_flags.json`.

### Deletion
- Delete current photo
- Batch delete all rejected photos
- Preview dialog before deletion
- Delete confirmation dialog (can be disabled in settings)
- Recycle bin: Deleted files can be recovered

### Focus Scoring
The program automatically analyzes focus point areas using Laplacian variance algorithm to generate sharpness scores.

- Automatic scoring starts when opening a folder
- CSV cache: Results stored in `.raw_scores.csv` to avoid recalculation
- Incremental update: Only new photos are scored

### Focus Point Visualization
Display camera-recorded focus points overlaid on photos.

| Brand | Format | Support |
|-------|--------|---------|
| Sony | .arw | Full support |
| Nikon | .nef | D5/D500/D850/Z series |
| Canon | .cr2/.cr3 | Partial |
| OM System | .orf | OM-1/OM-5 etc. |

### XMP Sync
Marks are written to XMP sidecar files, readable by Lightroom and Capture One.

- Pick mark → Lightroom Pick flag
- Reject mark → Lightroom Reject flag

### Zoom & View
- Fit to window: Auto-scale to fit window size
- 100% view: View at original pixel size
- Zoom lock: Lock zoom level across pages
- Pan & drag: Drag to navigate in 100% view

### Sorting & Filtering
- Sort by score
- Filter by score range
- Filter by mark status (Pick/Reject/Unmarked)

### Thumbnail Navigation
- Bottom thumbnail strip shows all photos in folder
- Thumbnails display score color codes and mark status
- Click thumbnail for quick navigation

### EXIF Information
Displays camera, lens, aperture, shutter speed, ISO, focal length, and more.

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `←` / `→` | Previous / Next |
| `Page Up` / `Page Down` | Previous / Next page |
| `Home` / `End` | First / Last |
| `+` / `-` | Zoom in / out |
| `Space` | Fit window / 100% toggle |
| `` ` `` | Mark Pick |
| `D` | Mark Reject |
| `U` | Clear mark |

## Limitations

- Scoring only applies to focus point areas, non-focused areas are not calculated
- Relies on camera-recorded focus point information, older models may not support
- Scores are for reference only, cannot replace final human judgment

## FAQ

**Q: How to batch delete rejected photos?**

A: Click "View Rejected List" in the menu, then click "Delete All" in the preview dialog.

**Q: What if I accidentally deleted photos?**

A: Deleted photos go to system recycle bin and can be recovered.

**Q: Why some photos don't show focus points?**

A: Some cameras don't record focus point information, this is a hardware limitation.

**Q: Where is the score cache file?**

A: It's in `.raw_scores.csv` in the folder root.

## System Requirements

- Windows 10/11
- Python 3.12+ (if running from source)
- PySide6
- rawpy
- ExifTool
- ultralytics

## License

This tool is designed to improve culling efficiency. Please do not completely rely on it for final decisions. Human review is recommended after initial culling.
