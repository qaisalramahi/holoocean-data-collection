# HoloOcean Data Collection Scripts

This directory contains Python scripts for collecting, processing, and visualizing synthetic underwater sonar and depth datasets using the [HoloOcean](https://github.com/byu-holoocean/HoloOcean) simulator (Unreal Engine 5.3 backend). The pipeline produces multimodal training data for underwater object recognition research: forward-looking sonar (FLS) image sequences, multi-view depth maps, and 3D point clouds, all captured from a BlueROV2 agent around a scene-center target object.

---

## Table of Contents

1. [Project Architecture](#project-architecture)
2. [Prerequisites and Installation](#prerequisites-and-installation)
3. [Simulator Concepts](#simulator-concepts)
4. [Script Reference](#script-reference)
   - [live_view.py — Pose Authoring Tool](#live_viewpy--pose-authoring-tool)
   - [linear_sweep_orientations8_6m.py — Sonar Sweep at 6 m Depth](#linear_sweep_orientations8_6mpy--sonar-sweep-at-6-m-depth)
   - [linear_sweep_orientations8_5m.py — Sonar Sweep at 5 m Depth](#linear_sweep_orientations8_5mpy--sonar-sweep-at-5-m-depth)
   - [depth_image_capture_5m.py — Depth Camera Capture at 5 m](#depth_image_capture_5mpy--depth-camera-capture-at-5-m)
   - [depth_image_capture_6m.py — Depth Camera Capture at 6 m](#depth_image_capture_6mpy--depth-camera-capture-at-6-m)
   - [reconstruct_data_1.py — 3D Point Cloud Reconstruction](#reconstruct_data_1py--3d-point-cloud-reconstruction)
   - [dataset_gui_open3d.py — Dataset Visualization GUI](#dataset_gui_open3dpy--dataset-visualization-gui)
5. [Data Output Structure](#data-output-structure)
6. [Pose Bookmark Format](#pose-bookmark-format)
7. [Engineering Notes](#engineering-notes)

---

## Project Architecture

```
HoloOcean UE5 Engine
        |
        |  shared memory (holoocean Python API)
        v
  Python client scripts
        |
        +-- live_view.py              (pose authoring, bookmark saving)
        |
        +-- linear_sweep_*_6m.py     (sonar sweep data collection, 6 m depth)
        +-- linear_sweep_*_5m.py     (sonar sweep data collection, 5 m depth)
        |
        +-- depth_image_capture_6m.py (depth camera rig capture, 6 m depth)
        +-- depth_image_capture_5m.py (depth camera rig capture, 5 m depth)
        |
        +-- reconstruct_data_1.py    (depth → 3D point cloud)
        +-- dataset_gui_open3d.py    (Tkinter + Open3D visualization GUI)
```

Two agent types appear in the simulation:

- **`auv0`** (`BlueROV2`): The main ROV carrying the imaging sonar. Controlled by an 8-element thruster command array or a 6-element pose PD target `[x, y, z, roll, pitch, yaw]`.
- **`sphere0`** (`HoveringAUV`): A passive "camera rig" carrying eight DepthCameras arranged in an arc. Used only by the depth capture scripts.

The UE5 `PoseSensor` returns a 4×4 homogeneous transform (row-major `float32`) in world coordinates. The `ImagingSonar` sensor returns a `(RangeBins × AzimuthBins)` `float32` intensity array. Each `DepthCamera` returns a dict with a `"depth"` key holding a `float32` 2-D array in metres.

---

## Prerequisites and Installation

**Python version:** 3.10+ recommended (matches HoloOcean's CI).

**Required packages:**

```
holoocean          # HoloOcean Python client (install per project docs)
numpy
opencv-python      # cv2
matplotlib
Pillow             # PIL
pynput             # keyboard listener
scipy              # scipy.io, scipy.spatial.transform
open3d             # 3D point cloud processing and visualization
pandas             # CSV loading in GUI
tkinter            # part of the Python standard library (built-in)
```

Install Python dependencies (excluding `holoocean`, which has its own installation procedure):

```bash
pip install numpy opencv-python matplotlib Pillow pynput scipy open3d pandas
```

The UE5 simulator binary (`HoloOcean`) must be installed and the `Target_object_only` package must be available before running any data-collection script.

---

## Simulator Concepts

### Tick loop

All scripts enter a `while True` loop and call `env.tick()` once per iteration. At 200 ticks/sec (the configured rate), each tick is ~5 ms of simulated time. Sensor data is returned in the `state` dictionary keyed by agent name.

### Control schemes

The BlueROV2 supports two control schemes toggled at runtime via `agent.set_control_scheme()`:

| Scheme | `ControlSchemes` constant | `act()` input |
|---|---|---|
| Thruster | `AUV_THRUSTERS` (0) | `np.ndarray` shape `(8,)` — per-thruster force |
| Pose PD | `AUV_CONTROL` | `np.ndarray` shape `(6,)` — `[x, y, z, roll, pitch, yaw]` goal |

### Sonar physics

The `ImagingSonar` implements acoustic multi-path propagation using an octree accelerator. Key config params:

| Parameter | Meaning |
|---|---|
| `RangeBins` / `AzimuthBins` | Resolution of the returned intensity array |
| `RangeMin` / `RangeMax` | Acoustic gate in metres |
| `Elevation` / `Azimuth` | Beam half-angles in degrees |
| `InitOctreeRange` | Mid-level Octree initialization depth |
| `ScaleNoise`, `AddSigma`, `MultSigma`, `RangeSigma` | Additive and multiplicative noise parameters |
| `MultiPath` | Enable/disable multi-path reflections |
| `AzimuthStreaks` | Streak artifact simulation (-1 = removal artifact) |

---

## Script Reference

---

### live_view.py — Pose Authoring Tool

**Purpose:** Interactive tool for manually placing the BlueROV2 (`auv0`) and the camera rig (`sphere0`) at the 8 desired orientations around the target object, then saving those poses to a JSON bookmark file for use by the data collection scripts.

#### How to run

```bash
python live_view.py
```

The UE5 viewport opens. Use the keyboard to drive the agents and compose each orientation, then save poses.

#### Keyboard controls

| Key | Action |
|---|---|
| `1` | Switch keyboard control to `auv0` (BlueROV2) |
| `2` | Switch keyboard control to `sphere0` (camera rig) |
| `W`/`S` | Forward / backward (horizontal thrusters) |
| `A`/`D` | Strafe left / right |
| `I`/`K` | Ascend / descend (vertical thrusters) |
| `J`/`L` | Yaw left / right |
| `u` | Auto-orbit: step both agents to the next of 8 evenly-spaced orientations (45° increments) |
| `z` | Rebase orbit — latch the current poses as the new orbit origin |
| `b` | Save a bookmark for the current pose of both agents |
| `n` / `m` | Select previous / next bookmark |
| `t` | Teleport both agents to the currently selected bookmark |
| `v` | Print all bookmarks to stdout |
| `r` | Remove the currently selected bookmark |
| `p` | Print current sensor poses (`PoseSonar`, `PoseDepthArcCam4`) to stdout |
| `q` / `Esc` | Quit |

#### Auto-orbit algorithm

When `u` is pressed for the first time after a reset (`z`), the orbit latches:
- `orbit_center` = current rig (`sphere0`) world position
- `orbit_offset0` = ROV position − rig position (the initial radial offset vector)
- `orbit_rig_yaw0`, `orbit_rov_yaw0` = current yaw angles of each agent

On each subsequent `u` press the orbit index increments by 1 mod 8:
```
deg = 45.0 * orbit_idx
new_rov_pos = orbit_center + Rz(deg) @ orbit_offset0
new_rov_yaw = orbit_rov_yaw0 + deg
new_rig_yaw = orbit_rig_yaw0 + deg
```

Both agents are teleported atomically, maintaining a constant radius and orientation relative to the rig center.

#### Bookmark file

Bookmarks are written to `pose_bookmarks_5m.json` (same directory as the script) in the following format (see [Pose Bookmark Format](#pose-bookmark-format) for schema details). The 6m scripts read `pose_bookmarks_6m.json` instead — create that file by renaming or copying.

#### Live depth display

Three depth camera views (cameras 1, 4, 8) are displayed in real-time OpenCV windows with seam correction applied (same `find_seam_rows` + `median_blur_seam` pipeline used by the capture scripts). This lets the operator verify the rig is looking at the target before saving a bookmark.

---

### linear_sweep_orientations8_6m.py — Sonar Sweep at 6 m Depth

**Purpose:** Automated collection of forward-looking sonar image sequences. The BlueROV2 is teleported to each of 8 pre-authored orientations and then commanded to perform a linear sweep, saving one sonar frame every 6 cm of travel.

#### How to run

```bash
python linear_sweep_orientations8_6m.py
```

The script opens a live matplotlib polar sonar display and the UE5 viewport. Use `WASD`/`IJKL` to observe the scene, then press `u` to start the automated sweep.

#### Scenario configuration

| Parameter | Value |
|---|---|
| World | `Target_object_only` |
| Ticks/sec | 200 |
| Main agent spawn | `[4.15, -5.485, -6.140]` (6 m depth) |
| Sonar socket pitch | 30° (downward look) |
| Sonar RangeMin / RangeMax | 0.1 m / 5 m |
| Sonar RangeBins / AzimuthBins | 512 / 512 |
| Sonar Azimuth / Elevation | 60° / 12° |
| ElevationBins | 256 |
| Sonar noise | `AddSigma=0.04`, `MultSigma=0.04`, `RangeSigma=0.01` |
| MultiPath | enabled |

#### State machine

The main loop implements a 3-state machine:

```
TELEOP ──(press u)──> INIT_ORIENT ──(waypoints built)──> SWEEP
                           ^                                 |
                           |____(next orientation)___________|
                           |
                      (all 8 done) ──> TELEOP
```

**TELEOP state**

Thruster commands are generated from the currently pressed keys and sent to `auv0` each tick. The latest sonar frame is cached in `latest_sonar_frame` for display.

**INIT_ORIENT state**

Executes once per orientation:

1. Reads the `rov_pos` and `rov_rot` fields for orientation `k` from `pose_bookmarks_6m.json`.
2. Calls `auv.teleport(pos, rot)` to place the ROV instantly.
3. Runs 5 warmup ticks so physics and rendering settle before reading pose back.
4. Reads the post-teleport pose via `PoseCOM` (4×4 matrix) and extracts a 6-DOF pose with `make_pose6_from_T()`.
5. Computes the sweep direction unit vector `u` (see [Sweep Direction](#sweep-direction-6m)).
6. Switches control scheme to `AUV_CONTROL` (pose PD).
7. Builds a list of waypoints spaced `WAYPOINT_DIST = 0.5` m apart along `u`, covering `SONAR_DIST = 2.2` m total.
8. Transitions to `SWEEP`.

**SWEEP state**

Executes on every tick while sweeping:

1. Commands the ROV to the current waypoint goal via `auv.act(goal_pose_6)`.
2. Reads current position from `PoseCOM[:3, 3]`.
3. Advances `wp_idx` when Euclidean distance to the goal is less than `EPS = 0.25` m.
4. Checks `dist_since_save = ||cur - last_save_pos||`. When ≥ `CAPTURE_STEP = 0.06` m and fewer than 24 frames collected, saves a sonar frame.
5. After all waypoints are exhausted, saves the `.npy` stack and transitions to the next orientation or back to `TELEOP`.

#### Sweep direction (6m)

The sweep direction is the **negative of the ROV's forward vector**:

```python
u = -[cos(yaw), sin(yaw), 0]
```

This means the ROV moves backward (away from the object it is facing). The yaw angle `yaw` is extracted from the `PoseCOM` rotation matrix column as `atan2(R[1,0], R[0,0])`.

#### Sonar rendering pipeline

Each saved sonar frame goes through:

1. Multiply raw `float32` intensity by `GAIN = 2.5` and clip to `[0, 1]`.
2. Update a persistent matplotlib `pcolormesh` on polar axes (theta = azimuth, r = range). The full sonar grid is set up once:
   ```python
   theta = np.linspace(-30, 30, 512) * pi/180   # azimuth half-angle in radians
   r     = np.linspace(0.1, 5.0, 512)            # range in metres
   T, R  = np.meshgrid(theta, r)
   ```
3. Force-render the figure to an RGBA buffer via `fig.canvas.buffer_rgba()`.
4. Crop to the tight bounding box of the polar axes (removing whitespace). The Y-axis is flipped because matplotlib's pixel origin is bottom-left but the buffer is top-left.
5. Extract the red channel `rgba[..., 0]` as an 8-bit grayscale image. Since the colormap is `'gray'`, R = G = B, so any channel works.
6. Resize to `(1500, 1500)` px using PIL LANCZOS resampling.
7. Save as a grayscale JPEG.

The `.npy` stack saves the same frames resized to `(128, 128)`, then replicated across 3 channels: shape `(N, 128, 128, 3)` — suitable for RGB-input CNN models.

Frame numbering counts **down** from 24 (`number = 24 - len(sonar_imgs)`) so that the last frame in the sequence is always `FLSc_1.jpg` and the first is `FLSc_24.jpg`. This is a deliberate design choice to use the final (closest-range) frame as the primary reference.

#### Key parameters

| Parameter | Value | Notes |
|---|---|---|
| `SONAR_DIST` | 2.2 m | Total sweep travel distance |
| `WAYPOINT_DIST` | 0.5 m | Spacing between PD control waypoints |
| `EPS` | 0.25 m | Waypoint arrival threshold (= 0.5 × WAYPOINT_DIST) |
| `CAPTURE_STEP` | 0.06 m | Distance between saved sonar frames |
| `MAX_FRAMES` | 24 | Maximum frames saved per orientation |
| `GAIN` | 2.5 | Sonar intensity multiplier |
| `OBJECT_POS` | `[1.529, -5.265, -5.950]` | Scene center (reference only) |

#### Output

- **Per-frame JPEGs:** `{main_path}/{object}_only/6m_deep/sonar/orient_{k}/FLSc_{n}.jpg` (1500×1500 px, grayscale)
- **Per-orientation NPY stack:** `{main_path}/cluster/objects_inputRGB/{object}_only_6m_deep_orient{k}.npy` (shape `(≤24, 128, 128, 3)`, `uint8`)

where `{main_path}` = `C:\Users\ramah\holooceanv2.0.1\engine\data\single_object_scenarios`.

---

### linear_sweep_orientations8_5m.py — Sonar Sweep at 5 m Depth

**Purpose:** Identical pipeline to the 6m version but adapted for a 5 m depth scenario with a longer sweep distance, coarser capture interval, and different sonar configuration.

This script imports `save_npy_stack` directly from `linear_sweep_orientations8_6m`:

```python
from linear_sweep_orientations8_6m import save_npy_stack
```

Both scripts must therefore be in the same directory.

#### Differences from the 6m version

| Parameter | 5m value | 6m value |
|---|---|---|
| Spawn location | `[5.14, -5.483, -5.140]` | `[4.15, -5.485, -6.140]` |
| Sonar pitch | 34° | 30° |
| `RangeMax` | 8 m | 5 m |
| `InitOctreeRange` | 30 | 20 |
| `ElevationBins` | 185 | 256 |
| `AzimuthStreaks` | -1 (disabled) | 0 |
| Sonar Hz | 12 | 10 |
| `SONAR_DIST` | 3.6 m | 2.2 m |
| `WAYPOINT_DIST` | 0.8 m | 0.5 m |
| `EPS` | 0.4 m (fixed) | 0.25 m (derived) |
| `CAPTURE_STEP` | 0.10 m | 0.06 m |
| Bookmark file | `pose_bookmarks_5m.json` | `pose_bookmarks_6m.json` |

#### Sweep direction (5m)

The sweep direction is computed as the **radial outward vector from the scene center to the ROV**:

```python
u = (object_pos - OBJECT_POS) / norm(object_pos - OBJECT_POS)
```

where `OBJECT_POS = [1.529, -5.265, -5.950]` is the known scene center. This is geometrically equivalent to: "move further away from the target." The Z component is zeroed so the sweep stays at constant depth.

This differs from the 6m version where the direction is derived from the ROV's yaw angle. The 5m approach is more robust when bookmarked poses are slightly off-axis, since the direction always points radially outward regardless of the ROV's heading.

#### Output

- **JPEGs:** `{main_path}/{object}_only/5m_deep/sonar/orient_{k}/FLSc_{n}.jpg`
- **NPY stack:** `{main_path}/cluster/objects_inputRGB/{object}_only_5m_deep_orient{k}.npy`

---

### depth_image_capture_5m.py — Depth Camera Capture at 5 m

**Purpose:** Capture a set of 8 depth images of the target object from a static camera rig (`sphere0`) simultaneously, for each of 8 orientations. Unlike the sonar scripts there is no sweep — the rig is teleported once per orientation and a single snapshot is saved.

#### Camera rig geometry

The `generate_arc_offset()` function builds the `sphere0` sensor configuration. Eight `DepthCamera` sensors are arranged in a horizontal arc:

```
angle_deg = -67.5, -45.0, -22.5, 0.0, +22.5, +45.0, +67.5, +90.0
           (cam 1)                                               (cam 8)
```

For camera `i` at angle `θ` degrees:

```python
x = 3.0 * cos(θ)          # metres, relative to sphere0 COM
y = 3.0 * sin(θ)
rotation = [0.0, 34.0, θ + 180.0]   # pitch 34° downward, yaw faces inward
```

The `+180°` yaw rotates each camera to face the rig center. The 34° pitch tilts each camera downward to look at the scene floor. Each camera captures 256×256 px at 60° horizontal field of view.

Each depth camera also has a paired `RotationSensor` and `LocationSensor` at the same socket offset (for post-hoc pose computation) and a central `PoseSensor` attached to the `SonarSocket` of the rig.

#### How to run

```bash
python depth_image_capture_5m.py
```

Press `u` to start automated capture across all 8 orientations.

#### State machine

This script uses a simpler 2-state machine (no SWEEP phase):

```
TELEOP ──(press u)──> INIT_ORIENT ──(saved)──> INIT_ORIENT ──...──> TELEOP
```

For each orientation `k`:

1. Reads `rov_pos`, `rov_rot`, `sphere_pos`, `sphere_rot` from `pose_bookmarks_5m.json`.
2. Teleports both `auv0` and `sphere0` simultaneously.
3. Runs 5 warmup ticks.
4. Calls `save_depth_data(state, main_path, k+1, "{object}_only_5m_deep")`.
5. Advances to the next orientation (or returns to TELEOP).

#### Depth post-processing pipeline

`save_depth_data()` processes all 8 `DepthArcCam{i}` sensors in `sphere0`:

**Step 1 — Range clipping (`depth_max_range`)**

```python
DEPTH_MIN = 0.5  # metres
DEPTH_MAX = 8.0  # metres
mask = (~isfinite(d)) | (d < DEPTH_MIN) | (d > DEPTH_MAX)
d = clip(d, DEPTH_MIN, DEPTH_MAX)
d[mask] = 0   # out-of-range pixels become black
```

Pixels outside the valid range (including `inf` and `nan` returned by UE5 for rays that hit nothing) are zeroed out rather than left as invalid floats.

**Step 2 — Seam detection (`find_seam_rows`)**

The UE5 depth buffer can exhibit horizontal discontinuity artifacts at certain row boundaries due to internal rendering tiling. The seam detector identifies these:

```python
vdiff = |depth[1:, :] - depth[:-1, :]|   # shape (H-1, W) row-to-row differences
score = nanmedian(vdiff, axis=1)           # one score per row boundary (H-1 values)
base  = nanmedian(score) + 1e-6            # global baseline jump size
seam_rows = where(score > 5.0 * base)[0] + 1
```

The `+1` converts from boundary index (between row `r` and `r+1`) to the absolute row index of the lower row. A seam boundary is flagged when its median column-wise jump is more than 5× the global median jump — a robust, parameter-free relative threshold.

**Step 3 — Seam smoothing (`median_blur_seam`)**

```python
rmin = max(min(seam_rows) - margin, 0)
rmax = min(max(seam_rows) + margin, H - 1)
roi  = depth[rmin:rmax+1, :]
roi  = cv2.medianBlur(roi.astype(float32), 3)
depth[rmin:rmax+1, :] = roi
```

A 3×3 median filter is applied to a band of `margin=1` rows above and below the detected seam cluster. The median filter is artifact-preserving for edge structures while removing the sharp tiling discontinuity.

**Step 4 — Stack and save**

```python
Z = np.stack([mat_data[f"DepthArcCam{i}"] for i in range(1, 9)], axis=0)
# Z.shape = (8, 256, 256), dtype float32, values in [0, 8.0] metres
sio.savemat(mat_path, {"Z": Z})
```

The 8 depth maps are stacked in camera index order (1–8) into a single MATLAB `.mat` file under the key `"Z"`.

#### Output

- **MAT file:** `{main_path}/cluster/objects_depth_fixed8/{object}_only_5m_deep_depth_orient{k}.mat`
  - Key `"Z"`: shape `(8, 256, 256)`, `float32`, depth in metres

where `{main_path}` = `C:\Users\ramah\holooceanv2.0.1\engine\data\single_object_scenarios`.

---

### depth_image_capture_6m.py — Depth Camera Capture at 6 m

**Purpose:** Identical to `depth_image_capture_5m.py` but configured for 6 m depth. The pipeline, state machine, and output format are the same.

#### Differences from the 5m version

| Aspect | 5m | 6m |
|---|---|---|
| Camera pitch | 34° | 30° |
| `PoseSensor` socket on rig | `"SonarSocket"` | `"COM"` |
| ROV spawn location | `[4.15, -5.485, -6.140]` | `[8.15, -5.485, -6.140]` |
| Bookmark file | `pose_bookmarks_5m.json` | `pose_bookmarks_6m.json` |
| `yaw_from_T()` euler order | `"zyx"`, radians | `"xyz"`, degrees |
| Seam blur margin | 1 row | 3 rows |
| Scenario name passed to save | `"{object}_only_5m_deep"` | `"{object}_only_6m_deep"` |

Note that `yaw_from_T()` is defined in both scripts but is not called in the main capture path (it appears in commented-out debug code). The difference in euler convention is therefore not functionally active in the current code.

#### Output

- **MAT file:** `{main_path}/cluster/objects_depth_fixed8/{object}_only_6m_deep_depth_orient{k}.mat`

---

### reconstruct_data_1.py — 3D Point Cloud Reconstruction

**Purpose:** Library module. Provides the `reconstruct3d()` function that back-projects a depth map into a 3D point cloud in world coordinates using pinhole camera geometry.

#### API

```python
pcd, pts_world = reconstruct3d(
    image,          # HxW RGB numpy array, or None
    depthmap,       # HxW float32 depth in metres
    x, y, z,        # camera position in world frame (metres)
    yaw,            # camera yaw in radians
    camera_params,  # 3x3 camera intrinsics matrix K
    step=1,         # pixel subsampling stride (1 = all pixels)
    roll=0.0,       # camera roll in radians
    pitch=0.0,      # camera pitch in radians
    depth_min=1e-6, # discard pixels with depth below this
    depth_max=None, # discard pixels with depth above this (None = no limit)
)
```

Returns:
- `pcd`: `open3d.geometry.PointCloud` with colors if `image` is provided
- `pts_world`: `(N, 3)` `float64` numpy array of world-frame 3D points

#### Camera model (X-forward convention)

The depth maps come from UE5's DepthCamera which uses an X-forward coordinate frame (X = depth axis, Y = right, Z = up). Ray directions in camera space are:

```python
dirs = np.stack([
    np.ones_like(d),       # X: depth axis (forward)
    (u - cx) / fx,         # Y: right
    -(v - cy) / fy         # Z: up (note negation — pixel v increases downward)
], axis=1)
```

3D points in camera frame are obtained by scaling each ray direction by the scalar depth value:

```python
pts_cam = dirs * d[:, None]   # (N, 3)
```

#### World transform

The camera-to-world rotation is built from `scipy.spatial.transform.Rotation` using an `"xyz"` Euler sequence (roll, pitch, yaw) in **radians**:

```python
R_wc = Rot.from_euler("xyz", [roll, pitch, yaw], degrees=False).as_matrix()
t_wc = np.array([x, y, z])
pts_world = (R_wc @ pts_cam.T).T + t_wc
```

This assumes the pose CSV files produced by the capture scripts store angles in radians, which matches the (currently commented-out) CSV-writing code in the depth capture scripts.

---

### dataset_gui_open3d.py — Dataset Visualization GUI

**Purpose:** Tkinter desktop application (`SonarCloudViz`) for browsing and inspecting a collected dataset. Loads sonar JPEGs, depth `.mat` files, and reconstructed 3D point clouds in a single window.

#### How to run

```bash
python dataset_gui_open3d.py
```

The GUI opens. Use **File > Open** (or `Ctrl+O`, or the "Open Folder" button) to select a dataset folder.

**Folder selection logic:** If you select an `orient_{k}` subfolder (e.g., `.../.../5m_deep/sonar/orient_3`), the GUI auto-detects `orient_idx = 3`, infers the object name, depth level, and dataset root from the path. If you select the dataset root directly, it defaults to `orient_idx = 1`.

#### UI layout

```
[Open Folder]  [Clear]  [Open 3D Viewer]
─────────────────────────────────────────
 Sonar image row 1  (FLSc_1 … FLSc_12)  100×100 px thumbnails
 Sonar image row 2  (FLSc_13 … FLSc_24)
─────────────────────────────────────────
 Depth camera thumbnails  (cam1 … cam8)  150×150 px
─────────────────────────────────────────
 [Point cloud status message / "Open 3D Viewer"]
```

#### Point cloud reconstruction flow (`display_pointcloud`)

1. Loads the `(8, 256, 256)` depth Z-stack from the `.mat` file.
2. Computes camera intrinsics from the known 60° horizontal FoV:
   ```python
   hfov = radians(60.0)
   fx = W / (2 * tan(hfov / 2))
   fy = H / (2 * tan(hfov / 2))    # square pixels assumed
   cx, cy = W/2, H/2
   ```
3. For each camera `i` (1–8):
   - Reads depth slice `Z[8 - i]` (reversed index aligns the physical camera ordering).
   - Reads the camera pose from `{depth_level}_poses/orient_{k}/pose_{i}.csv` (6 values: `x, y, z, roll, pitch, yaw` in radians).
   - Loads a reference RGB image from `rgb/{i}.jpg`.
   - Calls `reconstruct.reconstruct3d()` to get `pts_i` (Nx3 world-frame points).
4. Stacks all 8 clouds: `combined = np.vstack(pointcloud_v8)`.
5. Applies a −90° rotation around the X-axis to align the UE5 Z-up convention with Open3D's Y-up convention:
   ```python
   R_x = [[1, 0, 0], [0, cos(-90°), -sin(-90°)], [0, sin(-90°), cos(-90°)]]
   rotated = combined @ R_x.T
   ```
6. Creates an Open3D `PointCloud`, applies statistical outlier removal (`nb_neighbors=10`, `std_ratio=2.0`).
7. Colors each point by its distance to the nearest camera origin, using the `viridis_r` colormap with robust 2nd–98th percentile normalization:
   ```python
   d = min_distance_to_any_camera_origin   # per-point
   lo, hi = percentile(d, [2, 98])
   d_norm = clip((d - lo) / (hi - lo), 0, 1)
   colors = viridis_r(d_norm)[:, :3]
   ```
   Points closer to the cameras (typically the object surface) appear yellow; distant background points appear purple.

#### 3D viewer (subprocess isolation)

The Open3D interactive window is launched in a **child subprocess** rather than in the main Tkinter process:

```python
viewer_code = """
import open3d as o3d
pcd = o3d.io.read_point_cloud(ply_path)
vis = o3d.visualization.Visualizer()
vis.create_window(...)
...
"""
subprocess.run([sys.executable, "-c", viewer_code])
```

The point cloud is serialized to a temporary `.ply` file and the child process reads it independently. This design avoids X11/Tk event loop conflicts that arise when Open3D and Tkinter both try to manage a native window in the same process. The GUI remains responsive while the 3D viewer is open, though loading a new dataset is blocked until the viewer is closed.

#### Expected directory structure for the GUI

```
{dataset_root}/
  {object_name}/
    {depth_level}/            e.g., "5m_deep"
      sonar/
        orient_{k}/
          FLSc_1.jpg
          FLSc_2.jpg
          ...
          FLSc_24.jpg
  cluster/
    objects_depth_fixed8/
      {object}_{depth_level}_depth_orient{k}.mat

{cwd}/                        (working directory when the GUI is launched)
  {depth_level}_poses/        e.g., "5m_deep_poses"
    orient_{k}/
      pose_1.csv              6 rows × 1 col: x, y, z, roll, pitch, yaw (radians)
      pose_2.csv
      ...
      pose_8.csv
  rgb/
    1.jpg
    2.jpg
    ...
    8.jpg
```

---

## Data Output Structure

```
engine/data/single_object_scenarios/
  {object}_only/
    5m_deep/
      sonar/
        orient_1/  FLSc_1.jpg … FLSc_24.jpg    (1500×1500 px, grayscale)
        orient_2/
        ...
        orient_8/
    6m_deep/
      sonar/
        orient_1/  FLSc_1.jpg … FLSc_24.jpg
        ...
        orient_8/
  cluster/
    objects_inputRGB/
      {object}_only_5m_deep_orient1.npy         (≤24, 128, 128, 3) uint8
      {object}_only_5m_deep_orient2.npy
      ...
      {object}_only_6m_deep_orient1.npy
      ...
    objects_depth_fixed8/
      {object}_only_5m_deep_depth_orient1.mat   Z: (8, 256, 256) float32 metres
      ...
      {object}_only_6m_deep_depth_orient1.mat
      ...
```

---

## Pose Bookmark Format

Bookmark files (`pose_bookmarks_5m.json`, `pose_bookmarks_6m.json`) follow this schema:

```json
{
  "version": 1,
  "saved_at": "2025-01-15T10:30:00+00:00",
  "bookmarks": [
    {
      "id": 0,
      "timestamp_utc": "2025-01-15T10:30:00+00:00",
      "rov": {
        "pos": [x, y, z],
        "rpy": [roll_deg, pitch_deg, yaw_deg]
      },
      "rig": {
        "pos": [x, y, z],
        "rpy": [roll_deg, pitch_deg, yaw_deg]
      }
    },
    ...
  ]
}
```

All position values are in **metres** in the UE5 world frame. Rotation values are in **degrees** (as returned by `Rotation.as_euler("xyz", degrees=True)`). The `"rig"` key is present when bookmarks are saved using `live_view.py`'s `save_current_bookmark()` function; it is absent in older bookmark files which only contain `"rov"`. The depth capture scripts require `"rig"` to be present; the sonar sweep scripts only require `"rov"`.

The loader in all scripts handles both a raw list `[...]` and the wrapped `{"bookmarks": [...]}` format for backward compatibility.

---

## Engineering Notes

### Why two sweep direction strategies?

The 6m script uses the ROV's yaw as the sweep axis (backward from where it is pointing) while the 5m script uses the geometric vector from the object center to the ROV. Both achieve the same goal — moving the sonar away from the target —  but I realised the PD control during the 6m was swaying left and right too much and so I decided making that change in design.

### Why does frame numbering count down?

In `linear_sweep_orientations8_6m.py` and `_5m.py`, `number = MAX_FRAMES - len(sonar_imgs)` means the first captured frame is saved as `FLSc_24.jpg` and the last is `FLSc_1.jpg`. This was deliberately done by design in order to have the final frame be the closest one to the target object 

### Depth seam artifacts

UE5's depth buffer rendering has a known artifact where adjacent internal render tiles produce a visible horizontal seam row in the depth output. The `find_seam_rows` + `median_blur_seam` pipeline is designed to detect and suppress these automatically without requiring manual inspection of each captured image. The threshold multiplier `k=5.0` was chosen empirically: it is sensitive enough to catch real seams (which typically produce jumps 10–50× the background noise level) while not triggering on genuine depth edges in the scene.

### Subprocess isolation for Open3D

On Linux/X11 systems, mixing a Tkinter event loop with an Open3D `Visualizer` window in the same process causes `BadWindow` X errors when Tkinter destroys and recreates widgets while Open3D holds a reference to an X11 surface. The subprocess approach in `dataset_gui_open3d.py` sidesteps this by giving Open3D exclusive ownership of its X11 connection in a child process. The `.ply` handoff through the filesystem is cheap (a few MB for typical point clouds) and adds negligible latency.

### Coordinate frame conventions

| Frame | X | Y | Z |
|---|---|---|---|
| UE5 world | forward (north) | right (east) | up |
| `PoseSensor` output | 4×4 transform, same convention | | |
| Depth camera | depth (forward) | right | up (negated v) |
| `reconstruct3d` input yaw | rotation around world Z (up) axis, radians | | |
| Open3D display | right-hand Y-up | | |

The −90° X-axis rotation in `display_pointcloud` converts from UE5's Z-up to Open3D's Y-up convention.
