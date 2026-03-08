# For objects that are smaller than 40cm, we want to capture depth images from a closer distance 
# (1.5m instead of 3m) to get better resolution on the object. 
import holoocean
import numpy as np
import math
import cv2 
import matplotlib.pyplot as plt
from pynput import keyboard
from holoocean.agents import ControlSchemes
import os
from PIL import Image
import scipy.io as sio 
import csv
from scipy.spatial.transform import Rotation as Rot

def generate_arc_offset(
    sensor_type="DepthCamera",
    base_name="DepthArcCam",
    socket="SonarSocket",
    radius=3,
    num_cameras=8,
    angle_increment=22.5,
    start_angle=-67.5,
):
    cameras = []
    for i in range(num_cameras):
        angle_deg = start_angle + i * angle_increment
        angle_rad = math.radians(angle_deg)
        x = radius * math.cos(angle_rad)
        y = radius * math.sin(angle_rad)

        cameras.append(
            {
                "sensor_type": sensor_type,
                "socket": socket,
                "location": [x, y, 0.0],
                "rotation": [0.0, 34.0, angle_deg + 180.0],
                "sensor_name": f"{base_name}{i+1}",
                "Hz": 200, 
                "configuration": {
                    "CaptureWidth": 256,
                    "CaptureHeight": 256,
                    "FovAngle": 60,
                },
            }
        )
        cameras.append(
            {
                "sensor_type": "RotationSensor",
                "socket": socket,
                "location": [x, y, 0.0],
                "rotation": [0.0, 34.0, angle_deg + 180.0],
                "sensor_name": f"RotArcCam{i+1}",
                "Hz": 200,
            }
        )
        cameras.append(
            {
                "sensor_type": "LocationSensor",
                "socket": socket,
                "location": [x, y, 0.0],
                "rotation": [0.0, 34.0, angle_deg + 180.0],
                "sensor_name": f"LocArcCam{i+1}",
                "Hz": 200,
            }
        )
    cameras.append(
        {
            "sensor_type": "PoseSensor",
            "socket": "SonarSocket",
            "sensor_name": "PoseHoveringAUV",
            "Hz": 200,
        }
    )
    return cameras

config = {
    "name": "imaging_depth_sweep",
    "world": "Target_object_only",
    "package_name": "Target_object_only",
    "main_agent": "auv0",
    "ticks_per_sec": 200,
    "frames_per_sec": False,
    "octree_min": 0.008,
    "octree_max": 5.0,
    "agents": [
        {
            "agent_name": "auv0",
            "agent_type": "BlueROV2",
            "sensors": [
                {
                    "sensor_type": "PoseSensor",
                    "socket": "COM",
                    "sensor_name": "PoseCOM",
                },
                {
                    "sensor_type": "PoseSensor",
                    "socket": "SonarSocket",
                    "sensor_name": "PoseSonar",
                }
            ],
            "control_scheme": 0,
            "location": [4.15, -5.485, -6.140],
            "rotation": [0.0, 0.0, 180.0]
        },
        {
            "agent_name": "sphere0",
            "agent_type": "HoveringAUV",
            "sensors": generate_arc_offset(),
            "control_scheme": 0,
            "location": [1.529, -5.265, -5.950],
            "rotation": [0.0, 0.0, 360]
        },
    ],
    "window_width": 1280,
    "window_height": 720,
}

# keyboard movement
pressed_keys = set()
force = 100

def on_press(key):
    global pressed_keys
    if hasattr(key, 'char') and key.char is not None:
        pressed_keys.add(key.char)

def on_release(key):
    global pressed_keys
    if hasattr(key, 'char') and key.char is not None:
        pressed_keys.discard(key.char)

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

def parse_keys(keys, val):
    command = np.zeros(8)
    if 'i' in keys:
        command[0:4] += val
    if 'k' in keys:
        command[0:4] -= val
    if 'j' in keys:
        command[[4,7]] += val
        command[[5,6]] -= val
    if 'l' in keys:
        command[[4,7]] -= val
        command[[5,6]] += val

    if 'w' in keys:
        command[4:8] += val
    if 's' in keys:
        command[4:8] -= val
    if 'a' in keys:
        command[[4,6]] += val
        command[[5,7]] -= val
    if 'd' in keys:
        command[[4,6]] -= val
        command[[5,7]] += val

    return command
   
# helper functions
def make_pose6_from_T(T):
    pose6 = np.zeros(6, dtype=float)
    pose6[:3] = T[:3, 3] # xyz
    R = T[:3, :3]
    yaw = math.degrees(math.atan2(R[1,0], R[0,0]))
    pose6[3] = 0.0
    pose6[4] = 0.0
    pose6[5] = yaw
    return pose6

def warmup_ticks(env, n=5):
    state = None
    for _ in range(n):
        state = env.tick()
    return state

def yaw_from_T(T):
    T = np.asarray(T, dtype=float).reshape(4, 4)
    Rm = T[:3, :3]
    _,_,yaw = Rot.from_matrix(Rm).as_euler("zyx", degrees=False)
    return float(yaw)

# def _T_from_pos_rpy_deg(pos, rpy_deg):
#     pos = np.asarray(pos, dtype=float).reshape(3)
#     roll, pitch, yaw = [float(a) for a in rpy_deg]  # degrees
#     Rm = Rot.from_euler("XYZ", [roll, pitch, yaw], degrees=True).as_matrix()
#     T = np.eye(4, dtype=float)
#     T[:3, :3] = Rm
#     T[:3, 3] = pos
#     return T

# def _xyz_rpy_rad_from_T(T):
#     T = np.asarray(T, dtype=float).reshape(4, 4)
#     x, y, z = [float(v) for v in T[:3, 3]]
#     roll, pitch, yaw = Rot.from_matrix(T[:3, :3]).as_euler("XYZ", degrees=False)
#     return x, y, z, float(roll), float(pitch), float(yaw)

# def build_cam_specs_from_generate_arc_offset():
#     cams = []
#     for s in generate_arc_offset():
#         name = s.get("sensor_name", "")
#         if name.startswith("DepthArcCam"):  # only the 8 depth cams
#             idx = int(name.replace("DepthArcCam", ""))
#             cams.append((idx, s["location"], s["rotation"]))  # loc + rpy_deg
#     cams.sort(key=lambda t: t[0])
#     return cams

# def save_config_poses_one_orient(orientations, orient_idx, base_path, cam_specs,
#                                  out_dirname="5m_deep_poses_config", skip_existing=True):
#     orient = orientations[orient_idx - 1]  # orient_idx is 1..8
#     pose_dir = os.path.join(base_path, out_dirname, f"orient_{orient_idx}")
#     os.makedirs(pose_dir, exist_ok=True)

#     T_world_rig = _T_from_pos_rpy_deg(orient["sphere_pos"], orient["sphere_rot"])

#     for cam_idx, cam_loc, cam_rpy_deg in cam_specs:
#         csv_filename = os.path.join(pose_dir, f"pose_{cam_idx}.csv")
#         if skip_existing and os.path.exists(csv_filename):
#             continue

#         T_rig_cam = _T_from_pos_rpy_deg(cam_loc, cam_rpy_deg)
#         T_world_cam = T_world_rig @ T_rig_cam

#         x, y, z, roll, pitch, yaw = _xyz_rpy_rad_from_T(T_world_cam)

#         with open(csv_filename, mode="w", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerows([[x], [y], [z], [roll], [pitch], [yaw]])



# def wrap_to_pi(a):
#     # wraps angle(s) to [-pi, pi]
#     return (a + np.pi) % (2*np.pi) - np.pi

# def debug_arc_alignment(state, radius=3, start_angle=-67.5, angle_increment=22.5):
#     sphere_state = state["sphere0"]

#     # World pose of the rig (sphere0) from PoseSensor you already add
#     T_world_rig = np.asarray(sphere_state["PoseHoveringAUV"], dtype=float).reshape(4, 4)

#     print("\n=== ARC SENSOR ALIGNMENT DEBUG ===")
#     for i in range(1, 9):
#         angle_deg = start_angle + (i - 1) * angle_increment
#         angle_rad = math.radians(angle_deg)

#         rel_loc = np.array([radius * math.cos(angle_rad), radius * math.sin(angle_rad), 0.0], dtype=float)
#         rel_rpy = np.array([0.0, 34.0, angle_deg + 180.0], dtype=float)

#         # Expected world pose if the sensor mount is applied correctly
#         T_rig_cam = _T_from_pos_rpy_deg(rel_loc, rel_rpy)
#         T_world_cam = T_world_rig @ T_rig_cam
#         #exp_xyz, exp_rpy = _xyz_rpy_deg_from_T(T_world_cam)

#         loc_key = f"LocArcCam{i}"
#         rot_key = f"RotArcCam{i}"

#         exp_xyz, exp_rpy_deg = _xyz_rpy_deg_from_T(T_world_cam)

#         # measured (degrees)
#         meas_xyz = np.asarray(sphere_state[loc_key], dtype=float).reshape(3)
#         meas_rpy_deg = np.asarray(sphere_state[rot_key], dtype=float).reshape(3)

#         # convert to radians for comparison
#         exp_rpy = np.deg2rad(exp_rpy_deg)
#         meas_rpy = np.deg2rad(meas_rpy_deg)

#         d_xyz = meas_xyz - exp_xyz
#         d_rpy = wrap_to_pi(meas_rpy - exp_rpy)  # radians, wrapped

#         print(
#             f"Cam{i}: "
#             f"exp_xyz={exp_xyz} meas_xyz={meas_xyz} d_xyz={d_xyz} | "
#             f"exp_rpy_rad={exp_rpy} meas_rpy_rad={meas_rpy} d_rpy_rad={d_rpy}"
#         )

def save_depth_data(state, base_path, orient_idx,scenario_name):
    
    # Setup Paths
    dir_name = "objects_depth_fixed8"
    cluster_dir = os.path.join(base_path)
    
    os.makedirs(cluster_dir, exist_ok=True)

    mat_data = {} 
    sphere_state = state["sphere0"]
    
    # debug
    pose_keys = [k for k in sphere_state.keys() if "Pose" in k]
    print("POSE KEYS:", pose_keys)

    saved_count = 0
    
    for sensor_name, sensor_data in sphere_state.items():
        if "DepthArcCam" in sensor_name:
            # Extract depth buffer
            depth_array = sensor_data["depth"] 
            
            # incase of invalid values
            depth_array = depth_max_range(depth_array)
            seam_rows, _ = find_seam_rows(depth_array, k=5.0)
            if seam_rows:
                #depth_array = invalidate_seam_rows(depth_array, seam_rows, band=1)
                depth_array = median_blur_seam(depth_array, seam_rows, margin=1)
            
            # store in mat data
            mat_data[sensor_name] = depth_array

            saved_count += 1
    
    # save .mat files
    Z = np.stack([mat_data[f"DepthArcCam{i}"] for i in range(1,9)], axis=0)
    # (8,256,256) Z column
    mat_path = os.path.join(cluster_dir, f"{scenario_name}_depth_orient{orient_idx}.mat")
    sio.savemat(mat_path, {"Z": Z})
    
    # save pose as .csv 
    pose_dir = os.path.join(cluster_dir,"5m_deep_poses", f"orient_{orient_idx}")
    os.makedirs(pose_dir, exist_ok=True)
    
    # for i in range(8):
    #     loc_sensor_name = f"LocArcCam{i+1}"
    #     rot_sensor_name = f"RotArcCam{i+1}"

    #     if loc_sensor_name not in sphere_state or rot_sensor_name not in sphere_state:
    #             raise ValueError(f"Missing sensor data for {loc_sensor_name} or {rot_sensor_name} in sphere_state keys: {list(sphere_state.keys())}")

    #     x, y, z = np.asarray(sphere_state[loc_sensor_name], dtype=float)    
    #     roll_deg, pitch_deg, yaw_deg = np.asarray(sphere_state[rot_sensor_name], dtype=float)
    #     roll, pitch, yaw = np.deg2rad([roll_deg, pitch_deg, yaw_deg])
        
    #     # csv_filename = os.path.join(pose_dir, f"pose_{i+1}.csv")
    #     # with open(csv_filename, mode="w", newline="") as f:
    #     #     writer = csv.writer(f)
    #     #     # 1 column 6 rows
    #     #     writer.writerows([[x], [y], [z], [roll], [pitch], [yaw]])
            
    print(f"\t[Orient {orient_idx}] Saved {saved_count} images in .mat files and pose in .csv files")
    
    
    
def find_seam_rows(depth, k=5.0):
    
    depth = np.asarray(depth, np.float32)
    # vertical difference between adjacent rows for entire image
    vdiff = np.abs(depth[1:, :] - depth[:-1, :])  # (H-1, W)
    # median jump size of each row boundary for all columns
    score = np.nanmedian(vdiff, axis=1)                  

    # something relative to compare to and is the median jump size of entire image (single number)
    base = np.nanmedian(score) + 1e-6
    seams = np.where(score > k * base)[0] + 1            # +1 converts boundary index -> seam_row
    return seams.tolist(), score

def invalidate_seam_rows(depth, seam_rows, band=1):
    depth = np.asarray(depth, np.float32).copy()
    H, _ = depth.shape
    for r in seam_rows:
        r0 = max(r - band, 0)
        r1 = min(r + band, H - 1)
        depth[r0:r1+1, :] = np.nan
    return depth

def median_blur_seam(depth, seam_rows, margin=3):
    H, _ = depth.shape
    
    # ROI bounds after clampiing
    rmin = max(min(seam_rows) - margin, 0)
    rmax = min(max(seam_rows) + margin, H - 1)

    roi = depth[rmin:rmax+1, :].astype(np.float32).copy()
    roi = cv2.medianBlur(roi, 3)
    depth[rmin:rmax+1, :] = roi
    return depth
    
import json

def load_bookmarks(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # supports {"bookmarks":[...]} or a raw list
    return data["bookmarks"] if isinstance(data, dict) and "bookmarks" in data else data


OBJECT_POS = np.array(config['agents'][1]['location']) 

# for post processing of depth images
DEPTH_MIN = 0.5
DEPTH_MAX = 8.0

def depth_max_range(depth):
    d = np.asarray(depth, dtype=np.float32)
    mask = (~np.isfinite(d)) | (d < DEPTH_MIN) | (d > DEPTH_MAX)
    d = np.clip(d, DEPTH_MIN, DEPTH_MAX)
    # map [DEPTH_MIN, DEPTH_MAX] -> [0,255] for display
    #disp = ((d - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN + 1e-8) * 255.0).astype(np.float32)
    d[mask] = 0  # set out of range pixels to black
    return d


def main():
    
    main_path = r"C:\Users\ramah\holooceanv2.0.1\engine\data\single_object_scenarios\cluster\objects_depth_fixed8"
    
    with holoocean.make(scenario_cfg=config, verbose=True, show_viewport=True) as env:
        env.set_render_quality(0)
        auv = env.agents["auv0"]
        sphere = env.agents["sphere0"]

        # State variables
        orientations = []
        curr_orient  = 0
        mode = "TELEOP"
        tick = 0
        # choose which bookmarked pose to use
        bm_path = os.path.join(os.path.dirname(__file__), "pose_bookmarks_5m.json")
        print("Bookmark path:", bm_path)
        
        print("Ready. Controls: WASD/IJKL. Press 'u' to capture depth images.")

        while True: 
            if 'q' in pressed_keys:
                break
            tick += 1
            
            if mode == "TELEOP":
                action = parse_keys(pressed_keys, force) 
                auv.act(action)
                
            state = env.tick()
            #auv_state = state["auv0"] if "auv0" in state else state
            
            if 'u' in pressed_keys:
                print("DEPTH CAPTURE MODE INITIATED")
                pressed_keys.discard('u')

                bookmarks = load_bookmarks(bm_path)

                # use first 8 bookmarks as the 8 orientations
                orientations = []
                for i in range(8):
                    bm = bookmarks[i]

                    orientations.append({
                        "rov_pos": list(bm["rov"]["pos"]),
                        "rov_rot": list(bm["rov"]["rpy"]),      # exact rpy from file
                        "sphere_pos": list(bm["rig"]["pos"]),
                        "sphere_rot": list(bm["rig"]["rpy"]),   # exact rpy from file
                    })

                curr_orient = 0
                mode = "INIT_ORIENT"
                #cam_specs = build_cam_specs_from_generate_arc_offset()

            # CAPTURE LOOP
            elif mode == "INIT_ORIENT":
                print(f"\n--> Capturing Orientation {curr_orient+1}/8")
                
                # teleport Both Agents
                data = orientations[curr_orient]
                
                auv.teleport(data["rov_pos"], data["rov_rot"])
                sphere.teleport(data["sphere_pos"], data["sphere_rot"])
                
                state = warmup_ticks(env, n=5)
                #keys = sorted(state["sphere0"].keys())
                #print([k for k in keys if ("DepthArcCam" in k) or ("LocArcCam" in k) or ("RotArcCam" in k) or ("Pose" in k)])
                #debug_arc_alignment(state)
                #save_config_poses_one_orient(orientations, curr_orient+1, main_path, cam_specs)

                
                # # ================= DEBUG CAMERA VIEW =================
                #state = warmup_ticks(env, n=5)
                # sphere_state = state["sphere0"]
                # current_cam_idx = 0
                # while current_cam_idx < 8:
                #     depth_name = f"DepthArcCam{current_cam_idx+1}"
                # #     if depth_name in sphere_state:
                #     depth_payload = sphere_state[depth_name]
                #     depth_img = depth_payload["depth"]
                        
                        
                #     depth_img = np.asarray(depth_img, dtype=np.float32)
                #     depth_img = depth_max_range(depth_img)
                #     seam_rows, _ = find_seam_rows(depth_img, k=5.0)
                #     if seam_rows:
                #         depth_img = median_blur_seam(depth_img, seam_rows, margin=3)
                        
                #         cv2.imshow(f"{depth_name} (depth)  [press any key → next]", depth_img/10)
                #     else:
                #         print(f"WARNING: {depth_name} not found in sphere_state keys: {list(sphere_state.keys())}")
                #     key = cv2.waitKey(0) & 0xFF
                #     cv2.destroyAllWindows()
                #     if key == ord('q'):
                #         print("DEBUG VIEW aborted by user (q).")
                #         break
                #     current_cam_idx += 1
                
                # capture & save Depth Data
                object = "small_crate"
                if state is not None and "sphere0" in state:
                    save_depth_data(state, main_path, curr_orient+1, f"{object}_only_5m_deep")
                else:
                    print(f"Error: No state for orientation {curr_orient+1}")
                    
                # next orient
                curr_orient += 1
                if curr_orient >= len(orientations):
                    print("ALL CAPTURES DONE. Returning to TELEOP.")
                    mode = "TELEOP"
                    auv.set_control_scheme(ControlSchemes.AUV_THRUSTERS)
                else:
                    # Stay in INIT_ORIENT for the next angle
                    mode = "INIT_ORIENT"
                                 
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()