import io
import holoocean
import numpy as np
import math
import cv2 
import matplotlib.pyplot as plt
from pynput import keyboard
from holoocean.agents import ControlSchemes
import os
from PIL import Image
import json

from linear_sweep_orientations8_6m import save_npy_stack


def generate_arc_offset(
    sensor_type="DepthCamera",
    base_name="DepthArcCam",
    socket="CameraSocket",
    radius=1.4,
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
                "rotation": [0.0, 0.0, angle_deg + 180.0],
                "sensor_name": f"{base_name}{i}",
                "Hz": 0.1,
                "configuration": {
                    "CaptureWidth": 256,
                    "CaptureHeight": 256,
                },
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
                },
                {
                    "sensor_type": "ImagingSonar",
                    "socket": "SonarSocket",
                    "rotation": [0.0, 34, 0.0],
                    "Hz": 12,
                    "configuration": {
                        "RangeBins": 512,
                        "AzimuthBins": 512,
                        "RangeMin": 0.1,
                        "RangeMax": 8,
                        "InitOctreeRange": 30,
                        "Elevation": 12,
                        "Azimuth": 60,
                        "ElevationBins": 185,
                        "AzimuthStreaks": -1,
                        #"ShadowEpsilon":0.064,
                        "ScaleNoise": True,
                        "AddSigma": 0.04, 
                        "MultSigma": 0.04,
                        "RangeSigma": 0.01,
                        "MultiPath": True,
                        "ViewOctree": -1,
                        "ViewRegion": False,
                    },
                },
            ],
            "control_scheme": 0,
            "location": [5.14, -5.483, -5.140],
            "rotation": [0.0, 0.0, 180.0],
        },
        # {
        #     # Static rig of 8 depth cameras looking at the scene
        #     "agent_name": "sphere0",
        #     "agent_type": "HoveringAUV",
        #     "sensors": generate_arc_offset(),
        #     #+
        #                #generate_arc_offset(sensor_type="RGBCamera", base_name="RGBArcCam"),
        #     "control_scheme": 0,
        #     "location": [2.7, -5.65, -6.516],
        #     "rotation": [0.0, 0.0, 360.0],
        # },
    ],
    "window_width": 1280,
    "window_height": 720,
}


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

listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release)
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

def save_jpg(fig, ax, plot, frame, out_size, out_img):
    frame01 = np.clip(frame, 0.0, 1.0)
    plot.set_array(frame01.ravel())

    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    fig.patch.set_alpha(1.0)
    ax.patch.set_alpha(1.0)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    rgba = np.asarray(fig.canvas.buffer_rgba())
    H, W = rgba.shape[:2]

    bbox = ax.get_tightbbox(renderer)
    x0, y0, x1, y1 = bbox.extents

    x0i = max(int(np.floor(x0)), 0)
    x1i = min(int(np.ceil(x1)), W)
    y0i = max(int(np.floor(H - y1)), 0)
    y1i = min(int(np.ceil(H - y0)), H)

    rgba = rgba[y0i:y1i, x0i:x1i, :]
    img8 = rgba[..., 0].copy()  # grayscale copy from canvas

    if out_size is not None:
        img_pil = Image.fromarray(img8, mode="L")
        img_pil = img_pil.resize(out_size, Image.Resampling.LANCZOS)
        img8 = np.array(img_pil, dtype=np.uint8)

    os.makedirs(os.path.dirname(out_img), exist_ok=True)
    Image.fromarray(img8, mode="L").save(out_img)

    return img8  # (H,W) uint8

def save_npy_stack(frames_uint8, out_npy, out_size):
    
    if len(frames_uint8) == 0:
        raise ValueError("No frames to save")

    resized = []
    for f in frames_uint8:
        img_pil = Image.fromarray(f, mode="L")
        img_pil = img_pil.resize(out_size, Image.Resampling.LANCZOS)
        resized.append(np.array(img_pil, dtype=np.uint8))
        
    arr = np.stack([np.repeat(f[..., None], 3, axis=2) for f in resized], axis=0)
    os.makedirs(os.path.dirname(out_npy), exist_ok=True)
    np.save(out_npy, arr)
    return arr
        
def load_bookmarks(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # supports {"bookmarks":[...]} or a raw list
    return data["bookmarks"] if isinstance(data, dict) and "bookmarks" in data else data
    
    
# sonar sweep params

SONAR_DIST = 3.6 # meters
EPS = 0.4  # threshold for waypoint arrival
CAPTURE_STEP = 0.10 # meters per step
WAYPOINT_DIST = 0.8

#OBJECT_POS = np.array(config['agents'][1]['location'])  # position of target object
OBJECT_POS = np.array([1.529, -5.265, -5.950])  # center of circle

GAIN = 2.5

# 2 phase state machine

def main():
    
    bm_path = os.path.join(os.path.dirname(__file__), "pose_bookmarks_5m.json")

    main_path = r"C:\Users\ramah\holooceanv2.0.1\engine\data\single_object_scenarios"
    
    config2 = config['agents'][0]['sensors'][-1]["configuration"]
    azi = config2['Azimuth']
    minR = config2['RangeMin']
    maxR = config2['RangeMax']
    binsR = config2['RangeBins']
    binsA = config2['AzimuthBins']

    
    #### GET PLOT READY
    plt.ion()
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(8,5))
    ax.set_theta_zero_location("N")
    ax.set_thetamin(-azi/2)
    ax.set_thetamax(azi/2)

    theta = np.linspace(-azi/2, azi/2, binsA)*np.pi/180
    r = np.linspace(minR, maxR, binsR)
    T, R = np.meshgrid(theta, r)
    z = np.zeros_like(T)

    plt.grid(False)
    plot = ax.pcolormesh(T, R, z, cmap='gray', shading='auto', vmin=0, vmax=1)
    plt.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()
   

    with holoocean.make(scenario_cfg=config, verbose=True, show_viewport=True) as env:
        env.set_render_quality(0)
        auv = env.agents["auv0"]
        #sphere = env.agents["sphere0"]
        
        # variables
        orientations = []
        curr_orient  = 0
        mode = "TELEOP"
        tick = 0
        sweep_waypoints = [] # list of waypoints for sweep
        object_pose = None
        #sphere_pos = OBJECT_POS.copy()
        
        last_save_pos = None
        images_captured = 0
        latest_sonar_frame = None
        sonar_imgs = []
        
        while True: 
            tick += 1
            # phase 1 teleoperation mode
            if mode == "TELEOP":
                action = parse_keys(pressed_keys, force) 
                auv.act(action)
                
            state = env.tick()
            auv_state = state["auv0"] if "auv0" in state else state
            #pc = np.array(state["auv0"]["PoseCOM"])
            #print("Shape of PoseCOM:", pc.shape)
            
            if 'ImagingSonar' in auv_state:
                latest_sonar_frame = auv_state['ImagingSonar'].copy()
                    
            # sweep mode switch 
            if 'u' in pressed_keys:
                print("SWEEP MODE INITIATED")
                pressed_keys.discard('u') 
                bm = load_bookmarks(bm_path)
                
                if len(bm) < 8:
                    raise ValueError("Need at least 8 bookmarks for the orientations.")
                                      
                orientations = []
                for i in range(8):
                    bookmarks = bm[i]
                    orientations.append({
                        "rov_pos": list(bookmarks["rov"]["pos"]),
                        "rov_rot": list(bookmarks["rov"]["rpy"]),
                        "real_index": i
                    })
                    
                curr_orient = 0
                mode="INIT_ORIENT"
                
            elif mode == "INIT_ORIENT":
                real_index = orientations[curr_orient]["real_index"]
                print(f"\n--> Starting Orientation {real_index+1}/8")
                # teleport ROV  to the new orientation pose
                data = orientations[curr_orient]

                auv.teleport(data['rov_pos'], data['rov_rot'])

                # # rotate the rig as well 
                # sphere = env.agents["sphere0"]
                # sphere.teleport(sphere_pos, data['sphere_rot']) 

                # allow for stabilization
                state = warmup_ticks(env, n=5)
                auv_state = state["auv0"] if "auv0" in state else state
                
        # ================= DEBUG CAMERA VIEW =================
                # if curr_orient > 0:
                #     print(">>> DEBUG: Displaying Camera Views. Press SPACE to continue sweep...")
                #     sphere_state = state["sphere0"]
                #     debug_windows = []
            
                
                    
                # # Loop through all sensors found in the state
                #     for key in sphere_state:
                #         if "DepthArcCam" in key:
                #             # Normalize depth for visualization (0-255)
                #             depth_img = sphere_state[key]["depth"]
                #             # Adjust divisor based on your range (e.g., /10.0 or /20.0 meters)
                #             cv2.imshow(f"DEBUG: {key}", depth_img / 10.0)
                        
                #         if "RGBArcCam" in key:
                #             rgb_img = sphere_state[key]
                #             cv2.imshow(f"DEBUG: {key}", rgb_img[:,:,0:3])
                    
                #     cv2.waitKey(0)  # Wait for a key press to proceed
                #     for win in debug_windows:
                #         try: cv2.namedWindow(win)
                #         except: pass
                        
                # recompute object pose from the new teleported pose
                Tcom = np.array(auv_state["PoseCOM"], dtype=float)
                object_pose = make_pose6_from_T(Tcom)
                yaw_object = float(object_pose[5])
                object_pos = object_pose[:3].copy()
                
                last_save_pos = object_pos.copy()
                images_captured = 0
                sonar_imgs.clear()

                # recompute u for this orientation
                u = object_pos - OBJECT_POS
                u[2] = 0.0
                n = np.linalg.norm(u[:2])
                if n < 1e-6:
                    u = np.array([math.cos(math.radians(yaw_object)),
                                math.sin(math.radians(yaw_object)), 0.0], dtype=float)
                else:
                    u = u / n

                # switch to pose PD control and build waypoints for linear sweep
                auv.set_control_scheme(ControlSchemes.AUV_CONTROL)

                sweep_waypoints = []
                sdist = 0.0
                while sdist <= SONAR_DIST + 1e-9:
                    g = object_pose.copy()
                    g[0] = object_pose[0] + sdist * u[0]
                    g[1] = object_pose[1] + sdist * u[1]
                    g[2] = object_pose[2]
                    g[3] = 0.0
                    g[4] = 0.0
                    g[5] = yaw_object
                    sweep_waypoints.append(g)
                    sdist += WAYPOINT_DIST

                wp_idx = 0
                mode = "SWEEP"
            
            elif mode == "SWEEP":
                goal = sweep_waypoints[wp_idx]
                auv.act(goal)
                Tcom = np.array(auv_state["PoseCOM"], dtype=float)
                cur = Tcom[:3, 3]
                goal_pos = goal[:3]
                
                if np.linalg.norm(cur - goal_pos) < EPS:
                    wp_idx += 1
                    #print(f"\t[Orientation {curr_orient+1}] Reached Waypoint {wp_idx}/{len(sweep_waypoints)}")
                dist_since_save = np.linalg.norm(cur - last_save_pos)
               
                                 
                if dist_since_save >= CAPTURE_STEP and len(sonar_imgs) < 24:
                    # Only save if we have actual sonar data ready
                    if latest_sonar_frame is not None:
                        MAX_FRAMES = 24
                        images_captured += 1
                        number = MAX_FRAMES -len(sonar_imgs)
                        #number = f"{images_captured}"
                        #dir_name = f"orient_{curr_orient+1}"
                        real_idx = orientations[curr_orient]["real_index"]
                        dir_name = f"orient_{real_idx+1}"
                        object = "small_crate"
                        sonar_path_jpg = os.path.join(main_path,f"{object}_only","5m_deep","sonar", dir_name,f"FLSc_{number}.jpg")
                        #sonar_path_npy = os.path.join(main_path,"cluster","objects_inputRGB", dir_name,f"FLSc_{number}.npy")
    
                        
                        os.makedirs(os.path.dirname(sonar_path_jpg), exist_ok=True)
                        #os.makedirs(os.path.dirname(sonar_path_npy), exist_ok=True)

                        
                        latest_sonar_frame = latest_sonar_frame.astype(np.float32)
                        latest_sonar_frame_gain = latest_sonar_frame * GAIN

                        img = save_jpg(fig, ax, plot, latest_sonar_frame_gain, (1500,1500), sonar_path_jpg)
                        sonar_imgs.append(img)
                        
                        print(f"\t[Orientation {real_index+1}], Saved Img {images_captured} (Step: {dist_since_save:.3f}m)")
                        
                        # Reset tracking position
                        last_save_pos = cur.copy()
                        
                if wp_idx >= len(sweep_waypoints):
                        sonar_path_npy = os.path.join(main_path, "cluster", "objects_inputRGB", f"{object}_only_5m_deep_orient{real_index+1}.npy")
                        save_npy_stack(sonar_imgs[:24], sonar_path_npy, (128,128))
                        curr_orient += 1
                        if curr_orient >= len(orientations):
                            mode = "TELEOP"
                            auv.set_control_scheme(ControlSchemes.AUV_THRUSTERS)
                            print("ALL ORIENTATIONS DONE")
                        else:
                            # go setup next orientation
                            mode = "INIT_ORIENT"
                            real_idx = orientations[curr_orient]["real_index"]
                            print(f"NEXT ORIENTATION: {real_idx+1}/8")

            if 'ImagingSonar' in auv_state:
                s = auv_state['ImagingSonar']
                s_gain = np.clip(s * GAIN, 0.0, 1.0)
                plot.set_array(s_gain.ravel())

                fig.canvas.draw()
                fig.canvas.flush_events()
                    
            if 'q' in pressed_keys:
                break
                    
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
