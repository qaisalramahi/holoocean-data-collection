import re
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import traceback
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import ply
import reconstruct_data_1 as reconstruct
import argparse
from numpy import save
from numpy import asarray
import cv2
import math
import pandas as pd
import open3d as o3d
import tempfile
from pathlib import Path
import subprocess
import sys
import textwrap
import scipy.io as sio

class ImagePointCloudViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("SonarCloudViz")
        
        # Tracks whether the Open3D window is currently open
        self._o3d_window_open = False
        self.current_pcd = None
        # --- NEW: a fixed controls bar at the very top ---
        self.controls = tk.Frame(root)
        self.controls.pack(side=tk.TOP, fill=tk.X)

        # Load folder button stays visible here
        self.load_button = ttk.Button(self.controls, text="Open Folder…", command=self.load_folder)
        self.load_button.pack(side=tk.LEFT, padx=8, pady=6)

        # Optional: a 'Clear' button to wipe the view
        self.clear_button = ttk.Button(self.controls, text="Clear", command=self.clear_views)
        self.clear_button.pack(side=tk.LEFT, padx=4, pady=6)

        # after self.clear_button...
        self.o3d_button = ttk.Button(self.controls, text="Open 3D Viewer", command=self.open_o3d_viewer)
        self.o3d_button.pack(side=tk.LEFT, padx=4, pady=6)

        # keyboard shortcut
        self.root.bind("<Control-v>", lambda e: self.open_o3d_viewer())        
        
        # --- Menu + keyboard shortcut (Ctrl+O) ---
        menubar = tk.Menu(root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open…", command=self.load_folder, accelerator="Ctrl+O")
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=root.quit)
        menubar.add_cascade(label="File", menu=filemenu)
        self.root.config(menu=menubar)
        self.root.bind("<Control-o>", lambda e: self.load_folder())

        # Main content area (everything else lives below the controls bar)
        self.left_frame = tk.Frame(root)
        self.left_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.image_frame = tk.Frame(self.left_frame)
        self.image_frame.pack(side="top", fill="both", expand=True)

        self.depth_frame = tk.Frame(self.left_frame)
        self.depth_frame.pack(side="top", fill="both", expand=True)
        
        self.image_folder = None
        self.object_name = None
        self.depth_level = None
        
        # IMPORTANT: pointcloud frame inside left_frame (not root)
        self.pointcloud_frame = tk.Frame(self.left_frame, width=600, height=600)
        self.pointcloud_frame.pack(side=tk.TOP, pady=10, anchor='center')
        self.pointcloud_frame.pack_propagate(False)

        self.fig = plt.Figure(figsize=(3, 3))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas_plot = None
        
    
    def dataset_root(self) -> Path:
            if not getattr(self, "image_folder", None):
                raise RuntimeError("No dataset folder selected yet.")
            return Path(self.image_folder)

    def depth_dir(self) -> Path:
        return self.dataset_root() / "cluster" / "objects_depth_fixed8"

    # please use your own RGB directory
    def rgb_dir(self) -> Path:
        # Resolve RGB from the current working directory
        rgb = Path.cwd() / "rgb"
        if not rgb.is_dir():
            raise FileNotFoundError(f"RGB directory not found: {rgb}")
        return rgb

    def pose_dir(self) -> Path:
        #return self.dataset_root() / "cluster" / "objects_depth_fixed8"
        base = Path.cwd()
        expected = base / f"{self.depth_level}_poses"
        if not expected.is_dir():
            raise FileNotFoundError(f"Expected poses folder not found: {expected}")
        return base
    
    def clear_views(self):
        # Don’t clear while a native O3D window exists (avoids X11 BadWindow)
        if getattr(self, "_o3d_window_open", False):
            print("Please close the 3D viewer window first.")
            return

        for w in self.image_frame.winfo_children():
            w.destroy()
        for w in self.depth_frame.winfo_children():
            w.destroy()

        # Remove any message/toolbar inside the pointcloud frame
        for w in self.pointcloud_frame.winfo_children():
            w.destroy()

        # Reset point cloud reference
        self.current_pcd = None

        # Let Tk drain pending X events before rebuilding
        #self.root.update_idletasks()
        self.root.update()
        
    def _save_current_pcd_to_temp(self):
        if not hasattr(self, "current_pcd") or self.current_pcd is None:
            return None
        tmpdir = tempfile.mkdtemp(prefix="sonarcloudviz_")
        ply_path = os.path.join(tmpdir, "cloud.ply")
        o3d.io.write_point_cloud(ply_path, self.current_pcd, write_ascii=False, compressed=False)
        return ply_path       
    
    def load_folder(self):
        # Don’t open a new dataset while the O3D window is up
        if getattr(self, "_o3d_window_open", False):
            print("Close the 3D viewer before opening a new folder.")
            return

        #folder_path = filedialog.askdirectory()
        folder_path = filedialog.askdirectory(parent=self.root, mustexist=True)

        if not folder_path:
            return
        
        p = Path(folder_path)
        
        m = re.fullmatch(r"orient_(\d+)", p.name, flags=re.IGNORECASE)
        if m:
            self.orient_idx = int(m.group(1))
            # p = dataset_root/sonar/orient_k  -> dataset_root = p.parents[1]
            self.image_folder = str(p.parents[3])
            self.depth_level = p.parents[1].name
            self.object_name = p.parents[2].name
        else:
            # User picked dataset root directly
            self.orient_idx = 1
            self.image_folder = str(p)

            #self.image_folder = folder_path
            
        # print("dataset_root =", self.dataset_root())
        # print("orient_idx   =", self.orient_idx)
        # print("depth_dir    =", self.depth_dir())
        print("rgb_dir      =", self.rgb_dir())
        # print("pose_dir     =", self.pose_dir())
        
        # Clear current UI first
        self.clear_views()
        #self.image_folder = folder_path
    

        # Rebuild after a short delay so X11 finishes destroying old widgets
        def _rebuild():
            try:
                self.display_all_images()
                self.display_depth_images()
               # image_path = os.path.join(folder_path, 'cluster/objects_depth_fixed8')
               # print('image_path: ',image_path)
                self.display_pointcloud()  # builds self.current_pcd
                traceback.print_exc()
            except Exception as e:
                print("Error rebuilding UI:", e)
            finally:
                self.root.update_idletasks()

        self.root.after(1000, _rebuild)  # give X11 time to fully destroy old widgets
        
    def load_Z_stack(self, mat_path: Path) -> np.ndarray:
        mat = sio.loadmat(str(mat_path), squeeze_me=True, simplify_cells=True)
        Z = mat["Z"]                          # (8,256,256)
        Z = np.asarray(Z, dtype=np.float32)
        if Z.ndim != 3 or Z.shape[0] != 8:
            raise ValueError(f"Expected (8,H,W), got {Z.shape} in {mat_path}")
        return Z
    
    def display_image_and_pointcloud(self, event):
        # Get selected image
        selected_index = self.image_listbox.curselection()
        #if selected_index:
        image_file = self.images[selected_index[0]]
    
        print('image_file: ',image_file)
        print('self.image_folder: ',self.image_folder)
        image_path = os.path.join(self.image_folder, image_file)
        pointcloud_path = self.image_folder_p 
        print('pointcloud_path',pointcloud_path)
        # Display the image
        image = Image.open(image_path)
        image.thumbnail((400, 400))  # Resize for display
        image_tk = ImageTk.PhotoImage(image)
        self.image_canvas.create_image(200, 200, image=image_tk)
        self.image_canvas.image = image_tk  # Keep reference

        # Display corresponding point cloud
        self.display_pointcloud(image_path,pointcloud_path)
    
    def open_o3d_viewer(self):
        ply_path = self._save_current_pcd_to_temp()
        if not ply_path:
            print("No point cloud available yet.")
            return
        if self._o3d_window_open:
            print("3D viewer already open.")
            return

        self._o3d_window_open = True

        # Inline Python script run in a child process that owns its own Open3D/X11 window
        viewer_code = textwrap.dedent(f"""
            import open3d as o3d
            import numpy as np
            import sys
            ply_path = r'''{ply_path}'''
            pcd = o3d.io.read_point_cloud(ply_path)
            vis = o3d.visualization.Visualizer()
            try:
                vis.create_window(window_name="SonarCloudViz 3D", width=960, height=720, visible=True)
                vis.add_geometry(pcd)
                opt = vis.get_render_option()
                opt.point_size = 2.0
                opt.background_color = np.array([1, 1, 1])
                while vis.poll_events():
                    vis.update_renderer()
            finally:
                vis.destroy_window()
        """)

        try:
            # Start child process; it will block until window closes (but OUR Tk stays responsive)
            subprocess.run([sys.executable, "-c", viewer_code], check=False)
        finally:
            self._o3d_window_open = False
            
    def display_all_images(self):
        self.image_thumbnails = []
        
        orient_idx = getattr(self, "orient_idx", 1)

        for i in range(1, 25):
            image_name = f"{self.object_name}/{self.depth_level}/sonar/orient_{orient_idx}/FLSc_{i}.jpg"
            image_path = os.path.join(self.image_folder, image_name)
            if os.path.exists(image_path):
                image = Image.open(image_path)
                #print("Before thumbnail:", image.size)
                image.thumbnail((100, 100))  # Resize thumbnails
                #print("After thumbnail:", image.size)
                image_tk = ImageTk.PhotoImage(image)

                label = tk.Label(self.image_frame, image=image_tk)
                label.image = image_tk  # Keep a reference

                row = 0 if i <= 12 else 1
                col = (i - 1) % 12
                label.grid(row=row, column=col, padx=5, pady=5)

                self.image_thumbnails.append(image_tk)
            # else:
            #     print(f"Warning: {image_name} not found.")
        
    
    def display_depth_images(self):
        self.depth_thumbnails = []
        orient_idx = int(getattr(self, "orient_idx", 1))
        object = getattr(self, "object_name", "anchor_only")
        depth_lvl = getattr(self, "depth_level", "objects_depth_fixed8")
        depth_image_path = self.depth_dir() / f"{object}_{depth_lvl}_depth_orient{orient_idx}.mat"
        
        if not depth_image_path.exists():
            raise FileNotFoundError(depth_image_path)
        
        Z = self.load_Z_stack(depth_image_path)  # (8,H,W)
        # Depth images (8)
        for i in range(1, 9):
            
            depth_image = Z[i-1]  

            # Normalize depth image to fit range [0, 255] for display
            depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)

            # Convert to 8-bit image for display
            depth_image_normalized = np.uint8(depth_image_normalized)

            # Convert the depth image to an RGB format using OpenCV (for display)
            depth_image_rgb = cv2.cvtColor(depth_image_normalized, cv2.COLOR_GRAY2RGB)

            # Create a PIL image to display on Tkinter
            depth_image_pil = Image.fromarray(depth_image_rgb)
            depth_image_pil.thumbnail((150,150))  # Resize depth images to fit
            depth_image_tk = ImageTk.PhotoImage(depth_image_pil)

            label = tk.Label(self.depth_frame, image=depth_image_tk)
            label.image = depth_image_tk

            # Placing depth images horizontally below the sonar images
            label.grid(row=0, column=i - 1 if i > 0 else 0, padx=2, pady=2)

            self.depth_thumbnails.append(depth_image_tk)
        # else:
        #     print(f"Warning: {depth_image_name} not found.")
            
    def display_pointcloud(self):
        # Example: Generate a dummy point cloud for demonstration
        
        def load_image(image_path, name, color = True):
            try:
                if (color):
                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                else:
                    image = cv2.imread(image_path,cv2.IMREAD_ANYDEPTH)
            except Exception as e:
                #raise Exception("{} image is incorrect.".format(name)) from e
                traceback.print_exc()
                raise Exception("{} image is incorrect.".format(name))
            return image, {}
        
        orient_idx = int(getattr(self, "orient_idx", 1))
        object = getattr(self, "object_name", "anchor_only")
        depth_lvl = getattr(self, "depth_level", "objects_depth_fixed8")
        Z = self.load_Z_stack(self.depth_dir() / f"{object}_{depth_lvl}_depth_orient{orient_idx}.mat")  # (8,H,W)
        
        H, W = Z.shape[1], Z.shape[2]
        hfov = math.radians(60.0)
        fx = W / (2.0 * math.tan(hfov / 2.0))
        fy = H / (2.0 * math.tan(hfov / 2.0))
        cx, cy = W / 2.0, H / 2.0
        camera_params = np.array([[fx, 0, cx],
                                [0, fy, cy],
                                [0,  0,  1]], dtype=float)
        
        pointcloud_v8=[]
        cam_origins = []
        for i in range(1, 9):
            
            z_idx = 8 - i
		    
            #depth_filename = self.depth_dir() / f"depth_{orient_idx}.mat"
            depthmap = Z[z_idx].copy()
            
            #depthmap = np.fliplr(depthmap)
            #depth_filename = f"depth_{i-1}.tiff"
            ##print(depth_filename)
            rgb_path = self.rgb_dir() / f"{i}.jpg" 
            image, image_exif = load_image(str(rgb_path), "Source")
        
            #depth_image= Image.open(depth_filename)

            
            #depthmap=np.array(depth_image)

            mask_gt= depthmap != 10
            depthmap[~mask_gt] = 0
            
            depthmap[depthmap <= 0] = 0
		
            pose_csv = self.pose_dir() / f"{self.depth_level}_poses" / f"orient_{orient_idx}" / f"pose_{i}.csv"
            vals = pd.read_csv(str(pose_csv), header=None).to_numpy(dtype=float).reshape(-1)
            x, y, z, roll, pitch, yaw = vals
            cam_origins.append([x, y, z])

            pcd_i, pts_i = reconstruct.reconstruct3d(
                image, depthmap,
                x, y, z,
                yaw,
                camera_params,
                step=1,
                roll=roll,
                pitch=pitch,
                depth_max=8
            )
            
           # Data_Pd = pd.read_csv(str(pose_csv), header=None)
           # data=np.array(Data_Pd)
            #ninety_deg=math.radians(10)
            # x=data[2]
            # y=data[1]
            # z=data[0]
            # yaw=data[5]


            # image_height = image.shape[0]
            # image_width  = image.shape[1]

            # depthmap_height, depthmap_width = depthmap.shape[0], depthmap.shape[1]


            # hfov_degrees=60
            # vfov_degrees=60
            # hFov = math.radians(hfov_degrees)
            # vFov = math.radians(vfov_degrees)
            # cx, cy = image_width/2, image_height/2
            # fx = image_width/(2*math.tan(hFov/2))
            # fy = image_height/(2*math.tan(vFov/2))
            

            # if (depthmap_width != image_width or depthmap_height != image_height):
            #     #print("Depthmap size does not match with image size. Scaling depthmap...")
            #     depthmap = cv2.resize(depthmap, (image_width, image_height))

            # camera_params = np.array([[fx,  0, cx],
            #                     [ 0, fy, cy],
            #                     [ 0,  0,  1]])

           
            #scene, transformed_points = reconstruct.reconstruct3d(image,depthmap,x,y,z,yaw,camera_params,step=1,mesh='store_true')
            
            pointcloud_v8.append(pts_i)
            

            #pointcloud_file_path = f"pointclouds/{obj_data}_{terrain_name}_{terrain_moved_name}_{orientation}.ply" 
            #print(pointcloud_file_path)

            ####scene.save(pointcloud_file_path,color=args.color)
            
        print("Done")

        #pointcloud_b_array=asarray(pointcloud_v8)
        #combined_pointcloud = pointcloud_b_array.reshape(-1, 3)  # Shape: (8*16384, 3)
        combined_pointcloud = np.vstack(pointcloud_v8)
        # Around Z-axis
        #rotation_matrix = np.array([[0, -1, 0], 
        #                        [1, 0, 0], 
        #                        [0, 0, 1]])

        # Apply the rotation to each point in the point cloud
        
        theta = np.radians(-90)  # Convert 90 degrees to radians
        # around Y-axis
        #rotation_matrix = np.array([
        #    [np.cos(theta), 0, np.sin(theta)],
        #    [0, 1, 0],
        #    [-np.sin(theta), 0, np.cos(theta)]
        #])
        
        # around X-axis
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])
        
        #rotated_pointcloud = combined_pointcloud
        rotated_pointcloud = np.dot(combined_pointcloud, rotation_matrix.T)
        
        cam_origins = np.asarray(cam_origins, dtype=float)
        cam_origins_rot = np.dot(cam_origins, rotation_matrix.T)
        
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(rotated_pointcloud)
        pcd_filtered, ind = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
        
        points = np.asarray(pcd_filtered.points)
        print('points shape: ',points.shape)

       
        if points.size == 0:
            print("No points after filtering.")
            self.current_pcd = None
            # Show a message in the placeholder frame
            for w in self.pointcloud_frame.winfo_children():
                w.destroy()
            ttk.Label(self.pointcloud_frame, text="No points to display.").pack(pady=8)
            return

    #    # Color by Z (viridis-like via matplotlib colormap)
    #     z = points[:, 2]
    #     rng = z.max() - z.min()
    #     if rng > 0:
    #         z_norm = (z - z.min()) / rng
    #         import matplotlib.cm as cm  # ok to use only for colormap
    #         colors = cm.get_cmap('viridis')(z_norm)[:, :3]  # Nx3 in [0,1]
    #     else:
    #         colors = np.ones((points.shape[0], 3)) * 0.7
        
        # #Color by distance-to-camera (nearest camera origin)
        import matplotlib.cm as cm

        # distance to nearest camera origin
        d = np.linalg.norm(points[:, None, :] - cam_origins_rot[None, :, :], axis=2).min(axis=1)

        # robust normalization (avoids a few far points crushing the contrast)
        lo, hi = np.percentile(d, [2, 98])
        d_clip = np.clip(d, lo, hi)
        den = (hi - lo) if (hi - lo) > 1e-12 else 1.0
        d_norm = (d_clip - lo) / den  # 0..1

        # viridis colormap
        colors = cm.get_cmap("viridis_r")(d_norm)[:, :3]
        pcd_filtered.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
                    

        #pcd_filtered.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

        # Keep for the external viewer
        self.current_pcd = pcd_filtered

        # Show a simple message inside the Tk frame
        for w in self.pointcloud_frame.winfo_children():
            w.destroy()
        ttk.Label(
            self.pointcloud_frame,
            text="Point cloud ready. Click 'Open 3D Viewer' (or press Ctrl+V)."
        ).pack(pady=8)
# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = ImagePointCloudViewer(root)
    root.mainloop()
