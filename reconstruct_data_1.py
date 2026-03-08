# reconstruct_data.py
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as Rot
# :)
def _as_float(x):
    # just in case 
    a = np.asarray(x, dtype=float).reshape(-1)
    return float(a[0])

def reconstruct3d(
    image,
    depthmap,
    x, y, z,
    yaw,
    camera_params,
    step=1,
    #mesh=False,
    #*,
    roll=0.0,
    pitch=0.0,
    depth_min=1e-6,
    depth_max=None,
):

    depth = np.asarray(depthmap, dtype=np.float32)
    H, W = depth.shape[:2]

    # camera matrix
    K = np.asarray(camera_params, dtype=float).reshape(3, 3)
    # camera intrinsics
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    x = _as_float(x); y = _as_float(y); z = _as_float(z)
    roll = _as_float(roll); pitch = _as_float(pitch); yaw = _as_float(yaw)

    # valid mask
    valid = np.isfinite(depth) & (depth > depth_min)
    if depth_max is not None:
        valid &= (depth < float(depth_max))

    # pixel grid (subsample by step)
    us = np.arange(0, W, step, dtype=np.float32)
    vs = np.arange(0, H, step, dtype=np.float32)
    # horizontal and vertical pixel coordinates
    u, v = np.meshgrid(us, vs)  # (Hs, Ws)
    d = depth[v.astype(np.int32), u.astype(np.int32)]
    m = valid[v.astype(np.int32), u.astype(np.int32)]
    
    # just a check to avoid processing if no valid depth pixels
    if not np.any(m):
        pcd = o3d.geometry.PointCloud()
        return pcd, np.zeros((0, 3), dtype=np.float64)
    
    # applying the mask

    u = u[m]; v = v[m]; d = d[m]

    # x = d
    dirs = np.stack([  
        np.ones_like(d),               
        (u - cx) / fx,                   
        -(v - cy) / fy
    ], axis=1)  # (N,3)

    #if depth_is_ray_distance:
    
    # normalize ray directions to unit vectors
    #dirs /= (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
        
    # points in camera frame
    pts_cam = dirs * d[:, None]  # (N,3)
    

    # camera pose to world
    R_wc = Rot.from_euler("xyz", [roll, pitch, yaw], degrees=False).as_matrix() 
    t_wc = np.array([x, y, z], dtype=np.float64)

    pts_world = (R_wc @ pts_cam.T).T + t_wc  

    # build Open3D point cloud 
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_world.astype(np.float64))

    if image is not None:
        img = np.asarray(image)
        if img.ndim == 3 and img.shape[2] >= 3:
            if img.shape[0] == H and img.shape[1] == W:
                cols = img[v.astype(np.int32), u.astype(np.int32), :3].astype(np.float32) / 255.0
                pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))

    return pcd, pts_world