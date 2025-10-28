import numpy as np
import matplotlib.pyplot as plt
import cv2


def map_construction(point_path="semantic_3d_pointcloud/point.npy", color_path="semantic_3d_pointcloud/color01.npy", map_path="map.png"):
    # === Load data ===
    points = np.load(point_path)    # shape (N, 3)
    colors = np.load(color_path)    # shape (N, 3), range [0,1]

    # === Remove ceiling and floor points ===
    # use the y-axis to filter
    y = points[:, 1]
    floor_threshold = np.percentile(y, 22)
    ceiling_threshold = np.percentile(y, 55)
    mask = (y > floor_threshold) & (y < ceiling_threshold)

    points_filtered = points[mask]
    colors_filtered = colors[mask]
    
    # === Convert coordinates to 2D (x,z) and scale ===
    scale_factor = 10000.0 / 255.0
    coords_2d = points_filtered[:, [0, 2]] * scale_factor
    
    # === Add reference point ===
    world_ref = np.array([[0, -2], [2, 1]])
    coords_2d = np.vstack([coords_2d, world_ref])
    colors_filtered = np.vstack([colors_filtered, [[0, 0, 0]], [[0, 0, 0]]]) 

    plt.figure(figsize=(10, 10))
    plt.scatter(
        coords_2d[:, 1], 
        coords_2d[:, 0], 
        c=colors_filtered, 
        s=1)
    plt.axis("equal")
    plt.axis("off")

    plt.savefig(map_path, dpi=80, bbox_inches='tight', pad_inches=0)

    print(f"2D semantic map saved as {map_path}")

    # === Load saved map to find reference pixel ===
    img = cv2.imread(map_path, cv2.IMREAD_COLOR)  # BGR
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    mask = np.all(np.abs(img_rgb - np.array([0,0,0])) <= 5, axis=2)
    ys, xs = np.where(mask)
    if len(xs) != 2:
        raise ValueError(f"Should find 2 reference point but find {len(xs)}")

    pixel_ref = np.stack([xs, ys], axis=1)
    # 依 x 座標升序排序
    sorted_idx = np.argsort(pixel_ref[:,0])
    pixel_ref = pixel_ref[sorted_idx]
    
    print(f'world coordinate: {world_ref}')
    print(f'pixel coordinate: {pixel_ref}')
    return world_ref, pixel_ref