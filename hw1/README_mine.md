# 3D Reconstruction and BEV Projection

This repository contains implementations for two tasks:  
1. Projecting points from a top-view (BEV) image to a front-view image.  
2. Reconstructing 3D point clouds from RGB-D images using ICP registration.

---

## Task 1: Top-to-Front View Projection

You can project selected points from a top-view image to a front-view image by the code in `bev.py`

### Step 1: Modify input paths** in `main` if needed:

```python
front_rgb = "bev_data/front1.png"
top_rgb = "bev_data/bev1.png"
```

### Step 2: Run the Program

```bash
python bev.py
```

### Output
The projected image will be saved as `projection.png` in the current directory.

---

## Task 2: Point Cloud Reconstruction

### Step 1: Load Dataset
By default, the program loads the first floor dataset. You can load the second floor as follows:

```bash
python load.py           # Loads first floor dataset
python load.py -f 2      # Loads second floor dataset
```

### Step 2: Run Reconstruction
By default, the program uses my custom ICP implementation and the first floor dataset. You can modify options:

```bash
python reconstruct.py            # My ICP on first floor dataset
python reconstruct.py -v open3d  # Open3D ICP on first floor dataset
python reconstruct.py -f 2       # My ICP on second floor dataset
python reconstruct.py -v open3d -f 2  # Open3D ICP on second floor dataset
```

### Output
- Reconstructed 3D point cloud (visualized using Open3D).
- Estimated camera trajectory (red line) vs. ground truth trajectory (black line).
- Mean L2 distance between estimated and ground truth trajectories printed in the console.