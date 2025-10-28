# physical-ai-hw2
NYCU Physical AI 2025 Fall

Spec: https://drive.google.com/file/d/1jg5wRDpTQcx7Ux01hNzPmMdKGN-Mhxc0/view?usp=sharing

## Environments

請確保已安裝以下套件:

```bash
    pip install pandas openpyxl matplotlib opencv-python
```

## Run the Program

### 1. Modify Path

Modify input paths in `src/main.py` if needed:

```python
    # point cloud path
    point_path = "semantic_3d_pointcloud/point.npy"
    color_path = "semantic_3d_pointcloud/color01.npy"
    color_table_path = "color_coding_semantic_segmentation_classes.xlsx"
    # scene path
    test_scene = "replica_v1/apartment_0/habitat/mesh_semantic.ply"
    info_semantic = "replica_v1/apartment_0/habitat/info_semantic.json"
    # output
    result_dir = "results"
```

### 2. Run the Code

Execute the following command from the project root directory:

```bash
    python src/main.py
```

### 3. Output

The output map.png and navigation video can be find in `results` file by defaults
