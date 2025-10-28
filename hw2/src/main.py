import cv2
import numpy as np
import os
from map_construction import map_construction
from RRT import RRT, click_input, load_color_and_label_table, find_target_pixel
from navigation import navigatePath


def pixel_to_world2d(pixel_position, pixel_ref, world_ref):
    world_ref_swapped = world_ref[:, [1, 0]]
    # 計算 x、y 方向縮放比例
    scale_x = (world_ref_swapped[1,0] - world_ref_swapped[0,0]) / (pixel_ref[1,0] - pixel_ref[0,0])
    scale_y = (world_ref_swapped[1,1] - world_ref_swapped[0,1]) / (pixel_ref[1,1] - pixel_ref[0,1])

    world_x = world_ref_swapped[0,0] + (pixel_position[0] - pixel_ref[0,0]) * scale_x
    world_y = world_ref_swapped[0,1] + (pixel_position[1] - pixel_ref[0,1]) * scale_y
    return np.array([world_y, world_x])


if __name__ == '__main__':
    # point cloud path
    point_path = "semantic_3d_pointcloud/point.npy"
    color_path = "semantic_3d_pointcloud/color01.npy"
    color_table_path = "color_coding_semantic_segmentation_classes.xlsx"
    # scene path
    test_scene = "replica_v1/apartment_0/habitat/mesh_semantic.ply"
    info_semantic = "replica_v1/apartment_0/habitat/info_semantic.json"
    # output
    result_dir = "results"
    
    map_path = os.path.join(result_dir, "map.png")
    
    # Part 1: construct map
    world_ref, pixel_ref = map_construction(point_path, color_path, map_path)

    # Part 2: RRT algorithm
    # input
    start_position = click_input(map_path)
    target_category = input("Enter target category (e.g. sofa, rack, stair, cooktop): ").strip().lower()
    
    # Load map.png in RGB format
    map_img = cv2.imread(map_path)
    map_img = cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB)
    height, width = map_img.shape[:2]
    
    # find position for target color
    color_table, label_table = load_color_and_label_table(color_table_path)
    target_color = color_table[target_category]
    target_label = label_table[target_category]
    print(target_label)
    target_position = find_target_pixel(map_img, target_color)
    
    # run RRT and draw result
    rrt = RRT(map_img, start_position, target_position)
    
    # Part 3: Robot navigation
    # turn pixel coordinates to xyz-coordinate
    world_2d_path = []
    for pixel_position in rrt.path:
        world_position = pixel_to_world2d(pixel_position, pixel_ref, world_ref)
        world_2d_path.append(world_position)
        
    # navigate through the path
    save_video_path = os.path.join(result_dir, f"{target_category}.mp4")
    navigatePath(world_2d_path, target_label, test_scene=test_scene, info_semantic=info_semantic, save_video_path=save_video_path)