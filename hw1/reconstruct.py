import numpy as np
import open3d as o3d
import argparse
import os

# ------------------------------
# Camera intrinsic parameters
# ------------------------------
WIDTH = 512
HEIGHT = 512
FOV = 90

fx = fy = WIDTH / (2 * np.tan(np.deg2rad(FOV / 2)))
cx = WIDTH / 2
cy = HEIGHT / 2
intrinsic = o3d.camera.PinholeCameraIntrinsic(WIDTH, HEIGHT, fx, fy, cx, cy)

def depth_image_to_point_cloud(rgb, depth):
    # TODO: Get point cloud from rgb and depth image 
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb, depth, depth_scale=1, depth_trunc=5, convert_rgb_to_intensity=False
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    pcd.transform([[1, 0, 0, 0], 
                   [0, -1, 0, 0], 
                   [0, 0, -1, 0], 
                   [0, 0, 0, 1]])
    
    return pcd


def preprocess_point_cloud(pcd, voxel_size):
    # TODO: Do voxelization to reduce the number of points for less memory usage and speedup
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    # Estimate normal with search radius voxel_size * 2
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    # Compute FPFH feature with search radius voxel_size * 5
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


def local_icp_algorithm(source_down, target_down, trans_init, threshold):
    # TODO: Use Open3D ICP function to implement
    result = o3d.pipelines.registration.registration_icp(
        source_down, target_down, threshold, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    return result


def my_local_icp_algorithm(source_down, target_down, trans_init, voxel_size):
    # TODO: Write your own ICP function
    # ICP 參數
    max_iterations = 30
    distance_threshold = voxel_size * 1.5
    tolerance = 1e-6
    
    # 將點雲轉成 numpy 格式
    source_points = np.asarray(source_down.points)
    target_points = np.asarray(target_down.points)
    target_normals = np.asarray(target_down.normals)

    # 初始化變換矩陣
    trans = trans_init.copy()

    for _ in range(max_iterations):
        # change source point to homogeneous form, transfrom, then turn points back
        source_homo = np.hstack((source_points, np.ones((source_points.shape[0], 1))))
        transformed_source = (trans @ source_homo.T).T[:, :3]

        # 建立 KDTree 找最近點對應
        target_kd_tree = o3d.geometry.KDTreeFlann(target_down)
        correspondences = []
        for i, p in enumerate(transformed_source):
            [k, idx, dist] = target_kd_tree.search_knn_vector_3d(p, 1)
            if k > 0 and dist[0] < distance_threshold ** 2:
                correspondences.append((i, idx[0]))

        # Point-to-Plane
        A = []
        b = []
        for (i, j) in correspondences:
            p = transformed_source[i]
            q = target_points[j]
            n = target_normals[j]

            # A_i = [n × p, n]
            cross = np.cross(p, n)
            A.append(np.hstack((cross, n)))
            b.append(np.dot(n, q - p))

        A = np.array(A)
        b = np.array(b).reshape(-1, 1)

        # 解線性最小平方 (Ax = b) 
        # x = [alpha, beta, gamma, tx, ty, tz]
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        # 將 x 轉成小旋轉矩陣 + 平移
        alpha, beta, gamma, tx, ty, tz = x.flatten()
        R_delta = o3d.geometry.get_rotation_matrix_from_xyz([alpha, beta, gamma])
        t_delta = np.array([tx, ty, tz])
        
        delta_T = np.eye(4)
        delta_T[:3, :3] = R_delta
        delta_T[:3, 3] = t_delta

        # 更新總變換 
        trans = delta_T @ trans

        # 收斂判斷
        update_norm = np.linalg.norm(x)
        if update_norm < tolerance:
            break

    # === 回傳與 open3d 相同格式的結果 ===
    result = o3d.pipelines.registration.RegistrationResult()
    result.transformation = trans
    return result

def reconstruct(args):
    # TODO: Return results
    """
    For example:
        ...
        args.version == 'open3d':
            trans = local_icp_algorithm()
        args.version == 'my_icp':
            trans = my_local_icp_algorithm()
        ...
    """
    voxel_size = 0.05
    
    rgb_files_path = os.path.join(args.data_root, "rgb")
    depth_files_path = os.path.join(args.data_root, "depth")
    file_num = len(os.listdir(rgb_files_path))
    
    pcds = []
    pred_cam_pos = [np.eye(4)]
    
    for i in range(1, file_num + 1):
        file_name = str(i) + ".png"
        rgb_file = os.path.join(rgb_files_path, file_name)
        depth_file = os.path.join(depth_files_path, file_name)
        
        rgb = o3d.io.read_image(rgb_file)
        depth = o3d.io.read_image(depth_file)
        # 把 depth 還原成以米為單位
        depth_np = np.asarray(depth).astype(np.float32)
        depth_np = depth_np / 255 * 10 
        depth = o3d.geometry.Image(depth_np)
        
        # construct point cloud
        pcd = depth_image_to_point_cloud(rgb, depth)
        
        if i != 1:
            # voxelization
            target_down, target_fpfh = preprocess_point_cloud(pcd, voxel_size)
            source_down, source_fpfh = preprocess_point_cloud(pcds[-1], voxel_size)
            
            # global registration
            result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
            
            # local regsitration
            if args.version == 'open3d':
                result_icp = local_icp_algorithm(source_down, target_down, result_ransac.transformation, voxel_size * 0.4)
            else:
                result_icp = my_local_icp_algorithm(source_down, target_down, result_ransac.transformation, voxel_size * 0.4)

            # transform current pcd
            cam_pose = np.linalg.inv(result_icp.transformation)
            pred_cam_pos.append(cam_pose)
            pcd.transform(cam_pose)
            # o3d.visualization.draw_geometries([target_down])
            # o3d.visualization.draw_geometries([pcd, pcds[-1]])
        
        pcds.append(pcd)

    # merge point cloud
    result_pcd = o3d.geometry.PointCloud()
    for p in pcds:
        result_pcd += p
    
    return result_pcd, pred_cam_pos


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--floor', type=int, default=1)
    parser.add_argument('-v', '--version', type=str, default='my_icp', help='open3d or my_icp')
    parser.add_argument('--data_root', type=str, default='data_collection/first_floor/')
    args = parser.parse_args()

    if args.floor == 1:
        args.data_root = "data_collection/first_floor/"
    elif args.floor == 2:
        args.data_root = "data_collection/second_floor/"
    
    # TODO: Output result point cloud and estimated camera pose
    '''
    Hint: Follow the steps on the spec
    '''
    result_pcd, pred_cam_pos = reconstruct(args)
    # print("Point cloud bounds:", np.min(points, axis=0), np.max(points, axis=0))
    # o3d.visualization.draw_geometries([result_pcd])

    # TODO: Calculate and print L2 distance
    '''
    Hint: Mean L2 distance = mean(norm(ground truth - estimated camera trajectory))
    '''
    gt_pose_file = os.path.join(args.data_root, "GT_pose.npy")
    gt_poses = np.load(gt_pose_file)  # shape: (N, 7)
    
    # 取出 ground truth 和 predicted camera 的平移向量 (x, y, z)
    gt_traj = np.array([pose[:3] - gt_poses[0][:3] for pose in gt_poses])
    pred_traj = np.array([pose[:3, 3] for pose in pred_cam_pos])
    
    # L2 distance
    l2_distances = np.linalg.norm(gt_traj - pred_traj, axis=1)
    mean_l2 = np.mean(l2_distances)
    
    print("Mean L2 distance: ", mean_l2)

    # TODO: Visualize result
    '''
    Hint: Sould visualize
    1. Reconstructed point cloud
    2. Red line: estimated camera pose
    3. Black line: ground truth camera pose
    '''
    # 去除天花板
    points = np.asarray(result_pcd.points)
    mask = points[:, 1] < 0.4 # 假設 y 軸是高
    filtered_points = points[mask]
    filtered_colors = np.asarray(result_pcd.colors)[mask]
    
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

    # 建立紅色線段（estimated trajectory）
    pred_lines = []
    for i in range(len(pred_traj) - 1):
        pred_lines.append([i, i + 1])
    pred_line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pred_traj),
        lines=o3d.utility.Vector2iVector(pred_lines),
    )
    pred_colors = [[1, 0, 0] for _ in range(len(pred_lines))]  # 紅色
    pred_line_set.colors = o3d.utility.Vector3dVector(pred_colors)

    # 建立黑色線段（ground truth trajectory）
    gt_lines = []
    for i in range(len(gt_traj) - 1):
        gt_lines.append([i, i + 1])
    gt_line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(gt_traj),
        lines=o3d.utility.Vector2iVector(gt_lines),
    )
    gt_colors = [[0, 0, 0] for _ in range(len(gt_lines))]  # 黑色
    gt_line_set.colors = o3d.utility.Vector3dVector(gt_colors)

    # 顯示所有物件
    o3d.visualization.draw_geometries(
        [filtered_pcd, pred_line_set, gt_line_set],
        window_name="3D Reconstruction and Trajectory",
        width=960,
        height=720,
    )