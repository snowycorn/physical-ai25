import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import pandas as pd

# =============================
# 1. Load map and color labels
# =============================
def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    img, points = params
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ' ', y)
        points.append((x, y))
        
        img_copy = img.copy()
        cv2.circle(img_copy, (x, y), 8, (0, 255, 0), -1)
        cv2.imshow('image', img_copy)
        
def click_input(map_path):
    # click the starting pixels on window
    img = cv2.imread(map_path, 1)
    points = []
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event, (img, points))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # get the clicked points
    if len(points) == 0:
        raise ValueError("No point clicked on map.")
    print("Start point:", points[-1])
    return points[-1]


def load_color_and_label_table(filepath="color_coding_semantic_segmentation_classes.xlsx"):
    df = pd.read_excel(filepath)
    color_dict = {}
    label_dict = {}

    for idx, row in df.iterrows():
        name = str(row['Name']).strip().lower()
        # 解析 "(R, G, B)" 格式
        rgb_str = row['Color_Code (R,G,B)']
        rgb = tuple(map(int, rgb_str.strip('()').split(',')))
        color_dict[name] = rgb
        # index
        label_dict[name] = idx + 1

    return color_dict, label_dict


# =============================
# Find target region
# =============================
def find_target_pixel(img, target_rgb):
    mask = np.all(img == target_rgb, axis=-1)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("Target color not found on the map.")
    
    # 計算質心
    center_x = np.mean(xs)
    center_y = np.mean(ys)
    target_point = np.array([center_x, center_y]).astype(int)
    return target_point
    
    # return np.stack([xs, ys], axis=1) # (N, 2) 所有 target category color 的 pixel 


# =============================
# 3. RRT Core Algorithm
# =============================
class Node:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent
        self.position = np.array([x, y])
        
class RRT:
    def __init__(self, map, start, target, max_iter=3000, step_size=40, goal_threshold=55, bias=0.1):
        self.map = map
        self.start = Node(x=start[0], y=start[1])
        self.goal = None
        self.target = target
        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_threshold = goal_threshold
        self.bias = bias
        
        self.nodes = []
        self.path = self.find_path()
        self.visualize()
        
    
    def random_sample(self):
        """Randomly sample a point with goal bias."""
        if random.random() < self.bias:
            # pick one target pixel randomly (from all matching category pixels)
            return self.target
        else:
            # uniform random sample from map area
            rand_x = random.randint(0, self.map.shape[1] - 1)
            rand_y = random.randint(0, self.map.shape[0] - 1)
            return np.array([rand_x, rand_y])

    def nearest(self, rand_position):
        """ Find nearest node of rand_position """
        nearest_node = min(self.nodes, key=lambda node: np.linalg.norm(node.position - rand_position))
        return nearest_node
    
    def steer(self, parent_pos, random_pos):
        """ Steer towards random point """
        direction = random_pos - parent_pos
        distance = np.linalg.norm(direction)
        
        if distance <= self.step_size:
            return random_pos
        else:
            new_pos = parent_pos + direction * self.step_size / distance
            return new_pos.astype(int)
    
    def obstacle_free(self, parent_pos, child_pos):
        """ Check the map to see if the route encounter obstacle """
        num_points = int(np.linalg.norm(child_pos - parent_pos))  # 根據距離決定採樣密度
        xs = np.linspace(parent_pos[0], child_pos[0], num_points).astype(int)
        ys = np.linspace(parent_pos[1], child_pos[1], num_points).astype(int)
        
        for x, y in zip(xs, ys):
            if np.any(self.map[y, x] != [255, 255, 255]):  # map y 是 row, x 是 column
                return False
        return True
    
    def find_path(self):
        # apply RRT algorithm
        self.nodes.append(self.start)
        
        for i in range(self.max_iter):
            rand_position = self.random_sample()
            nearest_node = self.nearest(rand_position)
            new_position = self.steer(nearest_node.position, rand_position)
            
            if self.obstacle_free(nearest_node.position, new_position):
                new_node = Node(x=new_position[0], y=new_position[1], parent=nearest_node)
                self.nodes.append(new_node)
                # check if new_node is close enough to any pixel in self.target (N, 2)
                distances = np.linalg.norm(self.target - new_node.position)
                if distances < self.goal_threshold:
                    self.goal = new_node
                    print(f"Goal reached in {i+1} iterations.")
                    break
                
        # return the final path as a numpy list of (x, y) from start to goal
        if self.goal is None:
            print("No goal found.")
            return []

        path = []
        current = self.goal
        while current is not None:
            path.append(current.position)
            current = current.parent
            
        # reverse to make it from start → goal
        path.reverse()  
        return np.array(path)   # (N, 2)

    def visualize(self, target_category=""):
        vis_map = self.map.copy()

        # Draw all edges (black lines)
        for node in self.nodes:
            if node.parent is not None:
                cv2.line(vis_map, node.position, node.parent.position, (0, 0, 0), 1)

        # Draw all nodes (purple circle)
        for node in self.nodes:
            if node is not self.start and node is not self.goal:
                cv2.circle(vis_map, node.position, 3, (255, 0, 255), -1)

        # Draw start (green circle)
        cv2.circle(vis_map, self.start.position, 8, (0, 255, 0), -1)
        
        # Check if the path if found
        if self.goal:
            # Draw goal (cyan circle)
            cv2.circle(vis_map, self.goal.position, 8, (0, 255, 255), -1)

            # Draw final path (red line)
            prev_position = None
            for position in self.path:
                if prev_position is not None:
                    cv2.line(vis_map, position, prev_position, (255, 0, 0), 2)
                prev_position = position
        else:
            # Draw the mean of target if the path can't found
            cv2.circle(vis_map, self.target, 8, (0, 255, 255), -1)

        vis_map = cv2.cvtColor(vis_map, cv2.COLOR_RGB2BGR)
        cv2.imshow("Path Visualization", vis_map)
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# =============================
# Main
# =============================
if __name__ == "__main__":
    map_path = "map.png"
    
    start_position = click_input(map_path)
        
    # Load data
    map_img = cv2.imread(map_path)
    map_img = cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB)
    color_table, _ = load_color_and_label_table()

    # Find target color
    target_category = input("Enter target category (e.g. sofa, rack, stair, cooktop): ").strip().lower()
    target_rgb = color_table[target_category]
    target = find_target_pixel(map_img, target_rgb)

    # Run RRT and draw result
    rrt = RRT(map_img, start_position, target)