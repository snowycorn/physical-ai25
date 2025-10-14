import cv2
import numpy as np

points = []

class Projection(object):

    def __init__(self, image_path, points):
        """
            :param points: Selected pixels on top view(BEV) image
        """

        if type(image_path) != str:
            self.image = image_path
        else:
            self.image = cv2.imread(image_path)
        self.height, self.width, self.channels = self.image.shape
        self.points = np.array(points, dtype=np.float32)

    def top_to_front(self, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0, fov=90):
        """
            Project the top view pixels to the front view pixels.
            :return: New pixels on perspective(front) view image
        """

        ### TODO ###
        # --- 相機內參 ---
        f = (self.width / 2) / np.tan(np.deg2rad(fov / 2))
        K = np.array([
            [f, 0, self.width / 2],
            [0, f, self.height / 2],
            [0, 0, 1]
        ])
        K_inv = np.linalg.inv(K)

        # --- 相機外參 ---
        # Front camera to world
        R_front_to_world = self._rotation_matrix(0, 0, 0)
        C_front = np.array([[0], [1], [0]])
        
        # Top camera to world
        R_top_to_world = self._rotation_matrix(np.deg2rad(theta), np.deg2rad(phi), np.deg2rad(gamma)) 
        C_top = np.array([[0], [2.5], [0]])
                
        # --- 投影 ---
        # Top pixel 2D -> Top camera 3D
        points_top_pixel = np.hstack([self.points, np.ones((len(self.points), 1))]).T  # 3xN
        rays_top = K_inv @ points_top_pixel  # 3xN
        
        # Rotate to world
        rays_world = R_top_to_world @ rays_top  # 3xN
        # Intersect ground Y=0 in world coordinates
        t = - C_top[1,0] / rays_world[1, :]   # scale
        points_world = C_top + rays_world * t  # 3xN

        # World -> Front camera 3D
        points_front = R_front_to_world.T @ (points_world - C_front)

        # Project to front pixel 2D
        proj_points = K @ points_front
        proj_points /= proj_points[2, :]
        new_pixels = proj_points[:2, :].T.astype(np.int32)
        
        return new_pixels
    
    def _rotation_matrix(self, theta, phi, gamma):
        """Return rotation matrix """
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(theta), -np.sin(theta)],
                       [0, np.sin(theta), np.cos(theta)]])
        Ry = np.array([[np.cos(phi), 0, np.sin(phi)],
                       [0, 1, 0],
                       [-np.sin(phi), 0, np.cos(phi)]])
        Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                       [np.sin(gamma), np.cos(gamma), 0],
                       [0, 0, 1]])
        return Rz @ Ry @ Rx
    
    
    def show_image(self, new_pixels, img_name='projection.png', color=(0, 0, 255), alpha=0.4):
        """
            Show the projection result and fill the selected area on perspective(front) view image.
        """

        new_image = cv2.fillPoly(
            self.image.copy(), [np.array(new_pixels)], color)
        new_image = cv2.addWeighted(
            new_image, alpha, self.image, (1 - alpha), 0)

        cv2.imshow(
            f'Top to front view projection {img_name}', new_image)
        cv2.imwrite(img_name, new_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return new_image


def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        print(x, ' ', y)
        points.append([x, y])
        font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img, str(x) + ',' + str(y), (x+5, y+5), font, 0.5, (0, 0, 255), 1)
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('image', img)

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:

        print(x, ' ', y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        # cv2.putText(img, str(b) + ',' + str(g) + ',' + str(r), (x, y), font, 1, (255, 255, 0), 2)
        cv2.imshow('image', img)


if __name__ == "__main__":

    pitch_ang = -90

    front_rgb = "bev_data/front1.png"
    top_rgb = "bev_data/bev1.png"

    # click the pixels on window
    img = cv2.imread(top_rgb, 1)
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    projection = Projection(front_rgb, points)
    new_pixels = projection.top_to_front(theta=pitch_ang)
    projection.show_image(new_pixels)
