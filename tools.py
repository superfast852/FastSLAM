from numba import njit
import numpy as np
from PIL import Image

@njit(fastmath=True)
def line2dots(a: tuple, b: tuple):
    x1, y1, x2, y2 = a[0], a[1], b[0], b[1]
    points = []
    issteep = abs(y2-y1) > abs(x2-x1)
    if issteep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    rev = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        rev = True
    deltax = x2 - x1
    deltay = abs(y2-y1)
    error = int(deltax / 2)
    y = y1
    ystep = None
    if y1 < y2:
        ystep = 1
    else:
        ystep = -1
    for x in range(x1, x2 + 1):
        if issteep:
            points.append((y, x))
        else:
            points.append((x, y))
        error -= deltay
        if error < 0:
            y += ystep
            error += deltax
    # Reverse the list if the coordinates were reversed
    if rev:
        points.reverse()
    return points


def padMap(map, fill=1):
    assert len(map.shape) == 2
    max_dim = max(map.shape)

    # Calculate padding for each dimension
    pad_height = max_dim - map.shape[0]
    pad_width = max_dim - map.shape[1]

    # Distribute padding evenly, with extra padding going to the end
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top

    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    # Apply padding: ((top, bottom), (left, right))
    map_img = np.pad(map, ((pad_top, pad_bottom), (pad_left, pad_right)),
                     mode="constant", constant_values=fill)

    return map_img


def to_world(pose, local_points):
    """
    Transforms a set of local (robot-frame) points to the world frame.

    :param robot_pos: Tuple (X_r, Y_r) representing the robot's world position
    :param robot_theta: Float, robot's orientation in radians
    :param local_points: List of (x, y) tuples in the robot's local frame
    :return: List of transformed (X, Y) tuples in world coordinates
    """
    X_r, Y_r, robot_theta = pose
    cos_theta = np.cos(robot_theta)
    sin_theta = np.sin(robot_theta)

    world_points = np.array([
        (X_r + x * cos_theta - y * sin_theta,
         Y_r + x * sin_theta + y * cos_theta)
        for x, y in local_points
    ])

    return world_points


def normalize_angle(angle):
    """Normalize angle to [-pi, pi]"""
    return np.arctan2(np.sin(angle), np.cos(angle))


def native_normalize_angle(angle):
    """Normalize angle to [-pi, pi] using only while loops."""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def norm_angle_vec(angles):
    return np.arctan2(np.sin(angles), np.cos(angles))


def world_to_grid(pose, map):
    # With the pose being in meters, turn it into a location on the map in px.
    x, y = pose[0], pose[1]
    return int(x * map.m2px), int(y * map.m2px)


def transform_to_pose(T):
    R, t = T[:-1, :-1], T[:-1, -1]
    return np.array([t[0], t[1], np.arctan2(R[1, 0], R[0, 0])])


def pose_to_transform(pose):
    """Convert pose [x, y, theta] to transformation matrix"""
    x, y, theta = pose
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, x],
        [s, c, y],
        [0, 0, 1]
    ])


import cv2
from PIL import Image
import numpy as np


def opencv_to_gif(images, output_path, duration=10):
    """
    Convert OpenCV images to GIF

    Args:
        images: List of OpenCV images (numpy arrays)
        output_path: Path to save the GIF
        duration: Duration between frames in milliseconds
    """
    # Convert OpenCV images (BGR) to PIL images (RGB)
    pil_images = []
    for img in images:
        # Convert BGR to RGB
        pil_img = Image.fromarray(img[:, :, ::-1])
        pil_images.append(pil_img)

    # Save as GIF
    pil_images[0].save(
        output_path,
        save_all=True,
        append_images=pil_images[1:],
        duration=duration,
        loop=0
    )
