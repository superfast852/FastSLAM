import cv2
import numpy as np
from custom_sim import RobotSimulation, wall_following
from fastslam import SLAM, get_movement_noise
from tools import normalize_angle, opencv_to_gif

translation_std = 1
rotation_std = 1

sim = RobotSimulation("./lmap2.png", random_rot=False)
np.random.seed(42)
slam = SLAM(10, 800, sim.map.map_meters, 0.5, (translation_std, translation_std, rotation_std), thick_register=True)
prev_pose = None
frames = []

def func(pose, scan):
    global prev_pose
    # TODO: ax.scatter the skeletonized toScan map to see what's going on there
    og_pose = pose.copy()
    og_scan = scan.copy()

    pose[:2] *= sim.map.px2m
    scan     *= sim.map.px2m

    if prev_pose is None:
        prev_pose = pose
        slam.update(np.zeros((3,)), scan)
        return wall_following(og_pose, og_scan)
    diff = pose - prev_pose
    diff += get_movement_noise(translation_std, translation_std, rotation_std, diff[0], diff[1], diff[2], 1).flatten()
    diff[2] = normalize_angle(diff[2])
    prev_pose = pose

    pose_est, map = slam.update(diff, scan)
    frame = slam.animate_alpha(False)
    frames.append(frame)
    cv2.imshow("FastSLAM", frame)
    cv2.waitKey(1)
    return wall_following(og_pose, og_scan)

sim.run(func, max_steps=1000, cartesian=True, show=False)
opencv_to_gif(frames, "demo_short.gif")