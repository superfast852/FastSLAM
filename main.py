import numpy as np
from custom_sim import RobotSimulation, wall_following
from fastslam import SLAM
from tools import normalize_angle, opencv_to_gif

sim = RobotSimulation()
slam = SLAM(20, 800, sim.map.map_meters, 0.5, (1, 1, 1), thick_register=True)
prev_pose = None
frames = []
np.random.seed(42)
def func(pose, scan):
    global prev_pose
    og_pose = pose.copy()
    og_scan = scan.copy()

    pose[:2] *= sim.map.px2m
    scan     *= sim.map.px2m

    if prev_pose is None:
        prev_pose = pose
        slam.update(np.zeros((3,)), scan)
        return wall_following(og_pose, og_scan)
    diff = pose - prev_pose
    diff[2] = normalize_angle(diff[2])
    prev_pose = pose

    pose_est, map = slam.update(diff, scan)
    frame = slam.animate_alpha(False)
    frames.append(frame)
    return wall_following(og_pose, og_scan)

sim.run(func, cartesian=True)
opencv_to_gif(frames, "demo.gif")