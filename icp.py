from fast_update_transform import update_transform
import numpy as np
from cu_modules import icp_update_transform
from matplotlib import pyplot as plt
from scipy.spatial import KDTree

def transform_points(points, tx, ty, theta):
    """Transform points by translation (tx, ty) and rotation theta"""
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Rotation matrix
    R = np.array([[cos_theta, -sin_theta],
                  [sin_theta, cos_theta]])

    # Apply rotation then translation
    transformed = points.copy() @ R.T + np.array([tx, ty])
    return transformed

def vec_neighbors(points):
    tree = KDTree(points)
    idxs = np.array(tree.query(points, k=2))[1]
    idxs = idxs.astype(int)
    return points[idxs], tree

def getNaiveCorrespondence(src, segments, tree):
    """
    Gets the direct closest point and the previous segment to it.
    :param src: the points to transform
    :param tgt: the target points
    :return: Correspondences from the point to the nearest segment.
    """
    dists, indices = tree.query(src)
    indices = indices.astype(int)
    return [src, segments[indices][:, 0], segments[indices][:, 1]]

def projection(p, q1, q2):
    line_vec = q2 - q1
    point_to_start = p - q1
    t = np.dot(point_to_start, line_vec)/np.dot(line_vec, line_vec)
    # t = np.clip(t, 0.0, 1.0)
    return q1 + t*line_vec

def icp(og_src, tgt, show=False, max_iter=100, eps=1e-4):
    if show:
        fig, ax = plt.subplots()
    pose = np.array([0.0, 0.0, 0.0])
    cumulative_pose = np.zeros(3)
    src = og_src.copy()
    prev_err = 0
    err = float('inf')
    segments, tree = vec_neighbors(tgt)

    for _ in range(max_iter):
        if abs(err - prev_err) < eps:
            break
        prev_err = err
        src = transform_points(src, *pose)
        correspondences = getNaiveCorrespondence(src, segments, tree)
        if show:
            ax.clear()
            ax.scatter(*tgt.T, s=2)
            for p, q1, q2 in zip(*correspondences):
                ax.plot(*np.transpose([p, projection(p, q1, q2)]), c='k')
            ax.scatter(*src.T, s=2)
            plt.draw()
            plt.pause(0.5)
            # np.save("icp.npy", correspondences[1:])
        # print(np.hstack(correspondences[1:])[0])
        out = update_transform(*correspondences, max_iter=10)
        pose = out[:3]
        err = out[3]

        cumulative_pose += pose

    if show:
        plt.close(fig)
    return cumulative_pose, err, src

def icp_gpu(og_src: np.ndarray, maps: list[np.ndarray], poses: np.ndarray, max_iter=10, eps=1e-4, show=False):
    # icp_update_transform(og_src, q1s_all, q2s_all, poses)
    poses = poses.copy().astype(np.float32)  # ensure memory safety!
    n_particles = poses.shape[0]
    n_points = og_src.shape[0]
    segments_and_trees = [vec_neighbors(tgt) for tgt in maps]
    prev_err = np.full(n_particles, np.inf, dtype=np.float32)

    for iter_num in range(max_iter):
        q1s = np.zeros((n_particles * n_points, 2), dtype=np.float32)
        q2s = np.zeros((n_particles * n_points, 2), dtype=np.float32)
        for i, pose in enumerate(poses):
            tf_scan = transform_points(og_src, *pose)
            p, q1, q2 = getNaiveCorrespondence(tf_scan, *segments_and_trees[i])
            q1s[i * n_points:(i + 1) * n_points] = q1
            q2s[i * n_points:(i + 1) * n_points] = q2
        poses, err = icp_update_transform(og_src.copy(), q1s, q2s, poses)
        if np.all(np.abs(err - prev_err) < eps):
            break
        prev_err = err.copy()
    return poses, err

if __name__ == "__main__":
    from mapping import Map
    import cv2
    from custom_sim import LidarSim
    from time import perf_counter


    def bad_transform_points(points, tx, ty, theta):
        transformed = transform_points(points, tx, ty, theta)
        return transformed - 2 * np.array([tx, ty])

    map_img = 255 - cv2.threshold(cv2.imread("./lmap2.png", cv2.IMREAD_GRAYSCALE), 20, 255, cv2.THRESH_BINARY)[1]
    map = Map(map_img, 10)
    map.expand(1, val=1)
    start_pose = np.array([*map.map_center, 0])
    end_pose = start_pose + [-50, 10, 0]
    src = LidarSim(map).getScan(end_pose, True) + map.map_center
    tgt = map.toScan(True, True)
    src = bad_transform_points(src, *end_pose)
    src += map.map_center
    src *= map.px2m
    # plt.imshow(map_img, cmap="gray")
    # plt.show()
    plt.scatter(*tgt.T, s=3)
    plt.scatter(*src.T, s=2)
    plt.show()
    icp(src.copy(), tgt, show=False)
    # iterative_closest_point(src.copy(), tgt, return_score=True)

    start = perf_counter()
    pose1, err1, tf1_pts = icp(src.copy(), tgt)
    print(f"plicp time: {1/(perf_counter()-start)}")
    print(pose1, err1)
    start = perf_counter()
    pose2, err2 = icp_gpu(src.copy(), [tgt], np.array([[0, 0, 0]]))
    pose2 = pose2[0]
    err2 = err2[0]
    print(f"ogicp time: {1/(perf_counter()-start)}")
    print(pose1, err1)

    plt.scatter(*tgt.T, c='b', s=2)
    plt.scatter(*transform_points(src.copy(), *pose1).T, c='r', s=2)
    plt.scatter(*transform_points(src.copy(), *pose2).T, c='g', s=2)
    plt.show()