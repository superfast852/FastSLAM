import numpy as np
import scipy.optimize as optimize
from matplotlib import pyplot as plt
from numba import njit
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors
from tools import to_world, transform_to_pose, pose_to_transform

fig, ax = plt.subplots()
nbrs = NearestNeighbors(n_neighbors=1)


def compute_normals(points, k=10):
    """
    Estimate surface normals using PCA on k nearest neighbors.
    """
    tree = KDTree(points)
    normals = []
    for point in points:
        dists, idxs = tree.query(point, k=k + 1)  # includes the point itself
        neighbors = points[idxs[1:]]  # exclude the point itself
        cov = np.cov(neighbors - neighbors.mean(axis=0), rowvar=False)
        normal = np.linalg.svd(cov)[2][-1]
        normal /= np.linalg.norm(normal)
        normals.append(normal)
    return np.array(normals)


def vec_normals(points, k=5, returnTree=False):
    tree = KDTree(points)
    dists, idxs = np.array(tree.query(points, k=k+1))[:, :, 1:]
    idxs = idxs.astype(int)
    q1 = points[idxs][:, 0]
    q2 = points[idxs][:, -1]
    segments = np.stack([q1, q2], axis=1)
    normals = np.array([q1[:, 1] - q2[:, 1], q2[:, 0] - q1[:, 0]], dtype=np.float32).T
    normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]
    if returnTree:
        return normals, tree
    return normals


@njit(fastmath=True, cache=True)
def objective_point_to_plane_jit(tx, ty, theta, source_points, target_points, target_normals):
    """JIT-compiled point-to-plane error function"""
    # Create rotation matrix
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([
        [c, -s],
        [s, c]
    ])

    # Apply transformation to source points
    transformed = source_points @ R.T + np.array([tx, ty])

    # Calculate point-to-plane distances
    differences = target_points - transformed
    distances = np.sum(differences * target_normals, axis=1)

    return distances


def objective_point_to_plane(x, source_points, target_points, target_normals):
    """Wrapper for JIT-compiled function"""
    tx, ty, theta = x
    return objective_point_to_plane_jit(tx, ty, theta, source_points, target_points, target_normals)


@njit(fastmath=True, cache=True)
def jacobian_point_to_plane_jit(theta, source_points, target_normals):
    """JIT-compiled Jacobian calculation"""
    # Extract normal components for cleaner code
    nx = target_normals[:, 0]
    ny = target_normals[:, 1]

    # Source point coordinates
    px = source_points[:, 0]
    py = source_points[:, 1]

    # Precompute trig functions
    c = np.cos(theta)
    s = np.sin(theta)

    # Jacobian matrix
    n_points = len(source_points)
    J = np.zeros((n_points, 3))

    # Fill the Jacobian matrix
    for i in range(n_points):
        # Derivative with respect to tx: -nx
        J[i, 0] = -nx[i]

        # Derivative with respect to ty: -ny
        J[i, 1] = -ny[i]

        # Derivative with respect to theta
        J[i, 2] = nx[i] * (py[i] * c + px[i] * s) - ny[i] * (px[i] * c - py[i] * s)

    return J


def jacobian_point_to_plane(x, source_points, target_points, target_normals):
    """Wrapper for JIT-compiled Jacobian function"""
    return jacobian_point_to_plane_jit(x[2], source_points, target_normals)


def subsample_points(points, target_count):
    """Subsample points to target count while preserving contour"""
    n = len(points)
    if n <= target_count:
        return points

    # Use stride-based sampling (more efficient than random)
    stride = n // target_count
    indices = np.arange(0, n, stride)[:target_count]
    return points[indices]


def transform_points(T, points):
    return np.dot(points, T[:2, :2].T) + T[:2, 2]


def iterative_closest_point(data, target, max_iterations=100, tol=1e-5,
                            subsample = False, max_points=1000, initial_guess=np.zeros(3),
                            x0_guess = np.zeros(3), debug=False, return_score=False, animate=False):
    """
    Estimates the transformation between two point clouds using the Iterative Closest Point (ICP) algorithm.
    :param data: Nx2-shaped array of points to transform into target
    :param target: the goal pose of the data after the estimated transformation
    :param max_iterations: A limit on how many iterations to run.
    :param tol: threshold of error reduction between two iterations. It stops when it converges, NOT at that error.
    :param subsample: If true, reduces the number of points to max_points. Useful for large point clouds.
    :param max_points: a limit to the number of points to use, assuming subsample is enabled.
    :param initial_guess: transforms the data point into the initial_guess frame.
    :param debug: Print the final error and number of iterations taken
    :param return_score: return the initial sum of squared distances after correlation
    :return: The target transformation, and optionally the initial error
    """

    if subsample:
        if len(data) > max_points:
            data = subsample_points(data, max_points)
        if len(target) > max_points:
            target = subsample_points(target, max_points)

    src_points = to_world(initial_guess, data.copy())

    T_accumulated = np.eye(3)
    prev_error = float('inf')
    mean_error = float('inf')
    # target_normals = compute_normals(target)
    target_normals = vec_normals(target)
    x0 = x0_guess.copy()
    nbrs.fit(target)
    init_err = None
    for i in range(max_iterations):
        distances, indices = np.array(nbrs.kneighbors(src_points, 1)).reshape(2, -1)
        indices = indices.astype(int)
        matched_target = target[indices]
        matched_normals = target_normals[indices]

        # For early iterations, use faster but less accurate optimization
        method = 'lm' if i > 3 else 'trf'

        # Optimize transformation
        result = optimize.least_squares(
            objective_point_to_plane, x0,
            jac=jacobian_point_to_plane,
            args=(src_points, matched_target, matched_normals),
            method=method,
            ftol=1e-5 if i > 3 else 1e-3,  # Looser tolerance for early iterations
        )

        tx, ty, theta = result.x
        # Create transformation matrix
        c, s = np.cos(theta), np.sin(theta)
        T_step = np.array([[c, -s, tx],
                           [s, c, ty],
                           [0, 0, 1.0]])

        # Apply the transformation step to the current source points
        src_points = transform_points(T_step, src_points)
        T_accumulated = np.dot(T_step, T_accumulated)
        mean_error = np.mean(distances)
        if animate:
            ax.clear()
            ax.scatter(*src_points.T, s=1, c='r')
            ax.scatter(*target.T, s=1, c='b')
            plt.draw()
            plt.pause(0.1)
        if np.abs(prev_error - mean_error) < tol:
            break

        prev_error = mean_error

    if debug:
        print(f"Final error: {mean_error}, Iterations: {i + 1}")

    T_accumulated = np.dot(pose_to_transform(initial_guess), T_accumulated)
    if return_score:
        distances = np.array(nbrs.kneighbors(src_points, 1)).reshape(2, -1)[0]
        init_err = np.sum(distances ** 2)
        return T_accumulated, init_err

    return T_accumulated


def simple_icp(src, tgt, show=True, abs_err=1e-4, rel_err=1e-4, max_iter=20):
    tree = KDTree(src)
    dist = np.inf
    prev_dist = 0
    rot = np.eye(2)
    translation = np.zeros(2)
    for i in range(max_iter):
        distances, indices = tree.query(tgt)
        prev_dist = dist
        dist = np.sum(distances**2)
        print(dist)
        pairs = np.hstack([src[indices], tgt]).reshape(-1, 2, 2)
        centroid_src = np.mean(pairs[:, 0], axis=0)
        centroid_tgt = np.mean(pairs[:, 1], axis=0)
        p_src_i = (pairs[:, 0] - centroid_src).T
        p_src_j = (pairs[:, 1] - centroid_src).T
        u, s, v = np.linalg.svd(p_src_i@p_src_j.T)
        rot_t = v.T@u.T
        translation_t = centroid_tgt - rot@centroid_src
        tgt @= rot_t
        tgt -= translation_t
        rot @= rot_t
        translation += translation_t
        if show:
            for pair in pairs:
                src_pt, tgt_pt = pair
                plt.scatter(*src_pt, c="b")
                plt.scatter(*tgt_pt, c="r")
                plt.plot(*np.transpose([src_pt, tgt_pt]), c='gray')
            plt.show()
        if dist < abs_err or np.abs(dist - prev_dist) < rel_err:
            break
    return [*translation, np.arctan2(rot[1, 0], rot[0, 0])], dist


if __name__ == "__main__":
    import timeit
    from matplotlib import pyplot as plt
    from pickle import load
    def gen_sines(n_pts, dev, x_shift, rot, trans):

        """
        Generates a sine wave and creates a noisy and shifted copy for ICP testing.

        :param n_pts: Number of points to use.
        :param dev: Standard deviation of points (noise). Tends to be from 0 to 1.
        :param x_shift: Shift the noised sine wave (in radians).
        :param rot: Rotation of the noised sine wave (in radians).
        :param trans: Translation of the noised sine wave.
        :return: The pure sine wave and the noised sine wave.
        """

        x = np.linspace(0, 6.28, n_pts)
        y = np.sin(x)  # Pure target
        # Add the noise to the data
        y_shift = np.sin(x + x_shift)
        y_noised = y_shift + np.random.normal(0, dev, n_pts)

        return np.array([x, y]).T, to_world([*trans, rot], np.array([x, y_noised]).T)
    use_sin = False
    if use_sin:
        src, tgt = gen_sines(1000, 0, 0, 0, [1  , 1])
    else:
        with open("./data.pkl", 'rb') as f:
            scans, shifts, odoms = load(f).values()
        src, tgt = scans[50], scans[150]
    start = timeit.default_timer()
    T = iterative_closest_point(src, tgt, max_iterations=1000, tol=1e-7, animate=True)
    print(timeit.default_timer() - start)
    pose = transform_to_pose(T)
    print(pose)
    exit(0)
    print(100 / timeit.timeit(lambda: iterative_closest_point(src, tgt, max_iterations=100, tol= 1e-7), number=100))

    plt.scatter(*tgt.T)
    plt.scatter(*transform_points(T, src).T)
    plt.show()