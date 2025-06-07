from time import time
import cv2
import numpy as np
from mapping import Map, batch_update
from icp import iterative_closest_point as icp, to_world, transform_to_pose
from tools import normalize_angle

np.set_printoptions(suppress=True, precision=6)


class ParticleSystem:
    """Class to efficiently manage all particles using vectorized operations"""

    def __init__(self, num_particles, map_size=400, map_m=20):
        """
        Initialize particle system with vectorized storage

        Args:
            num_particles: Number of particles to maintain
            map_size: Size of occupancy grid in cells (width, height)
            map_m: Map size in meters
        """
        # Store poses as a single NumPy array [x, y, theta] for each particle
        self.poses = np.zeros((num_particles, 3))
        self.weights = np.ones(num_particles) / num_particles
        self.errors = np.zeros(num_particles)
        # Maps are still stored individually (can't easily vectorize heterogeneous objects)
        self.maps = [Map(map_size, map_m) for _ in range(num_particles)]

        # Store map parameters
        self.map_size = map_size
        self.map_m = map_m

    def get_particle(self, index):
        """Get particle data as a tuple (pose, weight, map)"""
        return (self.poses[index], self.weights[index], self.maps[index])

    def set_particle(self, index, pose, weight, map_obj=None):
        """Set particle data"""
        self.poses[index] = pose
        self.weights[index] = weight
        if map_obj is not None:
            self.maps[index] = map_obj

    def copy_particle(self, source_idx, target_idx):
        """Copy a particle from source to target index"""
        self.poses[target_idx] = self.poses[source_idx].copy()
        self.weights[target_idx] = self.weights[source_idx]
        # Deep copy the map
        self.maps[target_idx] = self.maps[source_idx].copy()

    def __getitem__(self, idx):
        return self.get_particle(idx)

# TODO: add smooth initialization
class SLAM:
    def __init__(self, n_particles, map_px, map_m, resample_threshold=0.5, odom_noise=(10, 10, 10),
                 l_occ=0.85,  # Log-odds for occupied cells
                 l_free=-0.4,  # Log-odds for free cells
                 max_laser_dist=np.inf,
                 thick_register=False
                 ):
        """
        Particle Filter SLAM (or GMapping) implementation in python, with some assistance from CUDA.
        :param n_particles: The desired number of particles, or guesses to use.
        :param map_px: The desired size of the map in pixels. Note that the map will always be square.
        :param map_m: The expected size of the world in meters. This affects a lot of things, so make sure it's correct!
        :param resample_threshold:
        :param odom_noise:
        :param l_occ:
        :param l_free:
        :param max_laser_dist:
        :param thick_register: lmao
        """
        self.n_particles = n_particles
        self.map_px = map_px
        self.map_m = map_m
        self.resample_threshold = resample_threshold
        self.map = Map(map_px, map_m)
        self.particles = ParticleSystem(n_particles, map_px, map_m)
        self.best_particle_idx = 0
        self.odom_noise = odom_noise
        self.mld = max_laser_dist
        self.thick = thick_register

        # Coordinate conversion functions (STANDARDIZED)
        self.m2px_scale = map_px / map_m
        self.px2m_scale = map_m / map_px

        # Log-odds parameters
        self.l_occ = l_occ
        self.l_free = l_free
        self.l0 = 0.0  # Prior log-odds (p=0.5)

        self.prev_scan = None
        self.x0_guess = np.zeros(3)

    def m2px(self, x_m, y_m):
        """Convert meters to pixel coordinates"""
        return (x_m * self.m2px_scale, y_m * self.m2px_scale)

    def px2m(self, x_px, y_px):
        """Convert pixel coordinates to meters"""
        return (x_px * self.px2m_scale, y_px * self.px2m_scale)

    def update(self, odom, scan):
        """
        Updates the particle filter with the given odometry and LiDAR scan data.
        :param odom: Expected to be the difference in movement between the current and previous odometry reading ([dx, dy, dtheta])
        :param scan: The 2D lidar scan in cartesian coordinates, as an Nx2-shaped array (in METERS, robot frame)
        :return: The estimated best particle index (reference with the particle array)
        """
        # scan[:, 1] *= -1  # im sorry, WHAT.
        # 1: Predict particle motion (motion model)
        # TODO: reinforce the odometry with ICP
        # if self.prev_scan is not None:
        #     self.x0_guess = transform_to_pose(icp(self.prev_scan, scan))

        self.predict_particle_motion(odom)

        # 2: Update each particle with the scan
        if self.prev_scan is not None:  # to ensure initialization :)
            for particle_idx in range(self.n_particles):
                self.update_particle(particle_idx, scan)
        self.particles.poses[:, 2] = normalize_angle(self.particles.poses[:, 2])
        self.prev_scan = scan.copy()
        self.update_maps(scan)

        # 3: Normalize weights
        total_weight = np.sum(self.particles.weights)
        if total_weight > 0:
            self.particles.weights /= total_weight
        else:
            self.particles.weights = np.ones(self.n_particles) / self.n_particles

        # 4: Check if resampling is needed
        n_eff = 1.0 / np.sum(self.particles.weights ** 2)
        position_std = np.std(np.linalg.norm(self.particles.poses - np.mean(self.particles.poses, axis=0), axis=1))
        print(n_eff, position_std)
        if n_eff < self.n_particles * self.resample_threshold or position_std > 0.02:
            self.resample_particles()

        self.best_particle_idx = np.argmax(self.particles.weights)
        return self.particles.poses[self.best_particle_idx], self.particles.maps[self.best_particle_idx]

    def predict_particle_motion(self, odom):
        """
        Apply motion model with noise to all particles
        Assumes odom is in world coordinates [dx, dy, dtheta]
        """
        dx, dy, dtheta = odom
        n_particles = self.n_particles

        # Extract current orientations
        theta = self.particles.poses[:, 2]

        # Generate motion noise
        sigma_x, sigma_y, sigma_theta = self.odom_noise

        # Motion noise proportional to motion magnitude
        '''
        noise_x = np.random.normal(0, sigma_x * (abs(dx) + abs(dtheta)), n_particles)
        noise_y = np.random.normal(0, sigma_y * (abs(dy) + abs(dtheta)), n_particles)
        noise_theta = np.random.normal(0, sigma_theta * abs(dtheta), n_particles)
        '''
        noise_x = np.random.normal(0, sigma_x * dx ** 2 + sigma_theta * dtheta ** 2, n_particles)
        noise_y = np.random.normal(0, sigma_y * dy ** 2 + sigma_theta * dtheta ** 2, n_particles)
        noise_theta = np.random.normal(0, sigma_theta * dtheta ** 2 + (sigma_x + sigma_y) * (dx ** 2 + dy ** 2), n_particles)

        # If odometry is in robot frame, transform to world frame
        world_dx = dx * np.cos(theta) - dy * np.sin(theta)
        world_dy = dx * np.sin(theta) + dy * np.cos(theta)
        self.particles.poses[:, 0] += world_dx + noise_x
        self.particles.poses[:, 1] += world_dy + noise_y
        self.particles.poses[:, 2] += dtheta + noise_theta

        # Normalize angles to [-pi, pi]
        self.particles.poses[:, 2] = np.arctan2(
            np.sin(self.particles.poses[:, 2]),
            np.cos(self.particles.poses[:, 2])
        )

    def update_particle(self, particle_idx, scan):
        """
        Updates the particle with the given LiDAR scan data using ICP and map updates.

        :param particle_idx: The index of the particle to update.
        :param scan: The 2D lidar scan in cartesian coordinates, as an Nx2-shaped array (METERS, robot frame).
        """
        # Get current particle state
        pose = self.particles.poses[particle_idx].copy()  # [x, y, theta] in meters
        particle_map = self.particles.maps[particle_idx]

        # Get existing map points (if any) in METERS, world frame
        map_points = particle_map.toScan(skel=self.thick, pose=pose, max_radius=self.mld) - self.px2m(
            *particle_map.map_center)  # This should return points in meters, world frame
        if map_points.shape[0] > 10:  # Need sufficient map points for ICP
            try:
                # Use predicted pose as initial guess for ICP
                initial_guess = pose  # ICP will find the correction
                global_scan = to_world(initial_guess, scan)
                pose_correction, icp_error = icp(
                    global_scan,
                    map_points,
                    max_iterations=20,
                    tol=1e-7,
                    return_score=True,
                )
                pose_correction = transform_to_pose(pose_correction)
                pose_correction[2] = normalize_angle(pose_correction[2])

                # Update particle pose
                self.particles.poses[particle_idx] += pose_correction

                # Calculate particle weight based on ICP fit quality
                # Better fit = lower error = higher weight
                self.particles.weights[particle_idx] = 1.0 / (1.0 + icp_error)
                self.particles.errors[particle_idx] = icp_error

            except Exception as e:
                print(f"ICP failed for particle {particle_idx}: {e}")
                self.particles.weights[particle_idx] = 0.00
        else:
            self.particles.weights[particle_idx] = 1/self.n_particles  # No penalty if there's no map yet

    def update_maps(self, scan):
        """
        Updates every particle's map on GPU with the current lidar scan
        """

        poses = self.particles.poses.copy()
        poses[:, :2] *= self.m2px_scale
        scan *= self.m2px_scale
        scan = scan.astype(np.float32)
        batch_update(scan, self.particles.maps, poses, self.thick)  # The transforms are done on-device, so they remain locally consistent

    def resample_particles(self):
        """Resample particles using systematic resampling"""
        print("Resampling...")

        # Create new particle system
        new_particles = ParticleSystem(self.n_particles, self.map_px, self.map_m)

        # Systematic resampling
        cum_weights = np.cumsum(self.particles.weights)
        step = 1.0 / self.n_particles
        r = np.random.uniform(0, step)

        indices = np.zeros(self.n_particles, dtype=int)
        for i in range(self.n_particles):
            u = r + i * step
            indices[i] = np.searchsorted(cum_weights, u)

        # Copy resampled particles
        for new_idx, old_idx in enumerate(indices):
            new_particles.poses[new_idx] = self.particles.poses[old_idx].copy()
            new_particles.maps[new_idx] = self.particles.maps[old_idx].copy()
            new_particles.weights[new_idx] = 1.0 / self.n_particles

        self.particles = new_particles

    def animate(self, show=False):
        best_pose, _, map = self.particles.get_particle(self.best_particle_idx)
        img = map.animate(show=False)
        # Draw all particles
        for p_pose in self.particles.poses:
            px, py = self.m2px(p_pose[0], p_pose[1])
            px_map = int(px + map.map_center[0])
            py_map = int(py + map.map_center[1])
            if 0 <= px_map < img.shape[1] and 0 <= py_map < img.shape[0]:
                cv2.circle(img, (px_map, py_map), 3, (0, 0, 255), -1)
        px, py = self.m2px(*best_pose[:2])
        px_map = int(px + map.map_center[0])
        py_map = int(py + map.map_center[1])
        if 0 <= px_map < img.shape[1] and 0 <= py_map < img.shape[0]:
            cv2.circle(img, (px_map, py_map), 3, (0, 255, 0), -1)
        if show:
            cv2.imshow("Map", img)
            cv2.waitKey(1)
        else:
            return img

    def animate_alpha(self, show=True):
        best_pose, _, map = self.particles.get_particle(self.best_particle_idx)

        # Base map image (BGR)
        img = map.animate(show=False)

        # Create transparent overlay (same size, 3-channel BGR)
        overlay = np.zeros_like(img, dtype=np.uint8)
        alpha_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)

        # Normalize weights to [0, 1] for blending
        weights = self.particles.weights
        max_weight = np.max(weights)
        alpha_values = weights / max_weight if max_weight > 0 else weights

        for i, p_pose in enumerate(self.particles.poses):
            px, py = self.m2px(p_pose[0], p_pose[1])
            px_map = int(px + map.map_center[0])
            py_map = int(py + map.map_center[1])
            if 0 <= px_map < img.shape[1] and 0 <= py_map < img.shape[0]:
                alpha = float(alpha_values[i])
                cv2.circle(overlay, (px_map, py_map), 3, (0, 0, 255), -1)
                cv2.circle(alpha_mask, (px_map, py_map), 3, alpha, -1)

        # Blend overlay using the alpha mask
        alpha_mask = alpha_mask[:, :, np.newaxis]  # HxWx1
        img_blended = (overlay * alpha_mask + img * (1 - alpha_mask)).astype(np.uint8)

        # Draw best particle in solid green
        px, py = self.m2px(*best_pose[:2])
        px_map = int(px + map.map_center[0])
        py_map = int(py + map.map_center[1])
        if 0 <= px_map < img.shape[1] and 0 <= py_map < img.shape[0]:
            cv2.circle(img_blended, (px_map, py_map), 3, (0, 255, 0), -1)
        if show:
            cv2.imshow("Map", img_blended)
            cv2.waitKey(1)
        else:
            return img_blended


if __name__ == "__main__":
    from stream import Stream

    sim = Stream("./data.pkl")
    slam = SLAM(100, 800, 10, thick_register=True)
    while True:
        for i in range(25):
            scan, pose, odom = sim()
            if sim.i >= sim.max:
                break
        if scan is None:
            break
        start = time()
        slam_pose, map = slam.update(odom[0], scan)
        print("Execution time:", 1/(time() - start))
        slam.animate_alpha()

        error = np.linalg.norm(pose[:2] - slam_pose[:2])
        print(f"Position error: {error:.3f}m\n")