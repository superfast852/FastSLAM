import cv2
import numpy as np
from numba import jit, prange
from mapping import OldMap


@jit(nopython=True, parallel=True, fastmath=True)
def multicast_ray_optimized(rx, ry, theta: float = 0.00, n_rays=360, max_distance=500,
                                   map_array=np.ones((10, 10)), noise_sigma=0.00, collision_bound=0.8):
    location = np.array([rx, ry], dtype=np.float32)

    # Generate rays in strictly descending order (2π to 0)
    base_rays = np.linspace(2 * np.pi, 0, n_rays)

    # Add noise while preserving the order
    if noise_sigma > 0:
        # Add noise but keep the ordering
        noise = np.random.normal(0, noise_sigma, n_rays)
        # Ensure noise doesn't cause angles to cross each other
        max_noise = np.diff(base_rays)[0] * 0.49  # Use 49% of the gap between angles
        rays = base_rays + np.clip(noise, -max_noise, max_noise)
    else:
        rays = base_rays

    # Pre-allocate the output array
    scan = np.zeros((n_rays, 2), dtype=np.float32)

    # Pre-calculate map boundaries for faster boundary checks
    max_x = map_array.shape[1] - 1
    max_y = map_array.shape[0] - 1

    for i in prange(n_rays):
        angle = rays[i]

        # Pre-calculate trig functions once
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        # Use direct ray casting approach for better performance
        for step in range(max_distance):
            # Calculate position directly
            x = location[0] + step * cos_angle
            y = location[1] + step * sin_angle

            # Clip and round in one step
            x_int = min(max(int(round(x)), 0), max_x)
            y_int = min(max(int(round(y)), 0), max_y)

            # Check for collision
            if (x_int == 0 or y_int == 0 or x_int == max_x or y_int == max_y or
                    map_array[y_int, x_int] >= collision_bound):
                # Calculate distance directly
                distance = np.sqrt((x - location[0]) ** 2 + (y - location[1]) ** 2)

                # Store the original angle (before theta shift)
                # This ensures angles remain in descending order from 2π to 0
                ray_angle = rays[i] - theta

                # Normalize to [0, 2π) if needed
                if ray_angle < 0:
                    ray_angle += 2 * np.pi
                elif ray_angle >= 2 * np.pi:
                    ray_angle -= 2 * np.pi

                scan[i, 0] = distance  + np.random.normal(0, noise_sigma)
                scan[i, 1] = ray_angle
                break

    return scan

class LidarSim:
    def __init__(self, map_obj: OldMap , n_rays=360, noise_sigma=0.00, max_dist=600, **kwargs):
        """
        A lidar handler object that can simulate or communicate with a lidar.
        :param map_obj: If a Map object is given, it assumes sim mode. If a string is given, it's real mode.
        :param n_rays: In sim mode, the amount of rays to cast.
        :param noise_sigma: In sim mode, the noise added to the read angle.
        :param max_dist: In sim mode, the maximum distance the lidar can read
        :param kwargs: arguments to pass to the real lidar
        """
        self.map = map_obj
        self.map_array = self.map.map
        self.n_rays = n_rays
        self.noise_sigma = noise_sigma
        self.max_dist = max_dist
        self.SH, self.SW = self.map_array.shape

    def getScan(self, pose: np.ndarray | None = None, cartesian: bool = True) -> np.ndarray:
        """
        Simulates a 2D lidar scan over the map, given a pose.
        :param pose: The 2D pose of the robot on the map, given as [x, y, theta]
        :param cartesian: Set as true if the returned array should be xy points, false if it should be distance-angle
        :return: The scan of the current position as cartesian points or distance-angle points.
        """

        scan = multicast_ray_optimized(pose[0], pose[1], float(pose[2]), self.n_rays, self.max_dist, self.map_array, self.noise_sigma)
        scan = np.array(sorted(scan, key=lambda x: x[1]))
        if cartesian:
            scan = self.rb_to_xy(scan)
        return scan.copy()

    @staticmethod
    def rb_to_xy(scan: np.ndarray) -> np.ndarray:
        return np.array([scan[:, 0]*np.cos(scan[:, 1]), scan[:, 0]*np.sin(scan[:, 1])]).T

multicast_ray_optimized(0, 0, 1, 1, 1, np.ones((10, 10)), 0.00) # compile the function


class DifferentialDriveRobot:
    def __init__(self, map_obj, init_pose = None, robot_radius=5, wheel_radius=1.0, wheel_base=2.0,
                 max_linear_speed=250, max_angular_speed=np.pi/2,
                 noise_std=0.0, lidar_rays=360, raycast=False):
        """
        Initialize a differential drive robot.

        Args:
            map_obj: Map object from NavStack
            robot_radius: Radius of the robot in pixels
            wheel_radius: Radius of the wheels in arbitrary units
            wheel_base: Distance between wheels in arbitrary units
            max_linear_speed: Maximum linear speed in pixels per step
            max_angular_speed: Maximum angular speed in radians per step
            noise_std: Standard deviation of motion noise
            lidar_rays: Number of rays for the LiDAR sensor
        """
        self.map = map_obj
        self.robot_radius = robot_radius
        self.wheel_radius = wheel_radius
        self.wheel_base = wheel_base
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.noise_std = noise_std

        # Initialize robot state: [x, y, theta]
        # Start in the middle of the map
        center = np.array(self.map.map_center)
        self.pose = np.array([center[0], center[1], 0]) if init_pose is None else init_pose
        print("init pose:", center)
        # Try to find a valid starting position
        while not self.is_valid_position(self.pose[:2]):
            # Try a random position if center is not valid
            random_pos = self.map.getValidPoint()
            self.pose = np.array([random_pos[0], random_pos[1], np.random.uniform(-np.pi, np.pi)])

        # Initialize LiDAR
        self.lidar = LidarSim(self.map, lidar_rays, max_dist=np.hypot(*self.map.map.shape))
        # Initialize visualization
        self.display_map = None
        self.rc = raycast

    def update(self, linear_velocity, angular_velocity, dt=0.1):
        """
        Update robot position and orientation based on control inputs.

        Args:
            linear_velocity: Forward velocity in pixels per second
            angular_velocity: Angular velocity in radians per second
            dt: Time step in seconds

        Returns:
            New pose after update
        """
        # Clamp velocities to max values
        linear_velocity = np.clip(linear_velocity, -self.max_linear_speed, self.max_linear_speed)
        angular_velocity = np.clip(angular_velocity, -self.max_angular_speed, self.max_angular_speed)

        # Add noise
        if linear_velocity != 0:
            linear_velocity += np.random.normal(0, self.noise_std * self.max_linear_speed)
        if angular_velocity != 0:
            angular_velocity += np.random.normal(0, self.noise_std * self.max_angular_speed)

        pose = self.motion_model(np.array([linear_velocity, angular_velocity]), dt)
        if pose[2] < 0:
            pose[2] += 2*np.pi
        pose[2] %= 2 * np.pi
        if self.is_valid_position(pose[:2]):
            # pose[1] = self.map.map.shape[0] - pose[1]
            self.pose = pose

        return self.pose

    def is_valid_position(self, position):
        """
        Check if a position is valid (no collision with obstacles).

        Args:
            position: (x, y) position to check

        Returns:
            True if position is valid, False otherwise
        """
        # Check if position is within map bounds
        x, y = int(position[0]), int(position[1])
        shape = self.map.map.shape

        if x < self.robot_radius or x >= shape[1] - self.robot_radius or \
                y < self.robot_radius or y >= shape[0] - self.robot_radius:
            return False

        # Check simple point collision
        if self.map.map[y, x] == 1:
            return False

        # Check surrounding points (approximating robot radius)
        for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            rx = int(x + self.robot_radius * np.cos(angle))
            ry = int(y + self.robot_radius * np.sin(angle))

            # Make sure we're in bounds
            if rx < 0 or rx >= shape[1] or ry < 0 or ry >= shape[0]:
                return False

            if self.map.map[ry, rx] == 1:
                return False

        return True

    def get_lidar_scan(self, cartesian=True):
        """
        Get LiDAR scan from current position.

        Args:
            cartesian: If True, returns scan in Cartesian coordinates, otherwise polar

        Returns:
            LiDAR scan data
        """
        return self.lidar.getScan(self.pose, cartesian)

    def visualize(self, display=True):
        """
        Visualize the robot and LiDAR scan on the map.

        Args:
            display: If True, display the visualization window

        Returns:
            Visualization image
        """
        # Create a color map for visualization
        self.display_map = self.map.animate(show=False)  # Refresh the map

        # Draw the robot as a circle with orientation line
        center = self.pose.astype(int)[:2]
        cv2.circle(self.display_map, center, self.robot_radius, (0, 0, 255), -1)

        # Draw LiDAR scan
        if self.rc:
            theta = -self.pose[2]
            scan = self.get_lidar_scan(cartesian=True)
            rot_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            scan = scan @ rot_mat
            for ray in scan:
                if ray[0] != 0 or ray[1] != 0:  # Only draw non-zero rays
                    end_point = (int(center[0] + ray[0]), int(center[1] + ray[1]))
                    cv2.line(self.display_map, center, end_point, (0, 255, 0), 1)
        # self.map.posearrow(self.display_map, self.pose, 10, t=2)
        if display:
            cv2.imshow("Robot Simulation", self.display_map)
            cv2.waitKey(1)

        return self.display_map

    def motion_model(self, u: np.ndarray, dt: float):
        # State vector
        X = self.pose.reshape((3, 1))
        theta = X[2][0]
        # Control input
        U = u.reshape((2, 1))

        # Motion model Jacobian
        J = np.array([
            [np.cos(theta), 0],
            [np.sin(theta), 0],
            [0, 1]
        ])

        # Compute new state
        return (X + J @ U * dt).flatten()


class RobotSimulation:
    def __init__(self, map_path="./lmap2.png", robot_radius=5, lidar_rays=360, noise=0.1, raycast=False):
        """
        Initialize the robot simulation.

        Args:
            map_path: Path to a PNG map image
            robot_radius: Radius of the robot in pixels
            lidar_rays: Number of rays for the LiDAR sensor
        """
        # Initialize the map
        try:
            map_img = cv2.threshold(cv2.imread(map_path, cv2.IMREAD_GRAYSCALE), 20, 255, cv2.THRESH_BINARY)[1].astype(np.float32)
            map_img /= 255
            map_img = 1 - map_img
            self.map = OldMap(map_img, 10)
        except Exception as e:
            print(e)
            print("[ERROR] Could not create map from image, falling back to manual...")
            # Create a simple map if loading fails
            self.map = OldMap(800, 10)
            # Add some obstacles
            self.map.map *= 0
            self.map.map[200:300, 200:300] = 1
            self.map.map[400:500, 400:500] = 1
            self.map.map[200:300, 600:700] = 1

        # Initialize the robot
        self.robot = DifferentialDriveRobot(self.map, robot_radius=robot_radius, lidar_rays=lidar_rays, raycast=raycast, noise_std=noise)

        # Simulation parameters
        self.dt = 1/60
        self.running = False

    def step(self, linear_velocity, angular_velocity):
        """
        Step the simulation forward.

        Args:
            linear_velocity: Forward velocity in pixels per second
            angular_velocity: Angular velocity in radians per second

        Returns:
            New robot pose, LiDAR scan
        """
        pose = self.robot.update(linear_velocity, angular_velocity, self.dt)
        scan = self.robot.get_lidar_scan()

        return pose, scan

    def render(self):
        """
        Render the simulation.

        Returns:
            Visualization image
        """
        return self.robot.visualize()

    def run(self, control_function=None, max_steps=1000, cartesian=False, show=True):
        """
        Run the simulation with a control function.

        Args:
            control_function: Function that takes (pose, scan) and returns (linear_vel, angular_vel)
            max_steps: Maximum number of steps to run

        Returns:
            Trajectory of poses
        """
        self.running = True
        trajectory = []

        if control_function is None:
            # Default random walk behavior
            def control_function(pose, scan):
                return np.random.uniform(0, 20), np.random.uniform(-0.5, 0.5)

        for i in range(max_steps):
            if not self.running:
                break

            pose = self.robot.pose.astype(np.float32)
            scan = self.robot.get_lidar_scan(cartesian=cartesian)
            # pose[1] = self.map.map.shape[0] - pose[1]


            linear_vel, angular_vel = control_function(pose, scan)
            self.step(linear_vel, angular_vel)
            if show:
                self.render()

                key = cv2.waitKey(int(self.dt * 1000)) & 0xFF
                if key == 27:  # ESC key
                    self.running = False

        cv2.destroyAllWindows()
        return np.array(trajectory)

    def stop(self):
        """Stop the simulation."""
        self.running = False


def rot_gen():
    return 0, 1

# Define a simple wall-following control function
def wall_following(pose, scan):
    fov = scan[:30] + scan[len(scan) - 30:]
    min_dist = np.min(np.linalg.norm(fov, axis=1))

    # If too close to a wall, turn away
    if min_dist < 250:
        return 50, 5
    else:
        return 1000, 0


# Example usage:
if __name__ == "__main__":
    # Create simulation
    sim = RobotSimulation(raycast=True)

    # Run with wall-following behavior
    trajectory = sim.run(control_function=wall_following)

    print("Simulation complete.")