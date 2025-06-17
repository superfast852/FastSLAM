from pickle import dump, load
from typing import List
from tools import to_world, padMap
import cv2
import numpy as np
from numba import njit, prange
from skimage.morphology import skeletonize
from cu_modules import update_map

pymap = lambda n, func: map(func, n)

class Map:
    paths = []
    # Free: 0
    # Occupied: 1
    # Unknown: 0.5
    # fixed
    def __init__(self, map=800, map_meters=35, confidence_thres=0.9):
        self.map_meters = map_meters
        self.thres = confidence_thres

        if isinstance(map, int):
            # generate a blank IR Map
            self.map = np.ones((map, map), dtype=np.float32)*0.5
        elif isinstance(map, np.ndarray) and map.ndim == 2:
            self.map = map.copy().astype(np.float32)
            if self.map.max() > 1:
                self.map /= 255.0
            if self.map.min() < 0:
                self.map[self.map < 0] = 0.5
            if self.map.shape[0] != self.map.shape[1]:
                map = self.map.copy()
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
                self.map = np.pad(map, ((pad_top, pad_bottom), (pad_left, pad_right)),
                                 mode="constant", constant_values=0.5)
        else:
            print("[NavStack/Map] Map format unidentified.")
            raise ValueError("Map format unidentified.")

        self.map_center = (self.map.shape[0]//2, )*2  # These are all square maps, so no need to worry.
        self.m2px = self.map.shape[0] / self.map_meters
        self.px2m = 1 / self.m2px

    # added
    def toScan(self, meters=True, skel=False, pose=None, max_radius=None):
        bin_map = (self.map > self.thres).astype(int)
        if skel:
            bin_map = skeletonize(bin_map)
        map_points = np.argwhere(bin_map.astype(bool))[:, ::-1]  * (self.px2m if meters else 1)  # [x, y] in pixels

        if pose is not None and max_radius is not None and max_radius != np.inf:
            print("Radius activated!")
            dxdy = map_points - pose[:2] - self.map_center
            dists = np.linalg.norm(dxdy, axis=1)
            map_points = map_points[dists <= max_radius]

        return map_points

    def save(self, name=None):
        if name is None:
            from datetime import datetime
            datetime = datetime.today()
            name = f"{datetime.day}-{datetime.month}-{datetime.year} {datetime.hour}:{datetime.minute}.pkl"
            del datetime
        with open(name, "wb") as f:
            dump(self.map, f)
        print(f"[NavStack/Map] Saved Map as {name}!")

    def isValidPoint(self, point, unknown=False):
        return self.map[point[1], point[0]] == 0 if not unknown else self.map[point[1], point[0]] < self.thres

    def getValidPoint(self, unknown=False) -> tuple:
        free = np.argwhere(self.map == 0) if not unknown else np.argwhere(self.map < self.thres)
        return tuple(free[np.random.randint(0, free.shape[0])])  # flip to get as xy

    def copy(self):
        new_map = Map(self.map.shape[0], self.map_meters, self.thres)
        new_map.map = self.map.copy()
        return new_map

    def __len__(self):
        return len(self.map)

    def __getitem__(self, item):
        if len(item) != 2:
            print("[NavStack/Map] Index of the map must be a point (X, Y).")
            raise IndexError("Index of the map must be a point (X, Y).")
        return self.map[item[1], item[0]]

    def __repr__(self):
        return f"TransientMap({self.map.shape}, {self.map_meters})"

    def tocv2(self, invert=1, img=None):  # oh boy.
        map = self.map.copy() if img is None else img
        map = (map*255).astype(np.uint8)
        if invert:
            map = 255 - map
        return cv2.cvtColor(map, cv2.COLOR_GRAY2BGR)

    def drawPoint(self, img, point, r=2, c=(0, 0, 255), t=2):
        return cv2.circle(img, point, r, c, t)

    def drawPx(self, img, point, c=(0, 0, 255), r=1):
        for a in range(-r, r):
            for b in range(-r, r):
                img[point[1]+a, point[0]+b] = c
        return img

    def drawLine(self, img, line, c=(0, 255, 0), **kwargs):
        return cv2.line(img, *line, c, **kwargs)

    def drawLineOfDots(self, img, line, c=(0, 255, 0)):
        [self.drawLine(img, (line[i], line[i+1]), c=c, thickness=2) for i in range(len(line)-1)]

    def getValidRoute(self, n, unknown=False):
        return [self.getValidPoint(unknown) for _ in range(n)]

    def posearrow(self, pose, r):
        x = r * np.cos(pose[2])
        y = r * np.sin(pose[2])
        return (round(pose[0] - x), round(pose[1] - y)), (round(pose[0] + x), round(pose[1] + y))

    def addPath(self, route: np.ndarray):
        try:
            if route is None:
                return
            int(route[0][0][0])  # check if we can index that deep and the value is a number
            # If that happens, we are sure it's a path
            self.paths.append(route)
        except TypeError:  # If we could not convert to an integer,
            [self.paths.append(i) for i in route]  # It means that route[0][0][0] was an array.
        except IndexError:  # If the probe was not successful, it's invalid.
            print("[NavStack/Map] Empty or Invalid path provided.")
        return

    def animate(self, img=None, pose=None, drawLines=True, arrowLength=20, thickness=5, show="Map"):
        # Pose is expected to be 2 coordinates, which represent a center and a point along a circle.
        if img is None:
            img = self.tocv2()
        if drawLines:
            for path in self.paths:
                try:
                    path = path.tolist()
                except AttributeError:  # Means it's already a list.
                    pass
                if path:
                    cv2.circle(img, path[0][0], 5, (0, 0, 255), -1)
                    cv2.circle(img, path[-1][1], 5, (0, 255, 0), -1)
                    for line in path:
                        cv2.line(img, *line, [211, 85, 186])
        if pose is not None:
            if pose == "center":
                cv2.arrowedLine(img, self.map_center, tuple(pymap(self.map_center, lambda x: x-5)), (0, 0, 255), 3)
            else:
                pt1, pt2 = self.posearrow(pose, arrowLength/2)
                cv2.arrowedLine(img, pt1, pt2, (0, 0, 255), thickness)
        if show:
            cv2.imshow(show, img)
            cv2.waitKey(1)
        else:
            return img

    def binary(self, val=1):
        return self.map.copy().round().astype(int)

    def collision_free(self, a, b) -> bool:
        return self.cf_wrap(self.map, (a[0], a[1]), (b[0], b[1]), self.thres)

    def expand(self, size, val=0.5):  # TODO: work on how expansion should work (scaling, optimal size. etc.)
        self.map = np.pad(self.map, size, "constant", constant_values=val)
        self.map_center = (self.map.shape[0] // 2,) * 2  # These are all square maps, so no need to worry.
        self.m2px = self.map.shape[0] / self.map_meters
        self.px2m = 1 / self.m2px

    @staticmethod
    @njit
    def cf_wrap(map, a, b, thres) -> bool:
        x1, y1, x2, y2 = int(a[0]), int(a[1]), int(b[0]), int(b[1])
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        issteep = abs(y2 - y1) > abs(x2 - x1)
        if issteep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
        deltax = x2 - x1
        deltay = abs(y2 - y1)
        error = int(deltax / 2)
        y = y1
        if y1 < y2:
            ystep = 1
        else:
            ystep = -1
        for x in range(x1, x2 + 1):
            if issteep:
                point = (y, x)
            else:
                point = (x, y)
            if (point[0] < 0 or point[1] < 0) or (point[0] >= map.shape[0] or point[1] >= map.shape[1]):
                return False
            if map[point[1], point[0]] >= thres:  # there is an obstacle
                return False
            error -= deltay
            if error < 0:
                y += ystep
                error += deltax
        return True

    def register(self, pose, scan):
        """
        Registers a new scan at a certain pose into the map. Expects inputs in meters.
        :param pose: The current pose where the scan was taken (m)
        :param scan: The scan taken at that pose (m)
        :return: None
        """

        px_scan = self.m2px * to_world(pose, scan)
        px_pose: np.ndarray = self.m2px * pose[:2]
        update_map(px_scan, self.map, px_pose)

    def _frombatch(self, map):
        self.map = map


class OldMap:
    paths = []
    # Free: 0
    # Occupied: 1
    # Unknown: -1
    def __init__(self, map="random", map_meters=35):
        # The IR Map is just the RRT Map format.
        self.map_meters = None

        if isinstance(map, str):
            if map.lower() == "random":
                # Get a randomly generated map for testing.
                try:
                    self.map = np.load("./Resources/map.npy")  # Formatted as an RRT Map.
                except FileNotFoundError:
                    self.map = np.load("../Resources/map.npy")
                self.map_meters = 35
            elif map.endswith(".pkl"):
                with open(map, "rb") as f:
                    values = load(f)
                if len(values) == 2 and (isinstance(values, tuple) or isinstance(values, list)):
                    meters, bytearr = values
                    if isinstance(bytearr, bytearray):
                        self.fromSlam(bytearr)
                        self.map_meters = meters
                    else:
                        self.__init__(values, meters)
                else:
                    self.__init__(values)
            else:
                print("[NavStack/Map] Could not load map from file.")
                raise ValueError("Could not extract map from .pkl file.")

        elif isinstance(map, bytearray):
            # convert from slam to IR
            # Slam maps are bytearrays that represent the SLAM Calculated map. Higher the pixel value, clearer it is.
            self.fromSlam(map)

        elif isinstance(map, np.ndarray):
            if np.max(map) > 1:
                self.map = self.nb_transient(map.flatten()) if map.ndim == 2 else self.nb_transient(map)
            else:  # If we get an IR Map, we just use it.
                self.map = map

        elif isinstance(map, int):
            # generate a blank IR Map
            self.map = np.ones((map, map), dtype=int)*-1

        else:
            print("[NavStack/Map] Map format unidentified.")
            raise ValueError("Map format unidentified.")


        if self.map_meters is None:
            self.map_meters = map_meters
        if self.map.shape[0] != self.map.shape[1]:
            self.map = padMap(self.map, fill=1)
        self.m2px = self.map.shape[0] / self.map_meters
        self.px2m = 1 / self.m2px
        self.map_center = (self.map.shape[0]//2, )*2  # These are all square maps, so no need to worry.



    def update(self, map):
        self.__init__(map)

    def fromSlam(self, map: bytearray):
        self.map = self.nb_transient(np.array(map)).astype(int)

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def nb_transient(map_array: np.ndarray) -> np.ndarray:
        tol = 1e-2
        len = int(map_array.size ** 0.5)
        map = (map_array.reshape(len, len) - 73) / 255
        for i in prange(map.shape[0]):
            for j in prange(map.shape[1]):
                if abs(map[i, j] - (54 / 255)) < tol:
                    map[i, j] = -1
                else:
                    map[i, j] = np.logical_not(round(map[i, j]))

        #mask = map != -1

        #for i in prange(map.shape[0]):
        #    for j in range(map.shape[1]):
        #        if mask[i, j]:
        #            map[i, j] = np.logical_not(round(map[i, j]))

        return map

    def toSlam(self):
        map = self.map.copy()
        map[map == -1] = 127
        map[map == 0] = 255
        map[map == 1] = 0
        return bytearray(map.astype(np.uint8).flatten().tolist())

    def save(self, name=None):
        if name is None:
            from datetime import datetime
            datetime = datetime.today()
            name = f"{datetime.day}-{datetime.month}-{datetime.year} {datetime.hour}:{datetime.minute}.pkl"
            del datetime
        with open(name, "wb") as f:
            dump(self.map, f)
        print(f"[NavStack/Map] Saved Map as {name}!")

    def isValidPoint(self, point, unknown=False):
        return self.map[point[1], point[0]] == 0 if not unknown else self.map[point[1], point[0]] != 1

    def getValidPoint(self, unknown=False) -> tuple:
        free = np.argwhere(self.map == 0) if not unknown else np.argwhere(self.map != 1)
        return tuple(free[np.random.randint(0, free.shape[0])][::-1])  # flip to get as xy

    def __len__(self):
        return len(self.map)

    def __getitem__(self, item):
        if len(item) != 2:
            print("[NavStack/Map] Index of the map must be a point (X, Y).")
            raise IndexError("Index of the map must be a point (X, Y).")
        return self.map[item[1], item[0]]

    def __repr__(self):
        return f"TransientMap({self.map.shape}, {self.map_meters})"

    def tocv2(self, invert=1, img=None):  # oh boy.
        map = self.map.copy() if img is None else img
        if invert:
            mask = map != -1
            map[mask] = np.logical_not(map[mask])*255
            map[map == -1] = 127
        else:
            map *= 255
            map[map == -255] = 127
        return cv2.cvtColor(map.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    def drawPoint(self, img, point, r=2, c=(0, 0, 255), t=2):
        return cv2.circle(img, point, r, c, t)

    def drawPx(self, img, point, c=(0, 0, 255), r=1):
        for a in range(-r, r):
            for b in range(-r, r):
                img[point[1]+a, point[0]+b] = c
        return img

    def drawLine(self, img, line, c=(0, 255, 0), **kwargs):
        return cv2.line(img, *line, c, **kwargs)

    def drawLineOfDots(self, img, line, c=(0, 255, 0)):
        [self.drawLine(img, (line[i], line[i+1]), c=c, thickness=2) for i in range(len(line)-1)]

    def getValidRoute(self, n, unknown=False):
        return [self.getValidPoint(unknown) for _ in range(n)]

    def posearrow(self, pose, r):
        x = r * np.cos(pose[2])
        y = r * np.sin(pose[2])
        return (round(pose[0] - x), round(pose[1] - y)), (round(pose[0] + x), round(pose[1] + y))

    def addPath(self, route: np.ndarray):
        try:
            if route is None:
                return
            int(route[0][0][0])  # check if we can index that deep and the value is a number
            # If that happens, we are sure it's a path
            self.paths.append(route)
        except TypeError:  # If we could not convert to an integer,
            [self.paths.append(i) for i in route]  # It means that route[0][0][0] was an array.
        except IndexError:  # If the probe was not successful, it's invalid.
            print("[NavStack/Map] Empty or Invalid path provided.")
        return

    def animate(self, img=None, pose=None, drawLines=True, arrowLength=20, thickness=5, show="Map"):
        # Pose is expected to be 2 coordinates, which represent a center and a point along a circle.
        if img is None:
            img = self.tocv2()
        if drawLines:
            for path in self.paths:
                try:
                    path = path.tolist()
                except AttributeError:  # Means it's already a list.
                    pass
                if path:
                    cv2.circle(img, path[0][0], 5, (0, 0, 255), -1)
                    cv2.circle(img, path[-1][1], 5, (0, 255, 0), -1)
                    for line in path:
                        cv2.line(img, *line, [211, 85, 186])
        if pose is not None:
            if pose == "center":
                cv2.arrowedLine(img, self.map_center, tuple(pymap(self.map_center, lambda x: x-5)), (0, 0, 255), 3)
            else:
                pt1, pt2 = self._posearrowext(pose, arrowLength/2)
                cv2.arrowedLine(img, pt1, pt2, (0, 0, 255), thickness)
        if show:
            cv2.imshow(show, img)
            cv2.waitKey(1)
        else:
            return img

    def binary(self, val=1):
        map = self.map.copy()
        map[map == -1] = val
        return val

    def collision_free(self, a, b) -> bool:
        return self.cf_wrap(self.map, (a[0], a[1]), (b[0], b[1]))

    def expand(self, size):  # TODO: work on how expansion should work (scaling, optimal size. etc.)
        self.map = np.pad(self.map, size, "constant", constant_values=-1)
        self.map_center = (self.map.shape[0] // 2,) * 2  # These are all square maps, so no need to worry.
        self.m2px = self.map.shape[0] / self.map_meters
        self.px2m = 1 / self.m2px

    @staticmethod
    @njit
    def cf_wrap(map, a, b) -> bool:
        x1, y1, x2, y2 = int(a[0]), int(a[1]), int(b[0]), int(b[1])
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        issteep = abs(y2 - y1) > abs(x2 - x1)
        if issteep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
        deltax = x2 - x1
        deltay = abs(y2 - y1)
        error = int(deltax / 2)
        y = y1
        if y1 < y2:
            ystep = 1
        else:
            ystep = -1
        for x in range(x1, x2 + 1):
            if issteep:
                point = (y, x)
            else:
                point = (x, y)
            if (point[0] < 0 or point[1] < 0) or (point[0] >= map.shape[0] or point[1] >= map.shape[1]):
                return False
            if map[point[1], point[0]] == 1:  # there is an obstacle
                return False
            error -= deltay
            if error < 0:
                y += ystep
                error += deltax
        return True


def batch_update(scan: np.ndarray, maps: List[Map], poses: np.ndarray[:, 3], thick=False):
    """
        Updates a batch of maps given 1 scan and poses, using Bresenham's algorithm.
        :param scan: a 2D array of x-y points in px representing the current scan
        :param maps: a list of Map objects for each particle.
        :param poses: a set of 3 0-centered px floats representing each particle's pose
        :param thick: Enable this to add a border to the obstacle points.
        :return: None. The maps are updated inplace.
        """
    map_arrs = np.array([map.map for map in maps])  # TODO: this line takes a lot of time! Maybe preallocate memory?
    new_maps = update_map(scan, map_arrs, poses, thick)
    for i, map_obj in enumerate(maps):
        map_obj.map = new_maps[i]