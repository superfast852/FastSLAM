import pycuda.autoinit
import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

mod = SourceModule("""
__device__ void swap(int& a, int& b) noexcept {
    int temp = a;
    a = b;
    b = temp;
}

__device__ void swap2(int2& a) {
    swap(a.x, a.y);
}

__device__ int2 transform_2d(float2 point, float3 pose) {
    float c, s;
    __sincosf(pose.z, &s, &c);
    int px = __float2int_rn(point.x * c - point.y * s + pose.x);
    int py = __float2int_rn(point.x * s + point.y * c + pose.y);
    return make_int2(px, py);
}

__device__ void mark_pixel_thick(int x, int y, unsigned char *map, int map_size, unsigned char value) {
    // Mark the main pixel and its 4-connected neighbors
    if (x >= 0 && x < map_size && y >= 0 && y < map_size) {
        map[y * map_size + x] = value;
    }
    // Mark neighbors to create thicker lines

    if (x-1 >= 0 && x-1 < map_size && y >= 0 && y < map_size) {
        map[y * map_size + (x-1)] = value;
    }
    if (x+1 >= 0 && x+1 < map_size && y >= 0 && y < map_size) {
        map[y * map_size + (x+1)] = value;
    }
    if (x >= 0 && x < map_size && y-1 >= 0 && y-1 < map_size) {
        map[(y-1) * map_size + x] = value;
    }
    if (x >= 0 && x < map_size && y+1 >= 0 && y+1 < map_size) {
        map[(y+1) * map_size + x] = value;
    }
    //
    if (x+1 >= 0 && x+1 < map_size && y-1 >= 0 && y-1 < map_size)
    {
        map[(y-1)*map_size + (x+1)] = value;
    }
    if (x-1 >= 0 && x-1 < map_size && y-1 >= 0 && y-1 < map_size){
        map[(y-1)*map_size + (x-1)] = value;
    }
    if (x-1 >= 0 && x-1 < map_size && y+1 >= 0 && y+1 < map_size) {
        map[(y+1) * map_size + (x-1)] = value;
    }
    if (x+1 >= 0 && x+1 < map_size && y+1 >= 0 && y+1 < map_size){
        map[(y+1) * map_size + (x+1)] = value;
    }
    //
}

__device__ void bresenham(int2 &pose, int2 &point, unsigned char *map, int map_size, int thick=0) {
    int2 og = point;
    bool isSteep = abs(point.y - pose.y) > abs(point.x - pose.x);
    if (isSteep) {
        swap2(point);
        swap2(pose);
    }
    if (pose.x > point.x) {
        swap(pose.x, point.x);
        swap(pose.y, point.y);
    }
    int dx = point.x - pose.x;
    int dy = abs(point.y - pose.y);
    int err = dx / 2;
    int y = pose.y;
    int ystep = (pose.y < point.y) ? 1 : -1;
    int map_x, map_y;

    for (int x = pose.x; x <= point.x; x++) {
        if (isSteep) {
            map_x = y;
            map_y = x;
        } else {
            map_x = x;
            map_y = y;
        }
        /*
        if (thick) {
            mark_pixel_thick(map_x, map_y, map, map_size, 0);
        } else if (map_x >= 0 && map_x < map_size && map_y >= 0 && map_y < map_size) {
            map[map_y * map_size + map_x] = 0;
        }
        */
        if (map_x >= 0 && map_x < map_size && map_y >= 0 && map_y < map_size) {
            map[map_y * map_size + map_x] = 0;
        }
        err -= dy;
        if (err < 0) {
            y += ystep;
            err += dx;
        }
    }

    // Mark endpoint as obstacle
    if (thick == 1)
    {
        mark_pixel_thick(og.x, og.y, map, map_size, 255);
    } else if (og.x >= 0 && og.x < map_size && og.y >= 0 && og.y < map_size)
    {
        map[og.y * map_size + og.x] = 255;
    }
}

__global__ void update_map(float2 *scan, unsigned char *maps, float3 *poses, int map_size, int thick=0) {
    float3 center = make_float3(map_size / 2, map_size / 2, 0.0f);
    unsigned int curr_particle = blockIdx.x;
    unsigned char *map_ptr = maps + curr_particle * map_size * map_size;
    float3 pose = poses[curr_particle];
    pose.x += center.x;
    pose.y += center.y;
    int2 pose_px = make_int2(static_cast<int>(pose.x), static_cast<int>(pose.y));
    int2 point = transform_2d(scan[threadIdx.x], pose);
    bresenham(pose_px, point, map_ptr, map_size, thick);
}

""")

def save_pgm(map: np.ndarray, filename: str):
    with open(filename, "wb") as f:
        f.write(b"P2\n")
        f.write(f"{map.shape[0]} {map.shape[1]}\n".encode())
        f.write(b"255\n")
        for row in map:
            for val in row:
                f.write(bytes([val]))
                f.write(b" ")
            f.write(b"\n")


def update_map(scan: np.ndarray, maps: np.ndarray, poses: np.ndarray, thick=False):
    """
    Updates a batch of 2D arrays given 1 scan and poses, using Bresenham's algorithm.
    :param scan: a 2D array of x-y points representing the current scan
    :param maps: a set of 2D arrays (from 0 to 1) representing the maps
    :param poses: a set of 3-element float arrays representing the pose to transform the scan to
    :param thick: Enable this to add a border to the obstacle points.
    :return: the updated set of maps in range (0, 1)
    """
    if maps.ndim == 2:
        maps = np.expand_dims(maps, 0)
    if poses.ndim == 1:
        poses = np.expand_dims(poses, 0)

    n_particles = len(poses)
    points = len(scan)
    map_size = maps.shape[1]

    assert scan.shape == (points, 2)
    assert maps.shape == (n_particles, map_size, map_size)
    assert poses.shape == (n_particles, 3)

    scan = scan.astype(np.float32)
    maps = (maps*255.0).astype(np.uint8)
    poses = poses.astype(np.float32)

    d_scan = cuda.mem_alloc(scan.nbytes)
    d_maps = cuda.mem_alloc(maps.nbytes)
    d_poses = cuda.mem_alloc(poses.nbytes)

    cuda.memcpy_htod(d_scan, scan)
    cuda.memcpy_htod(d_maps, maps)
    cuda.memcpy_htod(d_poses, poses)
    func = mod.get_function("update_map")
    func(
        d_scan, d_maps, d_poses, np.int32(map_size), np.int32(thick), block=(points, 1, 1), grid=(n_particles, 1)
    )
    cuda.Context.synchronize()
    cuda.memcpy_dtoh(maps, d_maps)
    return maps/255.0


update_map(np.zeros((3,2)), np.zeros((2, 2)), np.zeros((3,)))


if __name__ == "__main__":
    n_particles = 100
    points = 1024
    map_size = 800
    r = 50
    poses = np.array([[2.0*p, -2.0*p, 2*np.pi/100*p] for p in range(n_particles)])
    scan = np.array([[r*np.cos(theta), r*np.sin(theta)] for theta in np.linspace(0, np.pi, points)])
    maps = np.ones((n_particles, 800, 800), dtype=np.uint8) * 127
    new_maps = update_map(scan, maps, poses, 1)*255

    import cv2
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (800, 800))
    for img in new_maps.astype(np.uint8):
        for i in range(5):
            out.write(img*255)
            cv2.imshow("frame", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    out.release()
    cv2.destroyAllWindows()