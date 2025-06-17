import pycuda.autoinit
import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

map_module = SourceModule(r"""
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

tf_module = SourceModule(r"""
__device__ double2 transform_2d(double2 point, double3 pose, double &c, double &s) {
    return make_double2(point.x * c - point.y * s + pose.x,
                       point.x * s + point.y * c + pose.y);
}

__device__ double3 solve_system(const double H[9], const double3 b)
{
    // H = [ H[0] H[1] H[2] ]
    //     [ H[3] H[4] H[5] ]
    //     [ H[6] H[7] H[8] ]
    // (symmetric: H[1]==H[3], H[2]==H[6], H[5]==H[7])

    // LDLᵀ decomposition
    double L[9] = {0};
    double D[3] = {0};

    // Step 1: LDLᵀ
    L[0] = 1.0;
    D[0] = H[0];

    L[3] = H[3] / D[0];  // = H[1]/D[0]
    L[4] = 1.0;
    D[1] = H[4] - L[3]*L[3]*D[0];

    L[6] = H[6] / D[0];  // = H[2]/D[0]
    L[7] = (H[7] - L[6]*L[3]*D[0]) / D[1];  // = (H[5] - L[6]*L[3]*D[0]) / D[1]
    L[8] = 1.0;
    D[2] = H[8] - L[6]*L[6]*D[0] - L[7]*L[7]*D[1];

    // Step 2: solve L * y = b
    double y0 = b.x;
    double y1 = b.y - L[3]*y0;
    double y2 = b.z - L[6]*y0 - L[7]*y1;

    // Step 3: solve D * z = y
    double z0 = y0 / D[0];
    double z1 = y1 / D[1];
    double z2 = y2 / D[2];

    // Step 4: solve Lᵗ * x = z
    double x2 = z2;
    double x1 = z1 - L[7]*x2;
    double x0 = z0 - L[3]*x1 - L[6]*x2;

    return make_double3(x0, x1, x2);
}

__global__ void update_transform(double2* pts, double2* q1s, double2* q2s, double3* poses, double4* out)
{
    // The program is split as:
    // n_particles amount of blocks, and n_points amount of threads
    __shared__ double3 x;
    __shared__ double H[9];
    __shared__ double3 b;
    __shared__ double total_err, cos_theta, sin_theta;  // remember to reset to zero after iterating once!
    if (threadIdx.x == 0)
    {
        x = poses[blockIdx.x];
        for (auto &n: H) n = 0.0;
        b = make_double3(0, 0, 0);
        total_err = 0;
        cos_theta = cos(x.z);
        sin_theta = sin(x.z);
    }
    __syncthreads();

    unsigned int n_points = blockDim.x;
    unsigned int current_corr_id = n_points*blockIdx.x + threadIdx.x;

    double2 pt = pts[threadIdx.x];  // Only one scan for all blocks, so we access it by thread
    double2 q1 = q1s[current_corr_id];
    double2 q2 = q2s[current_corr_id];

    double2 line = make_double2(q2.x - q1.x, q2.y - q1.y);
    double ll_sq = line.x * line.x + line.y * line.y;
    double2 n = make_double2(-line.y, line.x);
    double n_norm =  sqrt(ll_sq);
    n.x /= n_norm;
    n.y /= n_norm;

    for (int iter = 0; iter < 10; iter++)
    {
        double2 p = transform_2d(pt, x, cos_theta, sin_theta);
        double t = ((p.x - q1.x)*line.x + (p.y - q1.y)*line.y) / ll_sq;
        double2 proj = make_double2(q1.x + t*line.x, q1.y + t*line.y);
        double r = (p.x - proj.x)*n.x + (p.y - proj.y)*n.y;
        atomicAdd(&total_err, r*r);

        double nx = n.x, ny = n.y;
        double mu = (-sin_theta*pt.x - cos_theta*pt.y)*nx + (cos_theta*pt.x - sin_theta*pt.y)*ny;
        atomicAdd(&H[0], nx*nx);
        atomicAdd(&H[1], nx*ny);
        atomicAdd(&H[2], nx*mu);

        atomicAdd(&H[3], ny*nx);
        atomicAdd(&H[4], ny*ny);
        atomicAdd(&H[5], ny*mu);

        atomicAdd(&H[6], mu*nx);
        atomicAdd(&H[7], mu*ny);
        atomicAdd(&H[8], mu*mu);

        atomicAdd(&b.x, -nx*r);
        atomicAdd(&b.y, -ny*r);
        atomicAdd(&b.z, -mu*r);

        __syncthreads();

        if (threadIdx.x == 0)
        {
            double3 dx = solve_system(H, b);
            // double dxarr[3];
            // solve3x3(H, b, dxarr);
            // double3 dx = make_double3(dxarr[0], dxarr[1], dxarr[2]);
            atomicAdd(&x.x, dx.x);
            atomicAdd(&x.y, dx.y);
            atomicAdd(&x.z, dx.z);
            out[blockIdx.x].w = total_err;

            for (double & i : H) i = 0.0;
            b = make_double3(0.0, 0.0, 0.0);
            double norm = dx.x*dx.x + dx.y*dx.y + dx.z*dx.z;
            if (sqrt(norm) < 1e-6)
            {
                total_err = -1.0;
            }else
            {
                total_err = 0;
                sincos(x.z, &sin_theta, &cos_theta);
            }

        }
        __syncthreads();
        if (total_err == -1.0)
        {
            break;
        }
    }

    if (threadIdx.x == 0)
    {
        out[blockIdx.x].x = x.x;
        out[blockIdx.x].y = x.y;
        out[blockIdx.x].z = x.z;
    }
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
    func = map_module.get_function("update_map")
    func(
        d_scan, d_maps, d_poses, np.int32(map_size), np.int32(thick), block=(points, 1, 1), grid=(n_particles, 1)
    )
    cuda.Context.synchronize()
    cuda.memcpy_dtoh(maps, d_maps)
    return maps/255.0


def icp_update_transform(pts: np.ndarray, q1s: np.ndarray, q2s: np.ndarray, poses: np.ndarray):
    """
    Runs the point-to-line ICP kernel on GPU for multiple particles.

    :param pts: (N, 2) array of scan points shared across all particles.
    :param q1s: (P*N, 2) array of start points of line segments (correspondences).
    :param q2s: (P*N, 2) array of end points of line segments (correspondences).
    :param poses: (P, 3) array of [x, y, theta] poses for each particle.
    :param module: Compiled CUDA SourceModule with 'update_transform' kernel.
    :return: Updated poses as (P, 3) array, and ICP residual errors (P,) array.
    """
    pts = pts.astype(np.double)
    q1s = q1s.astype(np.double)
    q2s = q2s.astype(np.double)
    poses = poses.astype(np.double)

    n_points = pts.shape[0]
    n_particles = poses.shape[0]

    assert q1s.shape == (n_particles * n_points, 2)
    assert q2s.shape == (n_particles * n_points, 2)

    # Allocate and upload inputs
    d_pts = cuda.mem_alloc(pts.nbytes)
    d_q1s = cuda.mem_alloc(q1s.nbytes)
    d_q2s = cuda.mem_alloc(q2s.nbytes)
    d_poses = cuda.mem_alloc(poses.nbytes)
    d_out = cuda.mem_alloc(n_particles * 4 * np.double().nbytes)  # 3 for pose + 1 for error

    cuda.memcpy_htod(d_pts, pts)
    cuda.memcpy_htod(d_q1s, q1s)
    cuda.memcpy_htod(d_q2s, q2s)
    cuda.memcpy_htod(d_poses, poses)

    # Call kernel
    func = tf_module.get_function("update_transform")
    func(
        d_pts, d_q1s, d_q2s, d_poses, d_out,
        block=(n_points, 1, 1), grid=(n_particles, 1)
    )

    # Retrieve updated poses and errors
    out = np.empty((n_particles, 4), dtype=np.double)
    cuda.Context.synchronize()
    cuda.memcpy_dtoh(out, d_out)

    updated_poses = out[:, :3]
    icp_errors = out[:, 3]

    return updated_poses, icp_errors


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