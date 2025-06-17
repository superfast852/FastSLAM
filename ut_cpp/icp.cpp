#include "eigen3/Eigen/Dense"
#include <iostream>
using namespace Eigen;

Vector4d update_transform(const MatrixXd& pt,
                          const MatrixXd& q1s,
                          const MatrixXd& q2s,
                          int max_iter=10)
{
    Vector3d x(0.0, 0.0, 0.0);
    double total_err = 0;

    for (int iter = 0; iter < 10; ++iter)
    {
        Matrix3d H = Matrix3d::Zero();
        Vector3d b = Vector3d::Zero();
        total_err = 0;

        double cos_theta = cos(x(2));
        double sin_theta = sin(x(2));

        for (int i = 0; i < pt.rows(); ++i)
        {
            // Transform point
            Vector2d p = pt.row(i).transpose();
            Vector2d Tp = Vector2d(
                cos_theta * p(0) - sin_theta * p(1) + x(0),
                sin_theta * p(0) + cos_theta * p(1) + x(1));

            // Line segment
            Vector2d q1 = q1s.row(i).transpose();
            Vector2d q2 = q2s.row(i).transpose();
            Vector2d line = q2 - q1;

            // Projection onto line segment
            double t = (Tp - q1).dot(line) / line.squaredNorm();
            // t = std::max(0.0f, std::min(1.0f, t));
            Vector2d proj = q1 + t * line;

            // Normal direction (perpendicular to line)
            Vector2d n(-line(1), line(0));
            n.normalize();

            // Residual (point-to-line distance)
            double r = (Tp - proj).dot(n);
            total_err += r * r;

            // Jacobian of Tp wrt [tx, ty, θ]
            Matrix<double, 2, 3> J_Tp;
            J_Tp << 1, 0, -sin_theta * p(0) - cos_theta * p(1),
                    0, 1,  cos_theta * p(0) - sin_theta * p(1);

            // Jacobian of residual
            RowVector3d J_r = n.transpose() * J_Tp;
            if (i==0){
                std::cout << J_Tp(0, 2) << " " << J_Tp(1, 2) << " " << J_r(2) << std::endl;
            }

            // Accumulate normal equations
            H += J_r.transpose() * J_r;
            b += -J_r.transpose() * r;


        }
        // Solve for update
        Vector3d dx = H.inverse()*b; // H.ldlt().solve(b);
        x += dx;

        // Optional: check convergence
        if (dx.norm() < 1e-6)
            break;
    }

    Vector4d result;
    result << x(0), x(1), x(2), total_err;
    return result;
}