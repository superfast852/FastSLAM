#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "icp.cpp"       // Includes update_transform

namespace py = pybind11;

PYBIND11_MODULE(pl_qcqp, m) {
    m.def("update_transform", &update_transform,
          py::arg("pt"), py::arg("q1s"),
          py::arg("q2s"), py::arg("max_iter"),
          "C++ implementation of the update_transform function.");
}
