#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <iostream>

#include "utils.hpp"
#include "lapjv/lapjv_eigen.cpp"
#include "gibbs_multisensor.hpp"
#include "gm_adaptive_birth.hpp"
#include "mc_adaptive_birth.hpp"
#include "run_filter.hpp"
#include "meanShift.hpp"
#include "kmeans.hpp"

namespace py = pybind11;

PYBIND11_MODULE(cppmsglmb, m) {
    m.def("gibbs_multisensor_approx_cheap", &gibbs_multisensor_approx_cheap);
    m.def("gibbs_multisensor_approx_dprobsample", &gibbs_multisensor_approx_dprobsample);
    m.def("multisensor_lapjv", &multisensor_lapjv);

    // Multi-sensor Joint Adaptive Birth Sampler
    py::class_<AdaptiveBirth>(m, "AdaptiveBirth")
        .def(py::init<>())
        .def(py::init<int>())
        .def("sample_adaptive_birth", &AdaptiveBirth::sample_adaptive_birth)
        .def("init_parameters", &AdaptiveBirth::init_parameters);
    py::class_<MCAdaptiveBirth>(m, "MCAdaptiveBirth")
        .def(py::init<>())
        .def(py::init<vector<MatrixXd>, int, int, double, double, double>())
        .def("setPriorParams", &MCAdaptiveBirth::setPriorParams)
        .def("setBirthProb", &MCAdaptiveBirth::setBirthProb)
        .def("setNumSensorDetect", &MCAdaptiveBirth::setNumSensorDetect)
        .def("setMeasureNoise", &MCAdaptiveBirth::setMeasureNoise)
        .def("sample_adaptive_birth", &MCAdaptiveBirth::sample_adaptive_birth)
        .def("sample_mc_sols", &MCAdaptiveBirth::sample_mc_sols);

    py::class_<KMeans>(m, "KMeans")
        .def(py::init<int>())
        .def("run", &KMeans::run);

    py::class_<MSGLMB>(m, "MSGLMB")
        .def(py::init<vector<MatrixXd>, string, int, bool>())
        .def("run_msglmb_ukf", &MSGLMB::run_msglmb_ukf);

    // Bounding boxes overlap
    m.def("bboxes_ioi_xyah_back2front_all", &bboxes_ioi_xyah_back2front_all);
    m.def("bboxes_ioi_xyah_back2front_all_v2", &bboxes_ioi_xyah_back2front_all_v2);

    // Mean Shift, Mode Seeking, and Clustering
    m.def("meanShift", &meanShift);
}
