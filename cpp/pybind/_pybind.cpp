#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <utility>
#include "../matrix/_matrix.hpp"
#include "../cg_method/_cg_method.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_cgpy, m){
    py::class_<Matrix>(m,"Matrix")
        .def(py::init<size_t, size_t>())
        .def(py::init<size_t, size_t, std::vector<double> const &>())
        .def(py::init<Matrix const&>())
        .def(py::init<std::vector<std::vector<double>> const &>())
        .def("__getitem__", [](Matrix &mat, std::pair<size_t, size_t> index) -> double{
	        return mat(index.first, index.second);
	    })
        .def("__repr__", [](Matrix &mat) -> std::vector<double>{
	        return mat.buffer_vector();
	    })
	    .def("__setitem__", [](Matrix &mat, std::pair<size_t, size_t> index, double val){
		    mat(index.first, index.second) = val;
	    })
        .def("__eq__", &Matrix::operator==)
        .def("__add__", &Matrix::operator+)
        .def("__sub__", static_cast<Matrix (Matrix::*)(Matrix const &)>(&Matrix::operator-))
        .def("__neg__", static_cast<Matrix (Matrix::*)()>(&Matrix::operator-))
        .def("__mul__", static_cast<Matrix (Matrix::*)(double const &)>(&Matrix::operator*))
        .def("__matmul__", static_cast<Matrix (Matrix::*)(Matrix const &)>(&Matrix::operator*))
        .def("norm", &Matrix::norm)
        .def_property_readonly("nrow", &Matrix::nrow)
        .def_property_readonly("ncol", &Matrix::ncol)
        .def("tolist", &Matrix::buffer_vector)
        .def("tolist2d", &Matrix::buffer_vector2d)
        ;
    py::class_<linear_CG>(m, "linear_CG")
        .def(py::init<Matrix const &, Matrix const &, Matrix const &>())
        .def(py::init<Matrix const &, Matrix const &, Matrix const &, double const &>())
        .def(py::init<Matrix const &, Matrix const &, Matrix const &, double const &, double const &>())
        .def("solve", &linear_CG::solve)
        ;
    m.def("multiply_naive", &multiply_naive);
    m.def("multiply_tile", &multiply_tile);
}