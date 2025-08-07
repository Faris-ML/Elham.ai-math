#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>, true)
#include "Node.hpp"
#include "Graph.hpp"

namespace py = pybind11;

PYBIND11_MODULE(ElhamMath, m) {
    py::class_<Node, std::shared_ptr<Node>>(m, "Node");

    py::class_<Variable, Node, std::shared_ptr<Variable>>(m, "Variable")
        .def(py::init<double, const std::string&>(),
             py::arg("value"), py::arg("name"))
        .def_readwrite("value", &Variable::value)
        .def_readwrite("grad", &Variable::grad)
        .def("forward", &Variable::forward);

    py::class_<Constant, Node, std::shared_ptr<Constant>>(m, "Constant")
        .def(py::init<double, const std::string&>(),
             py::arg("value"), py::arg("name"))
        .def("forward", &Constant::forward);

    py::class_<Operator, Node, std::shared_ptr<Operator>>(m, "Operator");

    py::class_<Add, Operator, std::shared_ptr<Add>>(m, "Add")
        .def(py::init<std::shared_ptr<Node>, std::shared_ptr<Node>, const std::string&>(),
             py::arg("x1"), py::arg("x2"), py::arg("name"))
        .def("forward", &Add::forward)
        .def("backward", &Add::backward);

    py::class_<Multiply, Operator, std::shared_ptr<Multiply>>(m, "Multiply")
        .def(py::init<std::shared_ptr<Node>, std::shared_ptr<Node>, const std::string&>(),
             py::arg("x1"), py::arg("x2"), py::arg("name"))
        .def("forward", &Multiply::forward)
        .def("backward", &Multiply::backward);

    py::class_<Graph>(m, "Graph")
        .def(py::init<std::shared_ptr<Node>>(), py::arg("root"))
        .def("forward", &Graph::forward)
        .def("backward", &Graph::backward)
        .def("print_grads", &Graph::printGrads);
    
    py::class_<Power, Operator, std::shared_ptr<Power>>(m, "Power")
    .def(py::init<std::shared_ptr<Node>, std::shared_ptr<Node>, const std::string&>(),
         py::arg("x1"), py::arg("x2"), py::arg("name"))
    .def("forward", &Power::forward)
    .def("backward", &Power::backward);

    py::implicitly_convertible<Variable, Node>();
    py::implicitly_convertible<Constant, Node>();
    py::implicitly_convertible<Operator, Node>();
    py::implicitly_convertible<Multiply, Node>();
    py::implicitly_convertible<Add, Node>();
    py::implicitly_convertible<Power, Node>();

}
