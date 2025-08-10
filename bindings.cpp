#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>, true)
#include "Node.hpp"
#include "Graph.hpp"
#include "Bindings.hpp"

namespace py = pybind11;

PYBIND11_MODULE(ElhamMath, m) {
    // py::class_<Node, std::shared_ptr<Node>>(m, "Node")
    
    py::class_<Node, PyNode, std::shared_ptr<Node>>(m, "Node")
        .def(py::init<std::string>())
        .def("forward", &Node::forward)
        .def("backward", &Node::backward)
        .def_readwrite("value", &Node::value)
        .def_readwrite("grad", &Node::grad)
        .def_readwrite("name", &Node::name);

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

    py::class_<add, Operator, std::shared_ptr<add>>(m, "add")
        .def(py::init<std::shared_ptr<Node>, std::shared_ptr<Node>, const std::string&>(),
             py::arg("x1"), py::arg("x2"), py::arg("name"))
        .def("forward", &add::forward)
        .def("backward", &add::backward);

    py::class_<multiply, Operator, std::shared_ptr<multiply>>(m, "multiply")
        .def(py::init<std::shared_ptr<Node>, std::shared_ptr<Node>, const std::string&>(),
             py::arg("x1"), py::arg("x2"), py::arg("name"))
        .def("forward", &multiply::forward)
        .def("backward", &multiply::backward);

    py::class_<Graph>(m, "Graph")
        .def(py::init<std::shared_ptr<Node>>(), py::arg("root"))
        .def("forward", &Graph::forward)
        .def("backward", &Graph::backward)
        .def("print_grads", &Graph::printGrads);
    
    py::class_<UnaryOperator, Node, std::shared_ptr<UnaryOperator>>(m, "UnaryOperator");
    
    py::class_<power, Operator, std::shared_ptr<power>>(m, "power")
    .def(py::init<std::shared_ptr<Node>, std::shared_ptr<Node>, const std::string&>(),
         py::arg("x1"), py::arg("x2"), py::arg("name"))
    .def("forward", &power::forward)
    .def("backward", &power::backward);
    
    // ---- ln(x) ----
py::class_<ln, UnaryOperator, std::shared_ptr<ln>>(m, "ln")
    .def(py::init<std::shared_ptr<Node>, const std::string&>(),
         py::arg("x"), py::arg("name") = "")
    .def("forward",  &ln::forward)
    .def("backward", &ln::backward);

// ---- exp(x) ----
py::class_<exp, UnaryOperator, std::shared_ptr<exp>>(m, "exp")
    .def(py::init<std::shared_ptr<Node>, const std::string&>(),
         py::arg("x"), py::arg("name") = "")
    .def("forward",  &exp::forward)
    .def("backward", &exp::backward);

// ---- sqrt(x) ----
py::class_<sqrt, UnaryOperator, std::shared_ptr<sqrt>>(m, "sqrt")
    .def(py::init<std::shared_ptr<Node>, const std::string&>(),
         py::arg("x"), py::arg("name") = "")
    .def("forward",  &sqrt::forward)
    .def("backward", &sqrt::backward);

// ---- log base b: log_b(x) ----
py::class_<log_base, Operator, std::shared_ptr<log_base>>(m, "log_base")
    .def(py::init<std::shared_ptr<Node>, std::shared_ptr<Node>, const std::string&>(),
         py::arg("x"), py::arg("base"), py::arg("name") = "")
    .def("forward",  &log_base::forward)
    .def("backward", &log_base::backward);
py::class_<divide, Operator, std::shared_ptr<divide>>(m, "divide")
    .def(py::init<std::shared_ptr<Node>, std::shared_ptr<Node>, const std::string&>(),
         py::arg("numerator"), py::arg("denominator"), py::arg("name") = "")
    .def("forward",  &divide::forward)
    .def("backward", &divide::backward);

// (optional; consistent with your style)
py::implicitly_convertible<divide, Node>();
    py::implicitly_convertible<UnaryOperator, Node>();
    py::implicitly_convertible<ln,           Node>();
    py::implicitly_convertible<exp,          Node>();
    py::implicitly_convertible<sqrt,         Node>();
    py::implicitly_convertible<log_base,      Node>();
    py::implicitly_convertible<Variable, Node>();
    py::implicitly_convertible<Constant, Node>();
    py::implicitly_convertible<Operator, Node>();
    py::implicitly_convertible<multiply, Node>();
    py::implicitly_convertible<add, Node>();
    py::implicitly_convertible<power, Node>();

}
