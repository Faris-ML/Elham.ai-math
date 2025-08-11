#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>, true)

#include "Tensor.hpp"
#include "Node.hpp"
#include "Graph.hpp"

namespace py = pybind11;

PYBIND11_MODULE(ElhamMath, m)
{
    // Tensor + Device (simple for now; later you can add NumPy buffer protocol)
    py::enum_<Device>(m, "Device")
        .value("CPU", Device::CPU)
        .value("CUDA", Device::CUDA);
    py::class_<Tensor>(m, "Tensor")
        // nested-list constructors
        .def(py::init<double, Device>(), py::arg("value"), py::arg("device") = Device::CPU)
        .def(py::init<const std::vector<double> &, Device>(),
             py::arg("value"), py::arg("device") = Device::CPU)
        .def(py::init<const std::vector<std::vector<double>> &, Device>(),
             py::arg("value"), py::arg("device") = Device::CPU)
        .def(py::init<const std::vector<std::vector<std::vector<double>>> &, Device>(),
             py::arg("value"), py::arg("device") = Device::CPU)
        .def(py::init<const std::vector<std::vector<std::vector<std::vector<double>>>> &, Device>(),
             py::arg("value"), py::arg("device") = Device::CPU)
        .def_static("scalar", &Tensor::scalar, py::arg("v"), py::arg("device") = Device::CPU)
        .def_readonly("shape", &Tensor::shape)
        .def_readonly("strides", &Tensor::strides)
        .def_readwrite("data", &Tensor::data)
        .def_readwrite("device", &Tensor::device)
        .def("size", &Tensor::size)
        .def("is_scalar", &Tensor::is_scalar)
        // keep "filled tensor" as a static factory so it doesn't steal kwargs
        .def_static("full", [](const std::vector<int64_t> &shape, double fill, Device dev)
                    {
            Tensor t;
            t.shape = shape; t.device = dev; t.recompute_strides();
            t.data.assign(t.size(), fill);
            return t; }, py::arg("shape"), py::arg("fill") = 0.0, py::arg("device") = Device::CPU);

    // Node base (abstract)
    py::class_<Node, std::shared_ptr<Node>>(m, "Node")
        .def_readwrite("name", &Node::name)
        .def_readwrite("value", &Node::value)
        .def_readwrite("grad", &Node::grad);

    // Operator bases
    py::class_<Operator, Node, std::shared_ptr<Operator>>(m, "Operator");
    py::class_<UnaryOperator, Operator, std::shared_ptr<UnaryOperator>>(m, "UnaryOperator");

    // Leaves
    py::class_<Variable, Node, std::shared_ptr<Variable>>(m, "Variable")
        .def(py::init<const Tensor &, const std::string &>(),
             py::arg("value"), py::arg("name"));

    py::class_<Constant, Node, std::shared_ptr<Constant>>(m, "Constant")
        .def(py::init<const Tensor &, const std::string &>(),
             py::arg("value"), py::arg("name"));

    // Binary elementwise ops
    py::class_<add, Operator, std::shared_ptr<add>>(m, "add")
        .def(py::init<std::shared_ptr<Node>, std::shared_ptr<Node>, const std::string &>(),
             py::arg("x1"), py::arg("x2"), py::arg("name") = "");

    py::class_<sub, Operator, std::shared_ptr<sub>>(m, "sub")
        .def(py::init<std::shared_ptr<Node>, std::shared_ptr<Node>, const std::string &>(),
             py::arg("x1"), py::arg("x2"), py::arg("name") = "");

    py::class_<mul, Operator, std::shared_ptr<mul>>(m, "mul")
        .def(py::init<std::shared_ptr<Node>, std::shared_ptr<Node>, const std::string &>(),
             py::arg("x1"), py::arg("x2"), py::arg("name") = "");

    py::class_<divide, Operator, std::shared_ptr<divide>>(m, "divide")
        .def(py::init<std::shared_ptr<Node>, std::shared_ptr<Node>, const std::string &>(),
             py::arg("numerator"), py::arg("denominator"), py::arg("name") = "");

    py::class_<power, Operator, std::shared_ptr<power>>(m, "power")
        .def(py::init<std::shared_ptr<Node>, std::shared_ptr<Node>, const std::string &>(),
             py::arg("x1"), py::arg("x2"), py::arg("name") = "");

    // Unary ops (export names "ln", "exp", "sqrt")
    py::class_<ln_op, UnaryOperator, std::shared_ptr<ln_op>>(m, "ln")
        .def(py::init<std::shared_ptr<Node>, const std::string &>(),
             py::arg("x"), py::arg("name") = "");

    py::class_<exp_op, UnaryOperator, std::shared_ptr<exp_op>>(m, "exp")
        .def(py::init<std::shared_ptr<Node>, const std::string &>(),
             py::arg("x"), py::arg("name") = "");

    py::class_<sqrt_op, UnaryOperator, std::shared_ptr<sqrt_op>>(m, "sqrt")
        .def(py::init<std::shared_ptr<Node>, const std::string &>(),
             py::arg("x"), py::arg("name") = "");

    // Others
    py::class_<log_base, Operator, std::shared_ptr<log_base>>(m, "log_base")
        .def(py::init<std::shared_ptr<Node>, std::shared_ptr<Node>, const std::string &>(),
             py::arg("x"), py::arg("base"), py::arg("name") = "");

    py::class_<matmul, Operator, std::shared_ptr<matmul>>(m, "matmul")
        .def(py::init<std::shared_ptr<Node>, std::shared_ptr<Node>, const std::string &>(),
             py::arg("A"), py::arg("B"), py::arg("name") = "");

    py::class_<dot, Operator, std::shared_ptr<dot>>(m, "dot")
        .def(py::init<std::shared_ptr<Node>, std::shared_ptr<Node>, const std::string &>(),
             py::arg("a"), py::arg("b"), py::arg("name") = "");

    py::class_<cross, Operator, std::shared_ptr<cross>>(m, "cross")
        .def(py::init<std::shared_ptr<Node>, std::shared_ptr<Node>, const std::string &>(),
             py::arg("a"), py::arg("b"), py::arg("name") = "");

    // Graph
    py::class_<Graph>(m, "Graph")
        .def(py::init<std::shared_ptr<Node>>(), py::arg("root"))
        .def("forward", &Graph::forward)
        .def("backward", &Graph::backward);
}
