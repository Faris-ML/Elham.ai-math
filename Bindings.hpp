#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "Node.hpp"

class PyNode : public Node {
public:
    using Node::Node;

    double forward() override {
        PYBIND11_OVERRIDE_PURE(double, Node, forward);
    }

    void backward(double d) override {
        PYBIND11_OVERRIDE_PURE(void, Node, backward, d);
    }
};
