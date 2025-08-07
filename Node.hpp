#pragma once
#include <string>
#include <vector>
#include <memory>
#include <cmath>
#include <iostream>

class Node {
public:
    std::string name;
    double value;
    double grad;
    Node(std::string n) : name(std::move(n)), value(0), grad(0) {}
    virtual double forward() = 0;
    virtual void backward(double d) = 0;
    virtual ~Node() = default;
};

using NodePtr = std::shared_ptr<Node>;

class Variable : public Node {
public:
    using Node::Node;
    Variable(double v, const std::string& n) : Node(n) { value = v; }
    double forward() override { return value; }
    void backward(double d) override { grad += d; }
};

class Constant : public Node {
public:
    Constant(double v, const std::string& n) : Node(n) { value = v; }
    double forward() override { return value; }
    void backward(double d) override {}
};

class Operator : public Node {
public:
    using Node::Node;
    NodePtr inp1, inp2;
    Operator(NodePtr x1, NodePtr x2, const std::string& n) : Node(n), inp1(std::move(x1)), inp2(std::move(x2)) {}
};

class Add : public Operator {
public:
    using Operator::Operator;
    Add(NodePtr x1, NodePtr x2, const std::string& n) : Operator(x1, x2, n) {}
    double forward() override {
        value = inp1->forward() + inp2->forward();
        return value;
    }
    void backward(double d) override {
        inp1->backward(d);
        inp2->backward(d);
    }
};

class Multiply : public Operator {
public:
    using Operator::Operator;
    Multiply(NodePtr x1, NodePtr x2, const std::string& n) : Operator(x1, x2, n) {}
    double forward() override {
        value = inp1->forward() * inp2->forward();
        return value;
    }
    void backward(double d) override {
        inp1->backward(d * inp2->value);
        inp2->backward(d * inp1->value);
    }
};

class Power : public Operator {
public:
    using Operator::Operator;
    Power(NodePtr x1, NodePtr x2, const std::string& name) : Operator(x1, x2, name) {}

    double forward() override {
        value = std::pow(inp1->forward(), inp2->forward());
        return value;
    }

    void backward(double d) override {
        double a = inp1->value;
        double b = inp2->value;
        double grad1 = d * b * std::pow(a, b - 1);
        double grad2 = d * std::log(a) * std::pow(a, b);
        inp1->backward(grad1);
        inp2->backward(grad2);
    }
};
