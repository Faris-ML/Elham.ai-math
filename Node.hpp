#pragma once
#include <string>
#include <memory>
#include "Tensor.hpp"
#include "Kernels.hpp"

class Node
{
public:
    std::string name;
    Tensor value;
    Tensor grad;

    explicit Node(std::string n) : name(std::move(n)) {}

    virtual Tensor forward() = 0;
    virtual void backward(const Tensor &upstream) = 0;
    virtual ~Node() = default;
};

using NodePtr = std::shared_ptr<Node>;

class Variable : public Node
{
public:
    Variable(const Tensor &v, const std::string &n) : Node(n)
    {
        value = v;
        grad = Tensor::like(v, 0.0);
    }
    Tensor forward() override { return value; }
    void backward(const Tensor &upstream) override
    {
        if (grad.shape.empty())
            grad = Tensor::like(value, 0.0);
        // accumulate with reduction if broadcasting happened upstream
        Tensor g_like = reduce_to_shape(upstream, value.shape);
        for (int64_t i = 0; i < grad.size(); ++i)
            grad.data[i] += g_like.data[i];
    }
};

class Constant : public Node
{
public:
    Constant(const Tensor &v, const std::string &n) : Node(n)
    {
        value = v;
        grad = Tensor::like(v, 0.0);
    }
    Tensor forward() override { return value; }
    void backward(const Tensor &) override { /* no-op */ }
};

class Operator : public Node
{
public:
    NodePtr a, b; // b may be null for unary
    Operator(NodePtr x, NodePtr y, const std::string &n) : Node(n), a(std::move(x)), b(std::move(y)) {}
};

// ---------- elementwise add ----------
class add : public Operator
{
public:
    using Operator::Operator;
    Tensor forward() override
    {
        value = ew_add(a->forward(), b->forward());
        grad = Tensor::like(value, 0.0);
        return value;
    }
    void backward(const Tensor &g) override
    {
        a->backward(reduce_to_shape(g, a->value.shape));
        b->backward(reduce_to_shape(g, b->value.shape));
    }
};

// ---------- elementwise sub ----------
class sub : public Operator
{
public:
    using Operator::Operator;
    Tensor forward() override
    {
        value = ew_sub(a->forward(), b->forward());
        grad = Tensor::like(value, 0.0);
        return value;
    }
    void backward(const Tensor &g) override
    {
        a->backward(reduce_to_shape(g, a->value.shape));
        // -g for b
        Tensor gm = reduce_to_shape(g, b->value.shape);
        for (auto &v : gm.data)
            v = -v;
        b->backward(gm);
    }
};

// ---------- elementwise multiply ----------
class mul : public Operator
{
public:
    using Operator::Operator;
    Tensor forward() override
    {
        const Tensor Av = a->forward(), Bv = b->forward();
        value = ew_mul(Av, Bv);
        grad = Tensor::like(value, 0.0);
        return value;
    }
    void backward(const Tensor &g) override
    {
        // dA = g ⊙ B ; dB = g ⊙ A, then reduce to input shapes
        Tensor dA = ew_mul(g, b->value);
        Tensor dB = ew_mul(g, a->value);
        a->backward(reduce_to_shape(dA, a->value.shape));
        b->backward(reduce_to_shape(dB, b->value.shape));
    }
};

// ---------- elementwise divide ----------
class divide : public Operator
{
public:
    using Operator::Operator;
    Tensor forward() override
    {
        const Tensor Av = a->forward(), Bv = b->forward();
        value = ew_div(Av, Bv);
        grad = Tensor::like(value, 0.0);
        return value;
    }
    void backward(const Tensor &g) override
    {
        // dA = g / B ; dB = - g ⊙ A / B^2
        Tensor dA = ew_div(g, b->value);
        Tensor dB = ew_div(ew_mul(g, a->value), ew_mul(b->value, b->value));
        for (auto &v : dB.data)
            v = -v;
        a->backward(reduce_to_shape(dA, a->value.shape));
        b->backward(reduce_to_shape(dB, b->value.shape));
    }
};

// ---------- elementwise power: a^b ----------
class power : public Operator
{
public:
    using Operator::Operator;
    Tensor forward() override
    {
        const Tensor Av = a->forward(), Bv = b->forward();
        value = ew_pow(Av, Bv);
        grad = Tensor::like(value, 0.0);
        return value;
    }
    void backward(const Tensor &g) override
    {
        // dy/da = b * a^(b-1) ; dy/db = ln(a) * a^b
        Tensor one(a->value.shape.empty() ? std::vector<int64_t>{} : a->value.shape, 1.0);
        Tensor b_minus_1 = ew_sub(b->value, one);
        Tensor a_bm1 = ew_pow(a->value, b_minus_1);
        Tensor dA = ew_mul(g, ew_mul(b->value, a_bm1));
        Tensor dB = ew_mul(g, ew_mul(ew_ln(a->value), value));
        a->backward(reduce_to_shape(dA, a->value.shape));
        b->backward(reduce_to_shape(dB, b->value.shape));
    }
};

// ---------- UnaryOperator (uses only 'a') ----------
class UnaryOperator : public Operator
{
public:
    explicit UnaryOperator(NodePtr x, const std::string &n) : Operator(std::move(x), nullptr, n) {}
};

// ln(x)
class ln_op : public UnaryOperator
{
public:
    using UnaryOperator::UnaryOperator;
    Tensor forward() override
    {
        value = ew_ln(a->forward());
        grad = Tensor::like(value, 0.0);
        return value;
    }
    void backward(const Tensor &g) override
    {
        Tensor dA = ew_div(g, a->value);
        a->backward(reduce_to_shape(dA, a->value.shape));
    }
};

// exp(x)
class exp_op : public UnaryOperator
{
public:
    using UnaryOperator::UnaryOperator;
    Tensor forward() override
    {
        value = ew_exp(a->forward());
        grad = Tensor::like(value, 0.0);
        return value;
    }
    void backward(const Tensor &g) override
    {
        Tensor dA = ew_mul(g, value);
        a->backward(reduce_to_shape(dA, a->value.shape));
    }
};

// sqrt(x)
class sqrt_op : public UnaryOperator
{
public:
    using UnaryOperator::UnaryOperator;
    Tensor forward() override
    {
        value = ew_sqrt(a->forward());
        grad = Tensor::like(value, 0.0);
        return value;
    }
    void backward(const Tensor &g) override
    {
        // 0.5 / sqrt(x) = 0.5 / value
        Tensor half = Tensor::like(value, 0.0);
        half.data.assign(half.size(), 0.5);
        Tensor dA = ew_div(g, ew_mul(value, half));
        a->backward(reduce_to_shape(dA, a->value.shape));
    }
};

// log_base(x,b) = ln(x)/ln(b)
class log_base : public Operator
{
public:
    using Operator::Operator;
    Tensor forward() override
    {
        value = ew_div(ew_ln(a->forward()), ew_ln(b->forward()));
        grad = Tensor::like(value, 0.0);
        return value;
    }
    void backward(const Tensor &g) override
    {
        // d/dx: 1/(x ln b) ; d/db: -ln(x)/(b (ln b)^2)
        Tensor ln_b = ew_ln(b->value);
        Tensor dxa = ew_div(g, ew_mul(a->value, ln_b));

        Tensor ln_b_sq = ew_mul(ln_b, ln_b);
        Tensor denom = ew_mul(b->value, ln_b_sq);
        Tensor db = ew_div(ew_mul(g, ew_ln(a->value)), denom);
        for (auto &v : db.data)
            v = -v;

        a->backward(reduce_to_shape(dxa, a->value.shape));
        b->backward(reduce_to_shape(db, b->value.shape));
    }
};

// matmul(A,B) 2D
class matmul : public Operator
{
public:
    using Operator::Operator;
    Tensor forward() override
    {
        value = ::matmul2d(a->forward(), b->forward());
        grad = Tensor::like(value, 0.0);
        return value;
    }
    void backward(const Tensor &g) override
    {
        // dA = g @ B^T ; dB = A^T @ g
        const Tensor &A = a->value;
        const Tensor &B = b->value;

        // Build B^T and A^T views (materialize for simplicity)
        Tensor Bt({B.shape[1], B.shape[0]});
        for (int64_t i = 0; i < B.shape[0]; ++i)
            for (int64_t j = 0; j < B.shape[1]; ++j)
                Bt.data[j * Bt.strides[0] + i * Bt.strides[1]] = B.data[i * B.strides[0] + j * B.strides[1]];

        Tensor At({A.shape[1], A.shape[0]});
        for (int64_t i = 0; i < A.shape[0]; ++i)
            for (int64_t j = 0; j < A.shape[1]; ++j)
                At.data[j * At.strides[0] + i * At.strides[1]] = A.data[i * A.strides[0] + j * A.strides[1]];

        a->backward(::matmul2d(g, Bt));
        b->backward(::matmul2d(At, g));
    }
};

// dot(a,b) for 1D → scalar
class dot : public Operator
{
public:
    using Operator::Operator;
    Tensor forward() override
    {
        value = ::dotvec(a->forward(), b->forward()); // scalar {}
        grad = Tensor::like(value, 0.0);
        return value;
    }
    void backward(const Tensor &g) override
    {
        // dA = g * b ; dB = g * a (expand scalar g)
        Tensor gA(b->value.shape, 0.0);
        gA.data.assign(gA.size(), g.data[0]);
        Tensor gB(a->value.shape, 0.0);
        gB.data.assign(gB.size(), g.data[0]);
        a->backward(ew_mul(gA, b->value));
        b->backward(ew_mul(gB, a->value));
    }
};

// cross(a,b) for (3,) → (3,)
class cross : public Operator
{
public:
    using Operator::Operator;
    Tensor forward() override
    {
        value = ::cross3(a->forward(), b->forward());
        grad = Tensor::like(value, 0.0);
        return value;
    }
    void backward(const Tensor &g) override
    {
        // dA = b × g ; dB = g × a
        a->backward(::cross3(b->value, g));
        b->backward(::cross3(g, a->value));
    }
};
