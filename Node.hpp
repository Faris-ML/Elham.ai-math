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

// ---- elementwise add ----
class add : public Operator {
public:
    using Operator::Operator;
    Tensor forward() override {
        value = ew_add(a->forward(), b->forward());
        grad  = Tensor::like(value, 0.0);
        return value;
    }
    void backward(const Tensor& g) override {
        a->backward(g);
        b->backward(g);
    }
};

// ---- elementwise multiply ----
class mul : public Operator {
public:
    using Operator::Operator;
    Tensor forward() override {
        const Tensor Av=a->forward(), Bv=b->forward();
        value = ew_mul(Av,Bv);
        grad  = Tensor::like(value, 0.0);
        const_cast<Tensor&>(a->value)=Av;
        const_cast<Tensor&>(b->value)=Bv;
        return value;
    }
    void backward(const Tensor& g) override {
        a->backward( ew_mul(g, b->value) );
        b->backward( ew_mul(g, a->value) );
    }
};

// ---- elementwise divide ----
class divide : public Operator {
public:
    using Operator::Operator;
    Tensor forward() override {
        const Tensor Av=a->forward(), Bv=b->forward();
        value = ew_div(Av,Bv);
        grad  = Tensor::like(value, 0.0);
        const_cast<Tensor&>(a->value)=Av;
        const_cast<Tensor&>(b->value)=Bv;
        return value;
    }
    void backward(const Tensor& g) override {
        a->backward( ew_div(g, b->value) );
        // -(g ⊙ A) / (B ⊙ B)
        Tensor num  = ew_mul(g, a->value);
        Tensor denom= ew_mul(b->value, b->value);
        for (int64_t i=0;i<num.size();++i) num.data[i] = -num.data[i];
        b->backward( ew_div(num, denom) );
    }
};

// ---- elementwise power: a^b ----
class power : public Operator {
public:
    using Operator::Operator;
    Tensor forward() override {
        const Tensor Av=a->forward(), Bv=b->forward();
        value = ew_pow(Av,Bv);
        grad  = Tensor::like(value, 0.0);
        const_cast<Tensor&>(a->value)=Av;
        const_cast<Tensor&>(b->value)=Bv;
        return value;
    }
    void backward(const Tensor& g) override {
        // dy/da = b * a^(b-1) ; dy/db = ln(a) * a^b
        Tensor one(a->value.shape, 1.0);
        Tensor b_minus_1 = ew_sub(b->value, one);
        Tensor a_bm1     = ew_pow(a->value, b_minus_1);
        a->backward( ew_mul(g, ew_mul(b->value, a_bm1)) );
        b->backward( ew_mul(g, ew_mul( ew_ln(a->value), value )) );
    }
};

// ---- ln(x) ----
class ln : public Node {
public:
    NodePtr x;
    ln(NodePtr in, const std::string& n): Node(n), x(std::move(in)) {}
    Tensor forward() override {
        const Tensor xv=x->forward();
        value = ew_ln(xv);
        grad  = Tensor::like(value, 0.0);
        const_cast<Tensor&>(x->value)=xv;
        return value;
    }
    void backward(const Tensor& g) override {
        x->backward( ew_div(g, x->value) );
    }
};

// ---- exp(x) ----
class exp : public Node {
public:
    NodePtr x;
    exp(NodePtr in, const std::string& n): Node(n), x(std::move(in)) {}
    Tensor forward() override {
        const Tensor xv=x->forward();
        value = ew_exp(xv);
        grad  = Tensor::like(value, 0.0);
        const_cast<Tensor&>(x->value)=xv;
        return value;
    }
    void backward(const Tensor& g) override {
        x->backward( ew_mul(g, value) );
    }
};

// ---- sqrt(x) ----
class sqrt : public Node {
public:
    NodePtr x;
    sqrt(NodePtr in, const std::string& n): Node(n), x(std::move(in)) {}
    Tensor forward() override {
        const Tensor xv=x->forward();
        value = ew_sqrt(xv);
        grad  = Tensor::like(value, 0.0);
        const_cast<Tensor&>(x->value)=xv;
        return value;
    }
    void backward(const Tensor& g) override {
        // 0.5 / sqrt(x) = 0.5 / value
        Tensor half = Tensor::like(value, 0.0); half.data.assign(half.size(), 0.5);
        x->backward( ew_div(g, ew_mul(value, half)) );
    }
};

// ---- log_base(x, b) = ln(x)/ln(b) ----
class log_base : public Operator {
public:
    using Operator::Operator;
    Tensor forward() override {
        value = ew_div( ew_ln(a->forward()), ew_ln(b->forward()) );
        grad  = Tensor::like(value, 0.0);
        const_cast<Tensor&>(a->value)=a->value; // already set by forward()
        const_cast<Tensor&>(b->value)=b->value;
        return value;
    }
    void backward(const Tensor& g) override {
        // d/dx: 1/(x ln b) ; d/db: -ln(x)/(b (ln b)^2)
        Tensor ln_b  = ew_ln(b->value);
        Tensor dxa   = ew_div( g, ew_mul(a->value, ln_b) );
        a->backward(dxa);

        Tensor ln_b_sq = ew_mul(ln_b, ln_b);
        Tensor denom   = ew_mul(b->value, ln_b_sq);
        Tensor db      = ew_div( ew_mul(g, ew_ln(a->value)), denom );
        for (int64_t i=0;i<db.size();++i) db.data[i] = -db.data[i];
        b->backward(db);
    }
};

// ---- matmul(A,B) 2D ----
class matmul : public Operator {
public:
    using Operator::Operator;
    Tensor forward() override {
        const Tensor Av=a->forward(), Bv=b->forward();
        value = matmul(Av,Bv);
        grad  = Tensor::like(value, 0.0);
        const_cast<Tensor&>(a->value)=Av;
        const_cast<Tensor&>(b->value)=Bv;
        return value;
    }
    void backward(const Tensor& g) override {
        // dA = g @ B^T ; dB = A^T @ g
        Tensor Bt({b->value.shape[1], b->value.shape[0]});
        for(int64_t i=0;i<b->value.shape[0];++i)
          for(int64_t j=0;j<b->value.shape[1];++j)
            Bt.data[j*Bt.shape[1]+i] = b->value.data[i*b->value.shape[1]+j];

        Tensor At({a->value.shape[1], a->value.shape[0]});
        for(int64_t i=0;i<a->value.shape[0];++i)
          for(int64_t j=0;j<a->value.shape[1];++j)
            At.data[j*At.shape[1]+i] = a->value.data[i*a->value.shape[1]+j];

        a->backward( matmul(g, Bt) );
        b->backward( matmul(At, g) );
    }
};

// ---- dot(a,b) vectors -> scalar ----
class dot : public Operator {
public:
    using Operator::Operator;
    Tensor forward() override {
        const Tensor av=a->forward(), bv=b->forward();
        value = dotvec(av,bv);           // scalar {}
        grad  = Tensor::like(value, 0.0);
        const_cast<Tensor&>(a->value)=av;
        const_cast<Tensor&>(b->value)=bv;
        return value;
    }
    void backward(const Tensor& g) override {
        Tensor ga(a->value.shape); ga.data.assign(ga.size(), g.data[0]);
        Tensor gb(b->value.shape); gb.data.assign(gb.size(), g.data[0]);
        a->backward( ew_mul(gb, b->value) );   // g * b
        b->backward( ew_mul(ga, a->value) );   // g * a
    }
};

// ---- cross(a,b) for 3-vectors ----
class cross : public Operator {
public:
    using Operator::Operator;
    Tensor forward() override {
        const Tensor av=a->forward(), bv=b->forward();
        value = cross3(av,bv);     // (3,)
        grad  = Tensor::like(value, 0.0);
        const_cast<Tensor&>(a->value)=av;
        const_cast<Tensor&>(b->value)=bv;
        return value;
    }
    void backward(const Tensor& g) override {
        // dA = b × g ; dB = g × a
        a->backward( cross3(b->value, g) );
        b->backward( cross3(g, a->value) );
    }
};