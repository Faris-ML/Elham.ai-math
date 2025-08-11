#pragma once
#include <map>
#include <memory>
#include "Node.hpp"

class Graph
{
public:
    NodePtr root;
    std::map<std::string, NodePtr> nodes;

    explicit Graph(NodePtr r) : root(std::move(r))
    {
        build(root);
    }

    void build(const NodePtr &n)
    {
        if (!n)
            return;
        if (nodes.count(n->name))
            return;
        nodes[n->name] = n;

        // Discover inputs if it's an Operator/unary_operator
        if (auto op = std::dynamic_pointer_cast<Operator>(n))
        {
            if (op->a)
                build(op->a);
            if (op->b)
                build(op->b);
        }
    }

    Tensor forward() { return root->forward(); }

    void backward()
    {
        // zero grads to shape of each node's value
        for (auto &kv : nodes)
            kv.second->grad = Tensor::like(kv.second->value, 0.0);
        // seed with ones matching root's shape
        Tensor seed = Tensor::like(root->value, 1.0);
        root->backward(seed);
    }
};
