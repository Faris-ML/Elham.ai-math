#pragma once
#include <map>
#include <vector>
#include <set>
#include "Node.hpp"

class Graph {
public:
    NodePtr root;
    std::map<std::string, NodePtr> nodes;

    Graph(NodePtr r) : root(std::move(r)) {
        buildGraph(root);
    }

    void buildGraph(const NodePtr& node) {
        if (nodes.find(node->name) == nodes.end()) {
            nodes[node->name] = node;
            auto op = std::dynamic_pointer_cast<Operator>(node);
            if (op) {
                buildGraph(op->inp1);
                buildGraph(op->inp2);
            }
        }
    }

    double forward() {
        return root->forward();
    }

    void backward() {
        for (auto& [_, node] : nodes) {
            node->grad = 0;
        }
        root->backward(1.0);
    }

    void printGrads() const {
        for (const auto& [name, node] : nodes) {
            std::cout << name << ": grad=" << node->grad << "\n";
        }
    }
};
