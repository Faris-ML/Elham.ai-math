#include <iostream>
#include "Node.hpp"
#include "Graph.hpp"
#include <chrono>

int main() {
    // Create variable x = 3
    NodePtr x = std::make_shared<Variable>(3.0, "x");

    // Create constants cons_1 = 2, cons_2 = 6
    NodePtr cons_1 = std::make_shared<Constant>(2.0, "cons_1");
    NodePtr cons_2 = std::make_shared<Constant>(6.0, "cons_2");

    NodePtr y = std::make_shared<Add>(
        std::make_shared<Power>(x, cons_1, "x^2"),
        cons_2,
        "y0"
    );

    for (int i = 1; i <= 500; ++i) {
        std::string node_name = "y" + std::to_string(i);
        y = std::make_shared<Add>(
            std::make_shared<Power>(x, cons_1, "x^2_" + std::to_string(i)),
            y,
            node_name
        );
    }

    // Build computation graph
    Graph graph(y);

    // Forward pass
    double result = graph.forward();
    std::cout << "The forward pass : " << result << std::endl;

    // Measure backward time
    auto start = std::chrono::high_resolution_clock::now();
    graph.backward();
    auto end = std::chrono::high_resolution_clock::now();

    // Compute elapsed time in milliseconds
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Backward elapsed time: " << elapsed.count() << " ms" << std::endl;

    std::cout << "Gradient wrt x: " << x->grad << std::endl;

    return 0;
}
