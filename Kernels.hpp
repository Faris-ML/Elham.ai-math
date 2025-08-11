#pragma once
#include "Tensor.hpp"

// Elementwise (supports broadcasting of any inputs, including scalars)
Tensor ew_add(const Tensor &a, const Tensor &b); // a + b
Tensor ew_sub(const Tensor &a, const Tensor &b); // a - b
Tensor ew_mul(const Tensor &a, const Tensor &b); // a ⊙ b
Tensor ew_div(const Tensor &a, const Tensor &b); // a / b
Tensor ew_pow(const Tensor &a, const Tensor &b); // a ^ b (elementwise)
Tensor ew_exp(const Tensor &x);
Tensor ew_ln(const Tensor &x); // natural log
Tensor ew_sqrt(const Tensor &x);

// Linear algebra
Tensor matmul2d(const Tensor &A, const Tensor &B); // (m,k)@(k,n)->(m,n)
Tensor dotvec(const Tensor &a, const Tensor &b); // (k,)·(k,)-> scalar
Tensor cross3(const Tensor &a, const Tensor &b); // (3,)×(3,)->(3,)

// Reduction helper (for gradients of broadcasted inputs)
// Reduces 'src' to 'target_shape' by summing over broadcasted axes.
Tensor reduce_to_shape(const Tensor &src, const std::vector<int64_t> &target_shape);
