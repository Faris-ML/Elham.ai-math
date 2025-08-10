#pragma once
#include "Tensor.hpp"

// Elementwise
Tensor ew_add (const Tensor& a, const Tensor& b);  // a + b
Tensor ew_sub (const Tensor& a, const Tensor& b);  // a - b
Tensor ew_mul (const Tensor& a, const Tensor& b);  // a ⊙ b
Tensor ew_div (const Tensor& a, const Tensor& b);  // a / b
Tensor ew_pow (const Tensor& a, const Tensor& b);  // a ^ b  (elementwise)
Tensor ew_exp (const Tensor& x);
Tensor ew_ln  (const Tensor& x);                   // natural log
Tensor ew_sqrt(const Tensor& x);

// Reductions / BLAS-ish
Tensor matmul(const Tensor& A, const Tensor& B);   // (m,k)@(k,n)->(m,n)
Tensor dotvec(const Tensor& a, const Tensor& b);   // (k,)·(k,)-> scalar

// Cross product for 3-vectors
Tensor cross3(const Tensor& a, const Tensor& b);   // (3,)×(3,)->(3,)
