#include "Kernels.hpp"
#include <cmath>

static Tensor binary_ew_impl(const Tensor &A, const Tensor &B,
                             double (*op)(double, double), const char *name)
{
    // Output shape via broadcasting
    auto out_shape = broadcast_shape(A.shape, B.shape);
    Tensor out(out_shape);

    // Aligned strides for inputs
    auto a_as = align_strides_for_broadcast(A.shape, A.strides, out_shape);
    auto b_as = align_strides_for_broadcast(B.shape, B.strides, out_shape);

    const int64_t N = out.size();
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int64_t idx = 0; idx < N; ++idx)
    {
        const int64_t ao = offset_from_linear(idx, out_shape, a_as);
        const int64_t bo = offset_from_linear(idx, out_shape, b_as);
        out.data[idx] = op(A.data[ao], B.data[bo]);
    }
    return out;
}

static Tensor unary_ew_impl(const Tensor &X, double (*op)(double), const char *name)
{
    Tensor out(X.shape);
    const int64_t N = X.size();
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int64_t i = 0; i < N; ++i)
        out.data[i] = op(X.data[i]);
    return out;
}

// ---- elementwise ----
Tensor ew_add(const Tensor &a, const Tensor &b)
{
    return binary_ew_impl(a, b, [](double x, double y)
                          { return x + y; }, "add");
}
Tensor ew_sub(const Tensor &a, const Tensor &b)
{
    return binary_ew_impl(a, b, [](double x, double y)
                          { return x - y; }, "sub");
}
Tensor ew_mul(const Tensor &a, const Tensor &b)
{
    return binary_ew_impl(a, b, [](double x, double y)
                          { return x * y; }, "mul");
}
Tensor ew_div(const Tensor &a, const Tensor &b)
{
    return binary_ew_impl(a, b, [](double x, double y)
                          { return x / y; }, "div");
}
Tensor ew_pow(const Tensor &a, const Tensor &b)
{
    return binary_ew_impl(a, b, [](double x, double y)
                          { return std::pow(x, y); }, "pow");
}
Tensor ew_exp(const Tensor &x)
{
    return unary_ew_impl(x, [](double v)
                         { return std::exp(v); }, "exp");
}
Tensor ew_ln(const Tensor &x)
{
    return unary_ew_impl(x, [](double v)
                         { return std::log(v); }, "ln");
}
Tensor ew_sqrt(const Tensor &x)
{
    return unary_ew_impl(x, [](double v)
                         { return std::sqrt(v); }, "sqrt");
}

// ---- reductions for broadcasted grads ----
Tensor reduce_to_shape(const Tensor &src, const std::vector<int64_t> &target_shape)
{
    // Fast path: already same shape
    if (src.shape == target_shape)
        return src;

    Tensor out(target_shape, 0.0);
    // Align target strides against src shape (for offset calc)
    auto t_as = align_strides_for_broadcast(target_shape,
                                            contiguous_strides_for(target_shape),
                                            src.shape);
    const int64_t N = src.size();
    // NOTE: reduction uses atomic adds to avoid race (ok for medium sizes)
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int64_t idx = 0; idx < N; ++idx)
    {
        const int64_t to = offset_from_linear(idx, src.shape, t_as);
#if defined(_OPENMP)
#pragma omp atomic
#endif
        out.data[to] += src.data[idx];
    }
    return out;
}

// ---- matmul (2D) ----
Tensor matmul2d(const Tensor &A, const Tensor &B)
{
    require_matmul_shapes_2d(A, B, "matmul");
    const int64_t m = A.shape[0], k = A.shape[1], n = B.shape[1];
    Tensor C({m, n}, 0.0);
#if defined(_OPENMP)
#pragma omp parallel for collapse(2)
#endif
    for (int64_t i = 0; i < m; ++i)
    {
        for (int64_t j = 0; j < n; ++j)
        {
            double acc = 0.0;
            for (int64_t p = 0; p < k; ++p)
            {
                acc += A.data[i * A.strides[0] + p * A.strides[1]] *
                       B.data[p * B.strides[0] + j * B.strides[1]];
            }
            C.data[i * C.strides[0] + j * C.strides[1]] = acc;
        }
    }
    return C;
}

// ---- dot for 1D ----
Tensor dotvec(const Tensor &a, const Tensor &b)
{
    if (!(a.shape.size() == 1 && b.shape.size() == 1 && a.shape[0] == b.shape[0]))
        throw std::runtime_error("dotvec: need same-length 1D vectors");
    const int64_t k = a.shape[0];
    double acc = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for reduction(+ : acc)
#endif
    for (int64_t i = 0; i < k; ++i)
        acc += a.data[i] * b.data[i];
    return Tensor::scalar(acc);
}

// ---- cross (3,) × (3,) ----
Tensor cross3(const Tensor &a, const Tensor &b)
{
    require_vec3(a, "cross3");
    require_vec3(b, "cross3");
    Tensor c({3});
    const double ax = a.data[0], ay = a.data[1], az = a.data[2];
    const double bx = b.data[0], by = b.data[1], bz = b.data[2];
    c.data[0] = ay * bz - az * by;
    c.data[1] = az * bx - ax * bz;
    c.data[2] = ax * by - ay * bx;
    return c;
}
