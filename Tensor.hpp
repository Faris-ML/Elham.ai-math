#pragma once
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <numeric>
#include <algorithm>
#include <type_traits>

enum class Device
{
    CPU,
    CUDA
};

struct Tensor
{
    std::vector<int64_t> shape;   // e.g., {}, {k}, {m,n}, {b,m,n}, ...
    std::vector<int64_t> strides; // row-major contiguous by default
    std::vector<double> data;     // row-major storage
    Device device = Device::CPU;  // default CPU

    Tensor() = default;
    // 1) canonical: std::vector<int64_t>
    explicit Tensor(std::vector<int64_t> s, double fill = 0.0, Device dev = Device::CPU)
        : shape(std::move(s)), device(dev)
    {
        for (auto d : shape)
            if (d <= 0)
                throw std::runtime_error("Bad shape");
        recompute_strides();
        data.assign(size(), fill);
    }
    static Tensor zeros(std::initializer_list<int64_t> s, Device dev = Device::CPU)
    {
        return Tensor(s, 0.0, dev);
    }
    static Tensor zeros(const std::vector<int64_t> &s, Device dev = Device::CPU)
    {
        return Tensor(s, 0.0, dev);
    }
    // 2) accept any integral vector (e.g., std::vector<pybind11::ssize_t>)
    template <class Int,
              class = std::enable_if_t<std::is_integral<Int>::value && !std::is_same<Int, int64_t>::value>>
    explicit Tensor(const std::vector<Int> &s, double fill = 0.0, Device dev = Device::CPU)
        : Tensor(std::vector<int64_t>(s.begin(), s.end()), fill, dev) {}

    // 3) nice brace-list ctor: Tensor({m,n,k}, fill)
    explicit Tensor(std::initializer_list<int64_t> s, double fill = 0.0, Device dev = Device::CPU)
        : Tensor(std::vector<int64_t>(s), fill, dev) {}
    // Tensor(std::vector<int64_t> s, double fill = 0.0, Device dev = Device::CPU)
    //     : shape(std::move(s)), device(dev)
    // {
    //     for (auto d : shape)
    //         if (d <= 0)
    //             throw std::runtime_error("Bad shape");
    //     recompute_strides();
    //     data.assign(size(), fill);
    // }

    static Tensor like(const Tensor &t, double fill = 0.0)
    {
        Tensor out(t.shape, fill, t.device);
        return out;
    }
    static Tensor scalar(double v, Device dev = Device::CPU)
    {
        Tensor out({}, 0.0, dev);
        out.recompute_strides();
        out.data = {v};
        return out;
    }

    int64_t size() const
    {
        if (shape.empty())
            return 1;
        int64_t sz = 1;
        for (auto d : shape)
            sz *= d;
        return sz;
    }

    bool is_scalar() const { return shape.empty(); }

    void recompute_strides()
    {
        strides.resize(shape.size());
        if (shape.empty())
        {
            strides.clear(); // <— important for scalars
            return;
        }
        int64_t stride = 1;
        for (int i = int(shape.size()) - 1; i >= 0; --i)
        {
            strides[i] = stride;
            stride *= shape[i];
        }
    }

    std::string desc() const
    {
        std::string s = "Tensor([";
        for (size_t i = 0; i < shape.size(); ++i)
        {
            s += std::to_string(shape[i]);
            if (i + 1 < shape.size())
                s += ",";
        }
        s += "], size=" + std::to_string(size()) + ", dev=";
        s += (device == Device::CPU ? "CPU" : "CUDA");
        s += ")";
        return s;
    }
    // is_vector<T>
    template <class T>
    struct is_vector : std::false_type
    {
    };
    template <class T, class A>
    struct is_vector<std::vector<T, A>> : std::true_type
    {
    };

    // Base: arithmetic leaf
    template <class T>
    static typename std::enable_if<std::is_arithmetic<T>::value, void>::type
    infer_shape_rec(const T &, std::vector<int64_t> &)
    {
        // scalar leaf => no dims appended here
    }

    // Vector branch: push size and recurse on first element; check rectangularity
    template <class Vec>
    static typename std::enable_if<is_vector<Vec>::value, void>::type
    infer_shape_rec(const Vec &v, std::vector<int64_t> &out)
    {
        out.push_back(static_cast<int64_t>(v.size()));
        if (v.empty())
            return;
        std::vector<int64_t> first_shape;
        infer_shape_rec(v[0], first_shape);
        for (size_t i = 1; i < v.size(); ++i)
        {
            std::vector<int64_t> si;
            infer_shape_rec(v[i], si);
            if (si != first_shape)
            {
                throw std::runtime_error("Tensor: ragged nested vector (non-rectangular)");
            }
        }
        out.insert(out.end(), first_shape.begin(), first_shape.end());
    }

    // Flatten: arithmetic leaf
    template <class T>
    static typename std::enable_if<std::is_arithmetic<T>::value, void>::type
    flatten_rec(const T &x, std::vector<double> &out)
    {
        out.push_back(static_cast<double>(x));
    }

    // Flatten: vector branch
    template <class Vec>
    static typename std::enable_if<is_vector<Vec>::value, void>::type
    flatten_rec(const Vec &v, std::vector<double> &out)
    {
        for (const auto &e : v)
            flatten_rec(e, out);
    }

    // Assign from nested vector (any depth)
    template <class Nested>
    void assign_from_nested(const Nested &nested, Device dev = Device::CPU)
    {
        device = dev;
        std::vector<int64_t> shp;
        infer_shape_rec(nested, shp);
        // scalar special-case: allow a single number (no outer vector)
        if (shp.empty())
        {
            // If Nested is arithmetic => scalar
            if constexpr (std::is_arithmetic<Nested>::value)
            {
                shape.clear();
                data.assign(1, static_cast<double>(nested));
                return;
            }
            else
            {
                throw std::runtime_error("Tensor: empty shape inferred from non-scalar");
            }
        }
        int64_t need = 1;
        for (auto d : shp)
        {
            if (d <= 0)
                throw std::runtime_error("Tensor: bad dimension inferred");
            need *= d;
        }
        std::vector<double> flat;
        flat.reserve(static_cast<size_t>(need));
        flatten_rec(nested, flat);
        if (static_cast<int64_t>(flat.size()) != need)
        {
            throw std::runtime_error("Tensor: flatten size mismatch");
        }
        shape = std::move(shp);
        data = std::move(flat);
        recompute_strides();
    }

    // -------- Constructors from nested vectors (1D..6D). Extend if needed. --------

    // scalar
    explicit Tensor(double v, Device dev = Device::CPU) : device(dev)
    {
        shape.clear();
        data = {static_cast<double>(v)};
    }
    // 1D
    explicit Tensor(const std::vector<double> &v, Device dev = Device::CPU) { assign_from_nested(v, dev); }
    // 2D
    explicit Tensor(const std::vector<std::vector<double>> &v, Device dev = Device::CPU) { assign_from_nested(v, dev); }
    // 3D
    explicit Tensor(const std::vector<std::vector<std::vector<double>>> &v, Device dev = Device::CPU) { assign_from_nested(v, dev); }
    // 4D
    explicit Tensor(const std::vector<std::vector<std::vector<std::vector<double>>>> &v, Device dev = Device::CPU) { assign_from_nested(v, dev); }
    // 5D
    explicit Tensor(const std::vector<std::vector<std::vector<std::vector<std::vector<double>>>>> &v, Device dev = Device::CPU) { assign_from_nested(v, dev); }
    // 6D
    explicit Tensor(const std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<double>>>>>> &v, Device dev = Device::CPU) { assign_from_nested(v, dev); }

    // Assign later from nested
    template <class Nested>
    void set_nested(const Nested &nested) { assign_from_nested(nested, device); }

    // Export to nested (C++ side). Caller must pass a correctly-typed nested vector to fill.
    // Example: std::vector<std::vector<double>> out; t.to_nested(out);
    template <class NestedOut>
    void to_nested(NestedOut &out) const
    {
        // Compile-time check: NestedOut must be a nested vector of double whose flattened size matches
        // We fill out by recursively reshaping.
        size_t idx = 0;
        build_nested_rec(out, shape, 0, idx);
        if (idx != data.size())
            throw std::runtime_error("Tensor::to_nested: size mismatch");
    }

private:
    // Build nested recursively into an already-typed NestedOut
    template <class NestedOut>
    typename std::enable_if<is_vector<NestedOut>::value, void>::type
    build_nested_rec(NestedOut &out, const std::vector<int64_t> &shp, size_t dim, size_t &idx) const
    {
        if (dim >= shp.size())
        {
            throw std::runtime_error("Tensor::to_nested: dimension overflow");
        }
        const int64_t n = shp[dim];
        out.clear();
        out.resize(static_cast<size_t>(n));
        if (dim + 1 == shp.size())
        {
            // Last dim: fill leaves
            for (int64_t i = 0; i < n; ++i)
            {
                out[static_cast<size_t>(i)] = std::vector<double>();
                auto &leaf = out[static_cast<size_t>(i)];
                leaf.resize(1); // will be replaced immediately
            }
            // But constructing inner vectors like this is clunky; instead we specialize for vector<double>:
        }

        for (int64_t i = 0; i < n; ++i)
        {
            build_nested_rec(out[static_cast<size_t>(i)], shp, dim + 1, idx);
        }
    }

    // Overload for the leaf vector<double>
    void build_nested_rec(std::vector<double> &out, const std::vector<int64_t> &shp, size_t dim, size_t &idx) const
    {
        if (dim + 1 != shp.size())
            throw std::runtime_error("Tensor::to_nested: leaf level mismatch");
        const int64_t n = shp[dim];
        out.resize(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; ++i)
        {
            if (idx >= data.size())
                throw std::runtime_error("Tensor::to_nested: flat index overflow");
            out[static_cast<size_t>(i)] = data[idx++];
        }
    }
};

// ---------- shape helpers ----------
inline void require_vec3(const Tensor &a, const char *op)
{
    if (!(a.shape.size() == 1 && a.shape[0] == 3))
        throw std::runtime_error(std::string(op) + ": requires shape (3,)");
}

inline void require_matmul_shapes_2d(const Tensor &A, const Tensor &B, const char *op)
{
    if (A.shape.size() != 2 || B.shape.size() != 2)
        throw std::runtime_error(std::string(op) + ": need 2D matrices");
    if (A.shape[1] != B.shape[0])
        throw std::runtime_error(std::string(op) + ": inner dims mismatch");
}

// Broadcast two shapes (NumPy-style). Align from the right.
inline std::vector<int64_t> broadcast_shape(const std::vector<int64_t> &a,
                                            const std::vector<int64_t> &b)
{
    size_t na = a.size(), nb = b.size();
    size_t n = std::max(na, nb);
    std::vector<int64_t> out(n, 1);
    for (size_t i = 0; i < n; ++i)
    {
        int64_t da = (i < n - na) ? 1 : a[i - (n - na)];
        int64_t db = (i < n - nb) ? 1 : b[i - (n - nb)];
        if (da == db || da == 1 || db == 1)
            out[i] = std::max(da, db);
        else
            throw std::runtime_error("Broadcast: incompatible shapes");
    }
    return out;
}

// Given a tensor (shape,strides) and a target shape, build aligned strides:
// same rank as target; if original dim=1 => stride 0 (broadcasted).
inline std::vector<int64_t> align_strides_for_broadcast(const std::vector<int64_t> &src_shape,
                                                        const std::vector<int64_t> &src_strides,
                                                        const std::vector<int64_t> &tgt_shape)
{
    size_t ns = src_shape.size(), nt = tgt_shape.size();
    std::vector<int64_t> aligned(nt, 0);
    for (size_t i = 0; i < nt; ++i)
    {
        int64_t ds = (i < nt - ns) ? 1 : src_shape[i - (nt - ns)];
        int64_t ss = (i < nt - ns) ? 0 : src_strides[i - (nt - ns)];
        int64_t dt = tgt_shape[i];
        if (ds == dt)
            aligned[i] = ss;
        else if (ds == 1)
            aligned[i] = 0; // broadcast along this axis
        else
            throw std::runtime_error("Broadcast align: incompatible shapes");
    }
    return aligned;
}

// Row-major contiguous strides for a target shape (used to decode linear idx)
inline std::vector<int64_t> contiguous_strides_for(const std::vector<int64_t> &shape)
{
    std::vector<int64_t> s(shape.size());
    int64_t stride = 1;
    for (int i = int(shape.size()) - 1; i >= 0; --i)
    {
        s[i] = stride;
        stride *= shape[i];
    }
    return s;
}

// Convert linear index -> data offset using aligned strides
inline int64_t offset_from_linear(int64_t linear_idx,
                                  const std::vector<int64_t> &out_shape,
                                  const std::vector<int64_t> &aligned_strides)
{
    if (out_shape.empty())
        return 0; // scalar
    int64_t off = 0;
    int64_t rem = linear_idx;
    const auto out_strides = contiguous_strides_for(out_shape);
    for (size_t i = 0; i < out_shape.size(); ++i)
    {
        const int64_t coord = rem / out_strides[i];
        rem %= out_strides[i];
        off += coord * aligned_strides[i];
    }
    return off;
}
