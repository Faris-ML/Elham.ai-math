#pragma once
#include <vector>
#include <cstdint>
#include <cassert>
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <iostream>

enum class Device { CPU, CUDA };

struct Tensor {
    std::vector<int64_t> shape;   // e.g., {m,n} for matrix; {} for scalar
    std::vector<double>  data;    // row-major
    Device device = Device::CPU;  // default CPU

    Tensor() = default;
    Tensor(std::vector<int64_t> s, double fill=0.0, Device dev=Device::CPU)
        : shape(std::move(s)), device(dev)
    {
        int64_t sz = 1;
        for (auto d : shape) { if (d<=0) throw std::runtime_error("Bad shape"); sz *= d; }
        data.assign(sz, fill);
    }

    int64_t size() const {
        int64_t sz=1; for (auto d: shape) sz*=d; return sz;
    }
    bool is_scalar() const { return shape.empty(); }

    double*       ptr()       { return data.data(); }
    const double* ptr() const { return data.data(); }

    static Tensor like(const Tensor& t, double fill=0.0) {
        Tensor out(t.shape, fill, t.device);
        return out;
    }

    static Tensor scalar(double v, Device dev=Device::CPU) {
        Tensor out({}, 0.0, dev);
        out.data = {v};
        return out;
    }

    std::string desc() const {
        std::string s="Tensor("; 
        s += "[";
        for (size_t i=0;i<shape.size();++i) { s+= std::to_string(shape[i]); if(i+1<shape.size()) s+=","; }
        s += "], size="+std::to_string(size())+", dev=";
        s += (device==Device::CPU?"CPU":"CUDA");
        s += ")";
        return s;
    }
};

// basic shape helpers
inline void require_same_shape(const Tensor& a, const Tensor& b, const char* op){
    if (a.shape != b.shape) throw std::runtime_error(std::string(op)+": shape mismatch");
}
inline void require_vec3(const Tensor& a, const char* op){
    if (!(a.shape.size()==1 && a.shape[0]==3)) throw std::runtime_error(std::string(op)+": requires shape (3,)");
}
inline void require_matmul_shapes(const Tensor& A, const Tensor& B, const char* op){
    if (A.shape.size()!=2 || B.shape.size()!=2) throw std::runtime_error(std::string(op)+": need 2D matrices");
    if (A.shape[1]!=B.shape[0]) throw std::runtime_error(std::string(op)+": inner dims mismatch");
}
