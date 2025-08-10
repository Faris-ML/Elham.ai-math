#include "Kernels.hpp"
#include <cmath>

static Tensor binary_ew(const Tensor& a, const Tensor& b,
                        double(*op)(double,double), const char* name)
{
    require_same_shape(a,b,name);
    Tensor out = Tensor::like(a);
    for (int64_t i=0;i<a.size();++i) out.data[i] = op(a.data[i], b.data[i]);
    return out;
}
static Tensor unary_ew(const Tensor& x, double(*op)(double), const char* name)
{
    Tensor out = Tensor::like(x);
    for (int64_t i=0;i<x.size();++i) out.data[i] = op(x.data[i]);
    return out;
}

Tensor ew_add (const Tensor& a, const Tensor& b){ return binary_ew(a,b,[](double x,double y){return x+y;}, "add"); }
Tensor ew_sub (const Tensor& a, const Tensor& b){ return binary_ew(a,b,[](double x,double y){return x-y;}, "sub"); }
Tensor ew_mul (const Tensor& a, const Tensor& b){ return binary_ew(a,b,[](double x,double y){return x*y;}, "mul"); }
Tensor ew_div (const Tensor& a, const Tensor& b){ return binary_ew(a,b,[](double x,double y){return x/y;}, "div"); }
Tensor ew_pow (const Tensor& a, const Tensor& b){ return binary_ew(a,b,[](double x,double y){return std::pow(x,y);}, "pow"); }
Tensor ew_exp (const Tensor& x){ return unary_ew(x,[](double v){return std::exp(v);},"exp"); }
Tensor ew_ln  (const Tensor& x){ return unary_ew(x,[](double v){return std::log(v);},"ln"); }
Tensor ew_sqrt(const Tensor& x){ return unary_ew(x,[](double v){return std::sqrt(v);},"sqrt"); }

Tensor matmul(const Tensor& A, const Tensor& B){
    require_matmul_shapes(A,B,"matmul");
    int64_t m=A.shape[0], k=A.shape[1], n=B.shape[1];
    Tensor C({m,n});
    for(int64_t i=0;i<m;++i){
        for(int64_t j=0;j<n;++j){
            double acc=0.0;
            for(int64_t p=0;p<k;++p){
                acc += A.data[i*k+p] * B.data[p*n+j];
            }
            C.data[i*n+j]=acc;
        }
    }
    return C;
}

Tensor dotvec(const Tensor& a, const Tensor& b){
    if (!(a.shape.size()==1 && b.shape.size()==1 && a.shape[0]==b.shape[0]))
        throw std::runtime_error("dotvec: need same-length vectors");
    double acc=0.0;
    for (int64_t i=0;i<a.shape[0];++i) acc += a.data[i]*b.data[i];
    return Tensor::scalar(acc);
}

Tensor cross3(const Tensor& a, const Tensor& b){
    require_vec3(a,"cross3"); require_vec3(b,"cross3");
    Tensor c({3});
    const double ax=a.data[0], ay=a.data[1], az=a.data[2];
    const double bx=b.data[0], by=b.data[1], bz=b.data[2];
    c.data[0] = ay*bz - az*by;
    c.data[1] = az*bx - ax*bz;
    c.data[2] = ax*by - ay*bx;
    return c;
}
