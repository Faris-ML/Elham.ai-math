# ElhamMath/__init__.py
"""
ElhamMath — tensor-based automatic differentiation (Python bindings)

This package re-exports the compiled extension module symbols. For editor
autocompletion and type checking, see the bundled `__init__.pyi` stub.
"""

from __future__ import annotations

# Import everything from the compiled extension.
# If some optional symbols (e.g., Device/Tensor) aren’t exported yet,
# we keep the import robust so the package still imports cleanly.
try:
    from .ElhamMath import (  # type: ignore[attr-defined]
        # core types
        Node, Variable, Constant, Operator, UnaryOperator, Graph,
        # operators (binary)
        add, mul, divide, power, log_base, matmul, dot, cross,sub,
        # operators (unary)
        ln, exp, sqrt,
        # (optional) low-level types if you bound them
        Tensor, Device,
    )
except Exception as _e:  # pragma: no cover
    # Fall back: import only what exists; this helps during iterative builds.
    from .ElhamMath import *  # type: ignore
    # Try to pick up optional names for nicer dir()/__all__
    _maybe = ("Tensor", "Device", "UnaryOperator", "ln", "exp", "sqrt",
              "add", "mul", "divide", "power", "log_base", "matmul", "dot", "cross")
    globals().update({n: globals().get(n) for n in _maybe})

# Public surface
__all__ = [
    name for name in (
        # core
        "Node", "Variable", "Constant", "Operator", "UnaryOperator", "Graph",
        # ops
        "add", "mul", "divide", "power", "log_base", "matmul", "dot", "cross",
        "ln", "exp", "sqrt",
        # optional low-level
        "Tensor", "Device",
    )
    if name in globals()
]

# ---- Light docstrings for better hover help in editors ----

if "Variable" in globals():
    Variable.__doc__ = """Variable(value, name)
A leaf differentiable node.

Parameters
----------
value : Tensor | float
    Initial value (scalar or tensor). If float is given, it’s treated as a scalar tensor.
name : str
    Name for graph bookkeeping.
"""

if "Constant" in globals():
    Constant.__doc__ = """Constant(value, name)
A non-differentiable leaf node (gradient does not propagate into it).

Parameters
----------
value : Tensor | float
name : str
"""

if "add" in globals():
    add.__doc__ = "add(x1, x2, name='') -> Node\nElementwise addition: y = x1 + x2."

if "mul" in globals():
    mul.__doc__ = "mul(x1, x2, name='') -> Node\nElementwise multiplication: y = x1 ⊙ x2."

if "divide" in globals():
    divide.__doc__ = "divide(numerator, denominator, name='') -> Node\nElementwise division: y = num / den."

if "power" in globals():
    power.__doc__ = "power(x, p, name='') -> Node\nElementwise power: y = x ** p."

if "log_base" in globals():
    log_base.__doc__ = "log_base(x, base, name='') -> Node\nElementwise log base change: log_base_b(x) = ln(x)/ln(b)."

if "ln" in globals():
    ln.__doc__ = "ln(x, name='') -> Node\nNatural logarithm (elementwise)."

if "exp" in globals():
    exp.__doc__ = "exp(x, name='') -> Node\nExponential (elementwise)."

if "sqrt" in globals():
    sqrt.__doc__ = "sqrt(x, name='') -> Node\nSquare root (elementwise)."

if "matmul" in globals():
    matmul.__doc__ = "matmul(A, B, name='') -> Node\nMatrix product: (m,k) @ (k,n) -> (m,n)."

if "dot" in globals():
    dot.__doc__ = "dot(a, b, name='') -> Node\nVector dot product -> scalar."

if "cross" in globals():
    cross.__doc__ = "cross(a, b, name='') -> Node\n3D vector cross product (3,) × (3,) -> (3,)."

if "Graph" in globals():
    Graph.__doc__ = """Graph(root)
A computation graph wrapper.

Methods
-------
forward() -> Tensor
backward() -> None
"""

def _prod(shape):
    p = 1
    for d in shape:
        p *= int(d)
    return p

def _to_nested(flat, shape):
    # flat: 1-D list[float]
    # shape: list[int]
    shape = list(shape)
    if not shape:
        return flat[0] if flat else 0.0
    step = _prod(shape[1:]) if len(shape) > 1 else 1
    return [_to_nested(flat[i*step:(i+1)*step], shape[1:])
            for i in range(int(shape[0]))]

# ---------- monkey patches on the pybind11 Tensor ----------
def _tensor_tolist(self):
    # Using list() to detach from the pybind11 view before slicing
    return _to_nested(list(self.data), list(self.shape))

def _tensor_repr(self):
    try:
        nested_preview = _tensor_tolist(self)
    except Exception:
        # Fallback if something goes wrong
        nested_preview = list(self.data)
    return f"Tensor(shape={list(self.shape)}, device={self.device}, data={nested_preview})"

def _tensor_numpy(self):
    # Optional convenience; avoids importing numpy unless called
    import numpy as _np
    return _np.array(_tensor_tolist(self))

Tensor.array = _tensor_tolist
Tensor.__repr__ = _tensor_repr
Tensor.__str__  = _tensor_repr
Tensor.numpy    = _tensor_numpy
Tensor.nested_data = property(_tensor_tolist)
